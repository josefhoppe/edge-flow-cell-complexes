"""
Module to configure and evaluate cell complex detection
"""

import itertools, time
from typing import Iterable, List, Set, Tuple, Dict, Literal
from collections import defaultdict

import numpy as np
import pandas as pd

from scipy.sparse import csc_array

from cell_flower.cell_complex import CellComplex, calc_edges, normalize_cell
from .flow_generator import generate_flows_edge_noise, FlowGeneratorCell
from cell_flower.detection import CellSearchFlowNormalization, FlowPotentialSpanningTree, cell_candidate_search_st, CellCandidateHeuristic, project_flow, score_cells_multiple
from .generator import CellComplexGenerator, TriangulationComplexGenerator

def one_missing_cells(cell: tuple) -> Set[tuple]:
    """
    Gets all cells that can be retrieved by removing one node from `cell`
    """
    return {normalize_cell(cell[:i] + cell[i+1:]) for i in range(len(cell))}

# default seeds
default_flow_seeds = (77367626, 170391664, 185278135, 92926967, 59345847, 40832980, 69907117,
                      258438430, 95881092, 69854436, 152878489, 261025528, 179519147, 5980565,
                      76801101, 187317209, 250694206, 212138740, 100587057, 18805444)

default_alg_seeds = (19356077, 165085303, 6015432, 75970932, 91207037, 108074582, 98227442,
                     189625962, 202713794, 172660770, 234964091, 243157323, 230948579, 166182846,
                     5063053, 212910573, 246681574, 162390301, 34053785, 98954684)

# literal to indicate usage of ground truth cells instead of CellCandidateHeuristic
GROUND_TRUTH = "ground_truth"


def find_combinations(a: tuple, b: tuple) -> List[tuple]:
    """
    Finds all combinations of the given 2-cells.

    can be multiple if there are multiple but distinct overlaps, i.e.:
    Cell a: `(1,2,3,4,5,6)`
    Cell b: `(1,2,4,5)`
    -> (1,2) and (4,5) are shared, the combinations are (2,3,4) and (1,5,6)
    """
    a_edges = set(calc_edges(a))
    b_edges = set(calc_edges(b))
    shared_edges = a_edges.intersection(b_edges)
    if len(shared_edges) > 0:
        combined_edges = a_edges.union(b_edges).difference(shared_edges)
        return find_cycles(combined_edges)
    else:
        return []


def find_cycles(edges: Set[Tuple[int,int]]) -> List[tuple]:
    """
    finds the disjoint cycles from edges.
    edges must not contain any edges not belonging to the cycles.
    `edges` will be modified.
    """
    result = []
    while len(edges) > 0:
        cycle = list(edges.pop())
        while cycle[0] != cycle[-1]:
            possible_edges = [edge for edge in edges if cycle[-1] in edge]
            if len(possible_edges) != 1:
                # either node-joint cycles or path -> nothing to detect here
                return []
            next_edge = possible_edges[0]
            edges.remove(next_edge)
            next_node = next_edge[0] if next_edge[0] != cycle[-1] else next_edge[1]
            cycle.append(next_node)
        result.append(normalize_cell(tuple(cycle[:-1])))
    return result


class CellComplexDetectionExperiment:
    """
    Defines, executes, and evaluates a single configuration of cell complex detection
    """
    __cell_compl: CellComplex
    __alg_seeds: Iterable[int]
    __flow_cache: List[np.ndarray]
    __runs: int
    __flow_counts: Iterable[int]
    __search_method: CellCandidateHeuristic | Literal['ground_truth']
    __n_clusters: int
    __search_flow_norm: CellSearchFlowNormalization
    __num_curls: int
    __approx_epsilon: float
    __cells_per_step: int
    __agg_results: bool
    __results: pd.DataFrame | None = None
    __summarized_results: pd.DataFrame | None = None
    __time_recorded: Dict[int, list] | None = None # num flows -> [ runtime with seed 1, ... ]
    __correct_cells: Dict[int, list] | None = None # num flows -> [ number of correct cells with seed 1, ... ]
    __approx_error: Dict[int, list] | None = None # num flows -> [ error after n with seed 1, ... ]
    __approx_cell_count: Dict[int, list] | None = None # num flows -> [ n to error < epsilon with seed 1, ... ]
    __cell_len_sum: Dict[int, list] | None = None # num flows -> [ sum of cell lengths (~ 1 / sparsity) with seed 1, ... ]
    __inferred_cells: Dict[int, List[list]] | None = None # num flows -> [ list of cells detected with seed 1, ... ]
    __detailed_result_df: pd.DataFrame | None = None # DataFrame with one row per simulated run and cols method, flows, iterations, total_cell_len, cell_candidates, run, approx_error
    __always_opt_cell: bool

    @property
    def cell_compl(self) -> CellComplex:
        """
        The ground-truth cell complex used in this experiment
        """
        return self.__cell_compl

    @property
    def search_method(self) -> CellCandidateHeuristic:
        """
        The `CellCandidateHeuristic` used in this experiment
        """
        return self.__search_method

    @property
    def flows(self) -> List[np.ndarray]:
        """
        Returns the list of flow matrices, one for each run of the algorithm
        """
        return self.__flow_cache

    @property
    def results(self) -> pd.DataFrame:
        """
        If necessary, uses self.run() to obtain results

        Result: Dataframe with index = flow_counts, columns = all possible cycles in cell_complex
        """
        if self.__results is None:
            self.run()
        return self.__results.copy()

    @property
    def summarized_results(self) -> pd.DataFrame:
        """
        If necessary, uses self.run() to obtain results

        Summarizes results by counting correct, one off, and other cells
        """
        if self.__summarized_results is None:
            correct = [cell for cell in self.correct_cells() if cell in self.results.columns]
            one_down_cols = [tpl for tpl in set().union(
                *[one_missing_cells(cell) for cell in correct]) if tpl in self.results.columns]
            one_up_cols = [cell for cell in self.results.columns if len(
                one_missing_cells(cell).intersection(correct)) > 0]

            to_delete = one_down_cols + list(correct) + one_up_cols
            self.__summarized_results = self.results.drop(
                to_delete, axis='columns')

            results: pd.DataFrame = self.results
            self.__summarized_results['correct'] = results[list(correct)].sum(axis=1)
            self.__summarized_results['one_down'] = results[one_down_cols].sum(axis=1)
            self.__summarized_results['one_up'] = results[one_up_cols].sum(axis=1)
            self.__summarized_results['at_most_one_away'] = self.__summarized_results['correct'] + \
                self.__summarized_results['one_down'] + \
                self.__summarized_results['one_up']

            self.__summarized_results.sort_values(max(self.__flow_counts), axis='columns', ascending=False, inplace=True)
        return self.__summarized_results.copy()
    
    @property
    def execution_time(self) -> Dict[int, list]:
        """
        If necessary, uses self.run() to obtain results

        Returns: a dictionary of `number of flows -> [ runtime with seed 1, ... ]`
        """
        if self.__time_recorded is None:
            self.run()
        return { key: list(val) for key, val in self.__time_recorded.items() }
    
    @property
    def correct_cells_found(self) -> Dict[int, list]:
        """
        If necessary, uses self.run() to obtain results

        Returns: a dictionary of `number of flows -> [ number of correct cells with seed 1, ... ]`
        """
        if self.__correct_cells is None:
            self.run()
        return { key: list(val) for key, val in self.__correct_cells.items() }

    @property
    def approx_error(self) -> Dict[int, list]:
        """
        If necessary, uses self.run() to obtain results

        Returns: a dictionary of `number of flows -> [ approx error after n cells with seed 1, ... ]`
        """
        if self.__approx_error is None:
            self.run()
        return { key: list(val) for key, val in self.__approx_error.items() }

    @property
    def approx_cell_count(self) -> Dict[int, list]:
        """
        If necessary, uses self.run() to obtain results

        Returns: a dictionary of `number of flows -> [ number cells for error < epsilon with seed 1, ... ]`
        """
        if self.__approx_cell_count is None:
            self.run()
        return { key: list(val) for key, val in self.__approx_cell_count.items() }

    @property
    def cell_len_sum(self) -> Dict[int, list]:
        """
        If necessary, uses self.run() to obtain results

        Returns: a dictionary of `number of flows -> [ sum of cell lengths (~ 1 / sparsity) with seed 1, ... ]`
        """
        if self.__cell_len_sum is None:
            self.run()
        return { key: list(val) for key, val in self.__cell_len_sum.items() }

    @property
    def inferred_cells(self) -> Dict[int, List[list]]:
        """
        If necessary, uses self.run() to obtain results

        Returns: a dictionary of `number of flows -> [ list of cells detected with seed 1, ... ]`
        """
        if self.__inferred_cells is None:
            self.run()
        return { key: [list(cells) for cells in val] for key, val in self.__inferred_cells.items() }
    
    @property
    def detailed_results(self) -> pd.DataFrame:
        """Get DataFrame with one row per simulated run.
        
        columns: 
        - method
        - flows
        - iterations
        - total_cell_len
        - cell_candidates
        - run
        - approx_error."""
        if self.__detailed_result_df is None:
            self.run()
        return self.__detailed_result_df.copy()
    
    def modified(self, **kwargs) -> 'CellComplexDetectionExperiment':
        """
        Returns a copy with only kwargs changed.
        any keyword from the __init__ function is permitted.

        Also note:
        - flows are only changed if passed explicitly as `flows`. All arguments related to flow generation are ignored.
        """
        kwargs.setdefault('cell_compl', self.__cell_compl)
        kwargs.setdefault('alg_seeds', self.__alg_seeds)
        kwargs.setdefault('flows', self.__flow_cache)
        kwargs.setdefault('flow_counts', self.__flow_counts)
        kwargs.setdefault('runs', self.__runs)
        kwargs.setdefault('search_method', self.__search_method)
        kwargs.setdefault('n_clusters', self.__n_clusters)
        kwargs.setdefault('search_flow_norm', self.__search_flow_norm)
        kwargs.setdefault('num_curls', self.__num_curls)
        kwargs.setdefault('approx_epsilon', self.__approx_epsilon)
        kwargs.setdefault('cells_per_step', self.__cells_per_step)
        kwargs.setdefault('rw_scoring_func', self.__rw_scoring_func)
        kwargs.setdefault('agg_results', self.__agg_results)
        kwargs.setdefault('always_opt_cell', self.__always_opt_cell)

        return CellComplexDetectionExperiment(**kwargs)

    def __init__(self, cell_compl: CellComplex | CellComplexGenerator = TriangulationComplexGenerator(),
                 alg_seeds: Iterable[int] = default_alg_seeds,
                 flow_seeds: Iterable[int] = default_flow_seeds,
                 runs: int = 20, flow_counts: Iterable[int] = tuple(range(1, 21)),
                 curl_flow_sigma: np.float64 = 1, noise_sigma: np.float64 = .5,
                 flows: List[np.ndarray] | None = None,
                 search_method: CellCandidateHeuristic | Literal['ground_truth'] = CellCandidateHeuristic.MAX,
                 n_clusters: int = 11,
                 search_flow_norm: CellSearchFlowNormalization = CellSearchFlowNormalization.LEN,
                 num_curls: int | None = None, approx_epsilon: float = np.inf, cells_per_step: int = 5,
                 agg_results:bool=False, always_opt_cell:bool=False):
        self.__alg_seeds = alg_seeds
        self.__runs = runs
        self.__cell_compl = cell_compl if type(cell_compl) == CellComplex else cell_compl.generate()
        self.__search_method = search_method
        self.__n_clusters = n_clusters
        self.__cells_per_step = cells_per_step
        self.__num_curls = num_curls if num_curls is not None else len(self.__cell_compl.get_cells(2))
        self.__flow_counts = flow_counts
        self.__search_flow_norm = search_flow_norm
        self.__approx_epsilon = approx_epsilon
        self.__agg_results = agg_results
        self.__always_opt_cell = always_opt_cell
        if flows is not None:
            self.__flow_cache = flows
        else:
            self.__flow_cache = []
            flow_cells = [FlowGeneratorCell(
                cell, -curl_flow_sigma, curl_flow_sigma) for cell in self.__cell_compl.get_cells(2)]
            for seed in flow_seeds:
                rnd = np.random.default_rng(seed)
                # Generate flows
                flows = generate_flows_edge_noise(
                    rnd, self.__cell_compl, flow_cells, noise_sigma, np.max(flow_counts))
                flows_matrix = np.stack([f.to_numpy() for f in flows])
                self.__flow_cache.append(flows_matrix)
    
    def correct_cells(self, only_detected: bool = True) -> Set[tuple]:
        """
        Get all correct cells (ground truth or linear combinations)
        """
        correct = set(self.cell_compl.get_cells(2))
        # linear combinations of correct cells are also correct
        for a, b in itertools.combinations(correct, 2):
            correct.update(find_combinations(a, b))
        if only_detected:
            correct = [cell for cell in correct if cell in self.results.columns]
        return correct

    def get_cell_candidates(self, rnd: np.random.Generator | int, current_compl: CellComplex, harmonic_flows: np.ndarray) -> List[Tuple[tuple, csc_array] | Tuple[tuple, csc_array, FlowPotentialSpanningTree]]:
        """
        Finds candidates to add to `current_compl`, guided by `harmonic_flows` and the configured cell search method

        - rnd: either random generator or number of algorithm seed (alg_seeds constructor param)
        """
        if isinstance(rnd, int):
            rnd = np.random.default_rng(seed=self.__alg_seeds[rnd])
        if self.__search_method == CellCandidateHeuristic.TRIANGLES:
            cand_list = [(np.sum(np.abs(boundary.T @ harmonic_flows.T)), cell, boundary) for cell, boundary in current_compl.triangles]
            cand_list.sort(key=lambda x: x[0], reverse=True)
            return [(cell, boundary) for _, cell, boundary in cand_list[:self.__cells_per_step]]
        elif self.__search_method == GROUND_TRUTH:
            b2 = self.__cell_compl.boundary_map(2)
            cell_list = [(cell, b2[:, [i]]) for i, cell in enumerate(self.__cell_compl.get_cells(2)) if cell not in current_compl.get_cells(2)]
            cand_list = [(np.sum(np.abs(boundary.T @ harmonic_flows.T)), cell, boundary) for cell, boundary in cell_list]
            cand_list.sort(key=lambda x: x[0], reverse=True)
            return [(cell, boundary) for _, cell, boundary in cand_list[:self.__cells_per_step]]
        else:
            return cell_candidate_search_st(rnd, current_compl, flows=harmonic_flows, result_count=self.__cells_per_step, method=self.__search_method, flow_norm=self.__search_flow_norm, n_clusters=self.__n_clusters)
        
    def get_harmonic_flows(self, current_compl: CellComplex, seed_num: int, flow_count: int) -> np.ndarray:
        """
        Helper function for further analysis
        """
        flows = self.__flow_cache[seed_num][:flow_count]
        no_gradient_flows = np.copy(flows)
        for i in range(flows.shape[0]):
            no_gradient_flows[i] -= project_flow(
                current_compl.boundary_map(1).T, flows[[i]])
        harmonic_flows = np.copy(no_gradient_flows)
        for j in range(harmonic_flows.shape[0]):
            harmonic_flows[j] -= project_flow(
                current_compl.boundary_map(2), no_gradient_flows[[j]])
        return harmonic_flows

    def simulate_run(self, seed_num: int, flow_count: int) -> tuple[list[tuple], float, int, list[dict]]:
        """
        simulates a single run of the algorithm with the `seed_num`th seed and `flow_count` many flows
        """
        rnd = np.random.default_rng(self.__alg_seeds[seed_num])
        current_compl = self.__cell_compl.skeleton()
        flows = self.__flow_cache[seed_num][:flow_count]
        no_gradient_flows = np.copy(flows)
        for i in range(flows.shape[0]):
            no_gradient_flows[i] -= project_flow(
                current_compl.boundary_map(1).T, flows[[i]])
        harmonic_flows = np.copy(no_gradient_flows)

        added_cells: List[tuple] = []

        n = 0
        approx_error = np.sum(np.square(harmonic_flows))
        n_to_epsilon = 0
        approx_error_at_n = approx_error if self.__num_curls == 0 else -1
        found_n = approx_error <= self.__approx_epsilon
        partial_res_lst = []
        partial_res_lst.append({
                    'method': self.__search_method,
                    'n_clusters': self.__n_clusters,
                    'flows': flow_count,
                    'iterations': 0,
                    'total_cell_len': 0,
                    'cell_candidates': self.__cells_per_step,
                    'run': seed_num,
                    'approx_error': approx_error
                })
        correct_cells = self.correct_cells(only_detected=False)
        while n < self.__num_curls or (approx_error > self.__approx_epsilon and self.__approx_epsilon >= 0):

            candidate_cells = self.get_cell_candidates(rnd,
                                                       current_compl, harmonic_flows)
            correct_candidates = []
            if self.__always_opt_cell:
                # check if one candidate is in true cells
                correct_candidates = [cell for cell in candidate_cells if cell[0] in correct_cells]
            if len(correct_candidates) == 0:
                score_vals, score_cells = score_cells_multiple(
                    current_compl, no_gradient_flows, [cell[:2] for cell in candidate_cells])
                scores = pd.DataFrame(score_vals, index=pd.Index(score_cells, tupleize_cols=False))
                next_cell = scores.mean(axis=1).idxmin()
            else:
                next_cell = correct_candidates[0][0]
            cell_boundaries = { cell[0]: cell[1] for cell in candidate_cells}

            if next_cell == ():
                print(
                    f'[WARN] detected empty cell: {seed_num}, {flow_count}, ({n})')
                return added_cells, approx_error_at_n, n_to_epsilon, partial_res_lst

            added_cells.append(next_cell)
            current_compl = current_compl.add_cell_fast(next_cell, cell_boundaries[next_cell])

            harmonic_flows = np.copy(no_gradient_flows)
            for j in range(harmonic_flows.shape[0]):
                harmonic_flows[j] -= project_flow(
                    current_compl.boundary_map(2), no_gradient_flows[[j]])
            
            approx_error = np.sum(np.square(harmonic_flows))
            n += 1


            partial_res_lst.append({
                    'method': self.__search_method,
                    'flows': flow_count,
                    'iterations': n,
                    'total_cell_len': sum(map(len, added_cells)),
                    'cell_candidates': self.__cells_per_step,
                    'run': seed_num,
                    'approx_error': approx_error
                })

            if approx_error <= self.__approx_epsilon and not found_n:
                found_n = True
                n_to_epsilon = n
            if n == self.__num_curls:
                approx_error_at_n = approx_error

        return added_cells, approx_error_at_n, n_to_epsilon, partial_res_lst
    
    def simulate_heuristic(self, seed_num: int, flow_count: int) -> list[tuple]:
        """
        simulates a single heuristic run.

        returns the list of cell candidates
        """
        rnd = np.random.default_rng(self.__alg_seeds[seed_num])
        current_compl = self.__cell_compl.skeleton()
        flows = self.__flow_cache[seed_num][:flow_count]
        no_gradient_flows = np.copy(flows)
        for i in range(flows.shape[0]):
            no_gradient_flows[i] -= project_flow(
                current_compl.boundary_map(1).T, flows[[i]])
        harmonic_flows = np.copy(no_gradient_flows)
        return [ tpl[0] for tpl in self.get_cell_candidates(rnd, current_compl, harmonic_flows) ]
    

    def eval_heuristic_step(self) -> list[tuple[int, int, int]]:
        """
        Performs the heuristic for the first iteration and checks how many correct cells (or linear combinations thereof) were found

        returns [(# of flows, run, number of correct cells)]
        """
        result = []
        correct_cells = self.correct_cells(only_detected=False)
        for num_flows in self.__flow_counts:
            for seed_num in range(self.__runs):
                candidates = self.simulate_heuristic(seed_num, num_flows)
                result.append((num_flows, seed_num, len(correct_cells.intersection(candidates))))
        return result


    def run(self):
        """
        Run the cell detection experiment for each seed and flow number
        """
        idx = pd.Index([], tupleize_cols=False, dtype='object')
        results = pd.DataFrame(
            dtype=np.int32, index=self.__flow_counts, columns=idx).fillna(0)
        time_recorded = defaultdict(list)
        correct_cells_found = defaultdict(list)
        approx_error = defaultdict(list)
        approx_cell_count = defaultdict(list)
        total_cell_len = defaultdict(list)
        inferred_cells = defaultdict(list)
        detailed_res_lst = []
        correct_cells = self.correct_cells(only_detected=False)
        for num_flows in self.__flow_counts:
            print(f'Starting Experiments: {num_flows} flows')
            flows_loc = results.index.get_loc(num_flows)
            for seed_num in range(self.__runs):
                before = time.time()
                res, approx_error_at_n, n_to_epsilon, partial_res_lst = self.simulate_run(seed_num, num_flows)
                inferred_cells[num_flows].append(res)
                time_recorded[num_flows].append(time.time() - before)
                correct_cells_found[num_flows].append(len(correct_cells.intersection(res)))
                approx_error[num_flows].append(approx_error_at_n)
                approx_cell_count[num_flows].append(n_to_epsilon)
                total_cell_len[num_flows].append(sum(map(len, res)))
                detailed_res_lst.extend(partial_res_lst)
                for cell in res:
                    if self.__agg_results:
                        if not cell in results.columns:
                            # pylint:disable=unsupported-assignment-operation
                            # pylint (incorrectly) thinks the dataframe does not allow assigning a column
                                results[cell] = 0
                        cell_loc = results.columns.get_loc(cell)
                        results.iloc[flows_loc, cell_loc] += 1
        self.__results = results
        self.__time_recorded = time_recorded
        self.__correct_cells = correct_cells_found
        self.__approx_cell_count = approx_cell_count
        self.__approx_error = approx_error
        self.__cell_len_sum = total_cell_len
        self.__inferred_cells = inferred_cells
        self.__detailed_result_df = pd.DataFrame(detailed_res_lst)