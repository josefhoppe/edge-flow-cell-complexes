
from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
from numpy.random import Generator
from scipy.spatial.qhull import Delaunay
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement
import itertools
import networkx as nx

from cell_flower.cell_complex import CellComplex, calc_edges, normalize_cell


class CellComplexGenerator(ABC):
    """
    Base class for generating Cell complexes.
    """
    _seed: int
    _cell_lengths: Iterable[int]
    _nodes: int

    def __init__(self, seed: int = 34905852, nodes: int = 20, cell_lengths: Iterable[int] = (5,5,10,10)):
        self._seed = seed
        self._nodes = nodes
        self._cell_lengths = cell_lengths

    @abstractmethod
    def generate(self) -> CellComplex:
        """
        Generates a CellComplex according to the settings.
        """
        
class SplitCircleComplexGenerator(CellComplexGenerator):
    """
    Generates a Cellcomplex that consists of:

    - a main path of length `split_circle_common_len`
    - 2-cells that include this main path and also `cell_lengths` other nodes
    - `edges` many randomly added edges
    """
    __edges: int
    __split_circle_common_len: int

    def __init__(self, seed: int = 34905852, nodes: int = 20, cell_lengths: Iterable[int] = (5, 5, 10, 10),
                 split_circle_common_len: int = 3, edges: int = 80):
        super().__init__(seed, nodes, cell_lengths)
        self.__edges = edges
        self.__split_circle_common_len = split_circle_common_len

    def generate(self) -> CellComplex:
        common_circle_part = list(range(1, self.__split_circle_common_len + 1))
        next_node = self.__split_circle_common_len + 1
        cells = []
        for split_len in self._cell_lengths:
            cells.append(tuple(common_circle_part + list(range(next_node, next_node + split_len))))
            next_node += split_len
        
        rnd = check_random_state(self._seed)
        for _ in range(self.__edges):
            cells.append(normalize_cell(tuple(i + 1 for i in sample_without_replacement(next_node - 1, 2, random_state=rnd))))

        return CellComplex(cells)

class TriangulationComplexGenerator(CellComplexGenerator):
    """
    CellComplexGenerator that generates a cell complex based on a Delaunay triangulation
    """
    __ndim: int
    __delete_nodes: int
    __delete_edges: int

    def __init__(self, seed: int = 34905852, nodes: int = 50, cell_lengths: Iterable[int] = (5, 5, 10, 10),
                 ndim: int = 2, delete_nodes: int = 5, delete_edges: int = 10):
        super().__init__(seed, nodes, cell_lengths)
        self.__ndim = ndim
        self.__delete_edges = delete_edges
        self.__delete_nodes = delete_nodes

    def generate(self) -> CellComplex:
        rnd = np.random.default_rng(self._seed)
        points = rnd.uniform(size=(self._nodes,self.__ndim))

        tri = Delaunay(points)
        cells = [ self.generate_cell(rnd, tri, length) for length in self._cell_lengths]
        
        can_delete_node = np.ones(self._nodes)
        for a,b in tri.convex_hull:
            can_delete_node[a] = 0
            can_delete_node[b] = 0
        for cell in cells:
            for node in cell:
                can_delete_node[node] = 0
        deleted_nodes = set()
        if np.sum(can_delete_node) > 0:
            deleted_nodes = set(rnd.choice(self._nodes, size=self.__delete_nodes, replace=False, p=can_delete_node / np.sum(can_delete_node)))
        else:
            print('WARN: cannot delete any nodes')
        idx_diff = np.zeros(self._nodes)
        for node in deleted_nodes:
            idx_diff[node:] += 1

        def new_lbl(node_lbl: int) -> int:
            return node_lbl - int(idx_diff[node_lbl])
        
        cells = [ tuple(new_lbl(n) for n in cell) for cell in cells ]
        edges = list(set().union(*[calc_edges(tuple(simpl)) for simpl in tri.simplices]))
        edges = [ (new_lbl(a), new_lbl(b)) for a,b in edges if len(deleted_nodes.intersection({a,b})) == 0 ]
        edges.sort()

        can_delete_edge = np.ones(len(edges))
        for (a,b) in tri.convex_hull:
            norm_edge = normalize_cell((new_lbl(a), new_lbl(b)))
            can_delete_edge[edges.index(norm_edge)] = 0
        for cell in cells:
            for edge in calc_edges(cell):
                can_delete_edge[edges.index(edge)] = 0

        delete_edges = rnd.choice(edges, self.__delete_edges, replace=False, p=can_delete_edge / np.sum(can_delete_edge))

        nodes_should_exist = { new_lbl(n) for n in set(range(self._nodes)).difference(deleted_nodes) }
        for edge in delete_edges:
            edges.remove(tuple(edge))

        # make sure no nodes are disconnected, adding edges if necessary
        nodes_after_edge_deletion = {e[0] for e in edges}.union({e[1] for e in edges})
        missing_nodes = nodes_should_exist.difference(nodes_after_edge_deletion)
        choice_list = list(nodes_after_edge_deletion)
        dont_delete_edges = [ (n, rnd.choice(choice_list)) for n in missing_nodes ]

        result = CellComplex(edges + dont_delete_edges + cells)
        result.embedding = points[[idx for idx in range(self._nodes) if not idx in deleted_nodes]]
        return result


    def generate_cell(self, rnd: Generator, tri: Delaunay, length: int) -> tuple:
        """
        samples a single cell of length `length` from tri by adding simplices until the total boundary is sufficiently long.
        """
        assert length >= 3

        added_simplices = np.full(tri.nsimplex, False)
        shared_boundary = np.zeros(tri.nsimplex, np.int8)
        is_first = True

        border_nodes = []

        while len(border_nodes) < length:
            next_add_simpl = None
            if is_first:
                next_add_simpl = rnd.choice(tri.nsimplex)
                border_nodes = list(tri.simplices[next_add_simpl])
            else:
                next_add_simpl = rnd.choice(tri.nsimplex, p=shared_boundary / np.sum(shared_boundary))

            is_loop = False
            # ensure no loop is closed
            for idx, point in enumerate(tri.simplices[next_add_simpl]):
                if point in border_nodes and not is_first:
                    # if the point is part of the border, but not adjacent to an adjacent simplex, we are closing a loop
                    if (not added_simplices[tri.neighbors[next_add_simpl][(idx + 1) % 3]]) and (not added_simplices[tri.neighbors[next_add_simpl][(idx + 2) % 3]]):
                        is_loop = True

            if not is_loop:
                shared_boundary[next_add_simpl] = 0
                added_simplices[next_add_simpl] = True
                closed_wedge = False
                for idx, point in enumerate(tri.simplices[next_add_simpl]):
                    # check if point is now an inner point of the cell
                    if (not is_first) and added_simplices[tri.neighbors[next_add_simpl][(idx + 1) % 3]] and added_simplices[tri.neighbors[next_add_simpl][(idx + 2) % 3]]:
                        border_nodes.remove(point)
                        closed_wedge = True
                if (not closed_wedge) and (not is_first):
                    new_point = [ p for p in tri.simplices[next_add_simpl] if p not in border_nodes][0]
                    existing_point_indices = [ border_nodes.index(p) for p in tri.simplices[next_add_simpl] if p != new_point ]
                    if 0 in existing_point_indices and 1 not in existing_point_indices:
                        # between first and last element -> append at end
                        border_nodes.append(new_point)
                    else:
                        # insert in the middle
                        border_nodes.insert(max(existing_point_indices), new_point)
                for neighbor in tri.neighbors[next_add_simpl]:
                    if neighbor != -1 and not added_simplices[neighbor]:
                        shared_boundary[neighbor] += 1
                is_first = False
        return normalize_cell(tuple(border_nodes))



class SmallWorldComplexGenerator(CellComplexGenerator):
    """
    CellComplexGenerator that generates a cell complex based on NetworkX' `newman_watts_strogatz_graph`
    """
    __p: float
    __k: int

    def __init__(self, seed: int = 34905852, nodes: int = 50, cell_lengths: Iterable[int] = (5, 5, 10, 10),
                 p: int = 0.01, k: int = 4):
        super().__init__(seed, nodes, cell_lengths)
        assert k > 3
        self.__p = p
        self.__k = k

    def generate(self) -> CellComplex:
        G = nx.watts_strogatz_graph(self._nodes, self.__k, 0, seed=self._seed)
        rnd = np.random.default_rng(self._seed + 1)
        cells = [self.gen_cell(rnd, length) for length in self._cell_lengths]

        for a,b in itertools.combinations(G.nodes, 2):
            if a < b:
                if rnd.choice([True, False], p=[self.__p, 1-self.__p]):
                    cells.append((a,b))
        
        return CellComplex(list(map(lambda e: tuple(sorted(e)), G.edges)) + cells)

    def gen_cell(self, rnd: Generator, length: int) -> tuple:
        """
        draws a node and generates a cycle of the next `len` nodes by alternating between 'path forward' and 'path backward' nodes.
        """
        start_node = rnd.choice(self._nodes)
        cell = (start_node,) + tuple(range(start_node+1,start_node+length, 2)) + tuple(reversed(range(start_node+2, start_node+length, 2)))
        return tuple(map(lambda x: x % self._nodes, cell))
