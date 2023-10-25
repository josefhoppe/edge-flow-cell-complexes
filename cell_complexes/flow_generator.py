

from typing import List, NamedTuple

import pandas as pd
from numpy.random import Generator

from cell_flower.cell_complex import CellComplex


class FlowGeneratorCell(NamedTuple):
    cell: tuple
    min: float
    max: float


def generate_flows_edge_noise(rnd: Generator, cell_compl: CellComplex, generator_cells: List[FlowGeneratorCell], max_noise: float, count: int = 10) -> List[pd.Series]:
    """
    generate flows from the given cells (with a gaussian flow each), with an added edge-level gaussian noise.
    """
    result = []
    

    for _ in range(count):
        curl_flow: pd.Series = pd.Series({g.cell: rnd.normal(
            (g.max+g.min) / 2, (g.max-g.min)/2) for g in generator_cells})
        indices = [cell_compl.get_cells(2).index(g.cell) for g in generator_cells]
        flow = cell_compl.boundary_map(2)[:, indices] @ curl_flow
        flow += rnd.normal(scale=max_noise, size=len(flow))
        result.append(pd.Series(flow, index=cell_compl.get_cells(1)))

    return result
