
# add code to import path
from pathlib import Path
import sys, resource
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

from cell_flower.detection import CellCandidateHeuristic
from cell_complexes.experiment import GROUND_TRUTH

method_map = {
    'max': CellCandidateHeuristic.MAX,
    'triangles': CellCandidateHeuristic.TRIANGLES,
    'ground_truth': GROUND_TRUTH,
    'similarity': CellCandidateHeuristic.SIMILARITY,
}
