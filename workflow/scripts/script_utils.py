
# add code to import path
from pathlib import Path
import sys, resource
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

from cell_flower.detection import CellSearchMethod
from cell_complexes.experiment import GROUND_TRUTH

method_map = {
    'max': CellSearchMethod.MAX,
    'triangles': CellSearchMethod.TRIANGLES,
    'ground_truth': GROUND_TRUTH,
    'similarity': CellSearchMethod.CLUSTER,
}
