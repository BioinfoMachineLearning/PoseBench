# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import numpy as np


def calculate_rmsd(positions1: np.array, positions2: np.array) -> float:
    """Calculate the root-mean-square deviation (RMSD) between two sets of
    positions.

    :param positions1: Array of positions for the first set of atoms.
    :param positions2: Array of positions for the second set of atoms.
    :return: RMSD between the two sets of positions.
    """
    diff_squared = np.sum((positions1 - positions2) ** 2, axis=1)
    rmsd = np.sqrt(np.mean(diff_squared))
    return rmsd
