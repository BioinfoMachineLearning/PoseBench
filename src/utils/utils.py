# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

from pathlib import Path

from beartype.typing import List


def find_protein_files(protein_file_dir: Path, extension: str = "pdb") -> List[Path]:
    """Find all protein files in the specified directory.

    :param protein_file_dir: The directory containing the protein files.
    :param extension: The file extension of the protein files.
    :return: A list of `Path` objects representing the protein files.
    """
    return list(protein_file_dir.rglob(f"*.{extension}"))


def find_ligand_files(ligand_file_dir: Path, extension: str = "sdf") -> List[Path]:
    """Find all ligand files in the specified directory.

    :param ligand_file_dir: The directory containing the ligand files.
    :param extension: The file extension of the ligand files.
    :return: A list of `Path` objects representing the ligand files.
    """
    return list(ligand_file_dir.rglob(f"*.{extension}"))
