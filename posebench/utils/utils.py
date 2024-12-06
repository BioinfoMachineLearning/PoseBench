# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import subprocess  # nosec
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


def run_command_with_timeout(command: str, timeout: int) -> int:
    """Run a command with a specified timeout in seconds.

    :param command: The command to run.
    :param timeout: The timeout for the command.
    :return: The return code of the command.
    """
    try:
        result = subprocess.run(command, shell=True, timeout=timeout, check=True)  # nosec
        return result.returncode
    except subprocess.TimeoutExpired:
        print(f"Command timed out: {command}")
        return -1
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
        return e.returncode
    except Exception as e:
        print(f"Command failed with error: {e}")
        return -1
