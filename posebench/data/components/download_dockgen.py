import gzip
import os
import shutil

import wget
from Bio.PDB import PDBIO, PDBParser, Select
from tqdm import tqdm


class NonWaterSelect(Select):
    """Custom selector to exclude water molecules."""

    def accept_residue(self, residue):
        return residue.resname not in {"HOH", "WAT"}


def remove_waters_with_biopython(input_pdb: str, output_pdb: str):
    """Removes water molecules from a PDB file using Biopython.

    :param input_pdb: Path to the input PDB file.
    :param output_pdb: Path to save the output PDB file without waters.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("Structure", input_pdb)

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb, NonWaterSelect())


def main():
    """Download and process DockGen set."""
    for name in tqdm(os.listdir(os.path.join("data", "dockgen_set"))):
        path = os.path.join(os.path.join("data", "dockgen_set"), name)
        pdb_id = name.split("_")[0]
        gz_file = os.path.join(path, f"{pdb_id}.pdb1.gz")
        pdb_file = os.path.join(path, f"{pdb_id}_processed.pdb")

        if pdb_id in ("alphafold", "dockgen", "plots") or not os.path.isdir(path):
            continue

        print(f"Downloading {pdb_id}...")
        wget.download(f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb1.gz", path)

        with gzip.open(gz_file, "rb") as f_in:
            with open(pdb_file.replace(".pdb", "_temp.pdb"), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(gz_file)

        remove_waters_with_biopython(pdb_file.replace(".pdb", "_temp.pdb"), pdb_file)
        os.remove(pdb_file.replace(".pdb", "_temp.pdb"))


if __name__ == "__main__":
    main()
