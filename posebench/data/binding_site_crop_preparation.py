# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import logging
import os

import hydra
import rootutils
from beartype import beartype
from beartype.typing import Any, List, Tuple
from Bio.PDB import PDBIO, NeighborSearch, PDBParser, Select
from omegaconf import DictConfig
from rdkit import Chem
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from posebench import register_custom_omegaconf_resolvers
from posebench.utils.data_utils import parse_inference_inputs_from_dir

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BindingSiteSelect(Select):
    """Custom Select class to filter residues based on binding site residue
    indices."""

    def __init__(self, structure_residues: List[Any], binding_site_residue_indices: List[int]):
        """Initialize the BindingSiteSelect class."""
        self.structure_residues = structure_residues
        self.binding_site_residue_indices = set(binding_site_residue_indices)

    def accept_residue(self, residue: Any) -> bool:
        """Accept residues based on whether they are part of the binding site
        or not.

        :param residue: Residue object from the Bio.PDB module.
        :return: Boolean indicating whether the residue is part of the
            binding site.
        """
        return self.structure_residues.index(residue) in self.binding_site_residue_indices


@beartype
def get_binding_site_residue_indices(
    protein_filepath: str,
    ligand_filepath: str,
    protein_ligand_distance_threshold: float = 10.0,
    num_buffer_residues: int = 7,
) -> List[int]:
    """Get the zero-based residue indices of the protein binding site based on
    native protein- ligand interactions.

    :param protein_filepath: Path to the protein structure PDB file.
    :param ligand_filepath: Path to the ligand structure SDF file.
    :param protein_ligand_distance_threshold: Heavy-atom distance
        threshold (in Angstrom) to use for finding protein binding site
        residues in interaction with ligand heavy atoms.
    :param num_buffer_residues: Number of residues to include as a
        buffer around each binding site residue.
    :return: List of zero-based residue indices that define the binding
        site.
    """
    assert os.path.exists(
        protein_filepath
    ), f"Protein structure file `{protein_filepath}` does not exist."
    assert os.path.exists(
        ligand_filepath
    ), f"Ligand structure file `{ligand_filepath}` does not exist."

    # parse the protein structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", protein_filepath)

    # load the ligand using RDKit
    ligand = Chem.MolFromMolFile(ligand_filepath, removeHs=True)
    assert ligand is not None, "Failed to load ligand structure."

    # get the coordinates of ligand heavy atoms
    ligand_conformer = ligand.GetConformer()
    ligand_coords = [
        ligand_conformer.GetAtomPosition(i)
        for i in range(ligand.GetNumAtoms())
        if ligand.GetAtomWithIdx(i).GetAtomicNum() > 1
    ]

    # flatten the list of coordinates into tuples
    ligand_coords = [(atom.x, atom.y, atom.z) for atom in ligand_coords]

    # extract all heavy atoms from the protein structure
    all_heavy_atoms = [atom for atom in structure.get_atoms() if atom.element != "H"]

    # use NeighborSearch to find all heavy atoms within the specified distance from any ligand heavy atom
    neighbor_search = NeighborSearch(all_heavy_atoms)
    nearby_atoms = set()
    for coord in ligand_coords:
        nearby_atoms.update(neighbor_search.search(coord, protein_ligand_distance_threshold))

    # collect the zero-based indices of residues that these atoms belong to
    binding_site_residue_indices = set()
    structure_residues = list(structure.get_residues())
    for atom in nearby_atoms:
        residue = atom.get_parent()
        binding_site_residue_indices.add(structure_residues.index(residue))
    assert (
        len(binding_site_residue_indices) > 0
    ), f"No binding site residues found for ligand associated with {protein_filepath}."

    # pad and connect the binding site residue indices with sequence-regional buffer residues
    padded_binding_site_residue_indices = list(
        range(
            max(0, min(binding_site_residue_indices) - num_buffer_residues),
            min(
                len(structure_residues) - 1,
                max(binding_site_residue_indices) + num_buffer_residues,
            ),
        )
    )

    return padded_binding_site_residue_indices


@beartype
def crop_protein_binding_site(
    protein_filepath: str,
    binding_site_residue_indices: List[int],
    output_dir: str,
    pdb_id: str,
    filename_midfix: str = "",
    filename_suffix: str = "",
):
    """Crop the protein binding site and save it to a separate file.

    :param protein_filepath: Path to the input protein structure file.
    :param binding_site_residue_indices: List of zero-based residue
        indices that define the binding site.
    :param output_dir: Path to the output directory.
    :param pdb_id: PDB ID of the protein-ligand complex.
    :param filename_midfix: Optional "midfix" to insert into the cropped
        protein structure filename.
    :param filename_suffix: Optional suffix to append to the cropped
        protein structure filename.
    """
    assert os.path.exists(
        protein_filepath
    ), f"Protein structure file `{protein_filepath}` does not exist."
    assert (
        len(binding_site_residue_indices) > 0
    ), "The binding site residue indices list must contain at least one residue index."
    assert os.path.exists(output_dir), f"Output directory `{output_dir}` does not exist."

    # parse the protein structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, protein_filepath)

    # create a selection class instance with the binding site indices
    select = BindingSiteSelect(list(structure.get_residues()), binding_site_residue_indices)

    # define the output file path
    output_filepath = os.path.join(
        output_dir, f"{pdb_id}{filename_midfix}_protein{filename_suffix}.pdb"
    )

    # write the selected residues to the output file
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_filepath, select)


@beartype
def save_cropped_protein_binding_site(
    smiles_and_pdb_id_list: List[Tuple[Any, str]],
    input_data_dir: str,
    input_protein_structure_dir: str,
    protein_ligand_distance_threshold: float = 10.0,
    num_buffer_residues: int = 7,
):
    """Save the cropped protein binding site to a separate file for each
    protein-ligand complex.

    :param smiles_and_pdb_id_list: A list of tuples each containing a
        SMILES string and a PDB ID.
    :param dataset: Dataset name.
    :param input_data_dir: Path to directory of input protein-ligand
        complex subdirectories.
    :param input_protein_structure_dir: Path to the directory containing
        the protein structure input files.
    :param protein_ligand_distance_threshold: Heavy-atom distance
        threshold (in Angstrom) to use for finding protein binding site
        residues in interaction with ligand heavy atoms.
    :param num_buffer_residues: Number of residues to include as a
        buffer around each binding site residue.
    """
    output_protein_structure_dir = input_protein_structure_dir + "_bs_cropped"
    os.makedirs(output_protein_structure_dir, exist_ok=True)
    for _, pdb_id in tqdm(smiles_and_pdb_id_list, desc="Cropping protein binding sites"):
        pred_protein_filepath = os.path.join(
            input_protein_structure_dir,
            f"{pdb_id}_holo_aligned_predicted_protein.pdb",
        )
        ref_protein_filepath = os.path.join(input_data_dir, pdb_id, f"{pdb_id}_protein.pdb")
        ref_ligand_filepath = os.path.join(input_data_dir, pdb_id, f"{pdb_id}_ligand.sdf")
        if not os.path.exists(pred_protein_filepath):
            logger.warning(
                f"Predicted protein structure file `{pred_protein_filepath}` does not exist. Skipping binding site cropping for this complex..."
            )
            continue
        assert os.path.exists(
            ref_protein_filepath
        ), f"Reference (ground-truth) protein structure file `{ref_protein_filepath}` does not exist."
        assert os.path.exists(
            ref_ligand_filepath
        ), f"Reference (ground-truth) ligand structure file `{ref_ligand_filepath}` does not exist."
        binding_site_residue_indices = get_binding_site_residue_indices(
            ref_protein_filepath,
            ref_ligand_filepath,
            protein_ligand_distance_threshold=protein_ligand_distance_threshold,
            num_buffer_residues=num_buffer_residues,
        )
        crop_protein_binding_site(
            pred_protein_filepath,
            binding_site_residue_indices,
            output_protein_structure_dir,
            pdb_id,
            filename_midfix="_holo_aligned_predicted",
        )
        crop_protein_binding_site(
            ref_protein_filepath,
            binding_site_residue_indices,
            os.path.join(input_data_dir, pdb_id),
            pdb_id,
            filename_suffix="_bs_cropped",
        )


@hydra.main(
    version_base="1.3",
    config_path="../../configs/data",
    config_name="binding_site_crop_preparation.yaml",
)
def main(cfg: DictConfig):
    """Parse a data directory containing subdirectories of protein-ligand
    complexes and prepare corresponding inference CSV file for the DiffDock
    model.

    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    smiles_and_pdb_id_list = parse_inference_inputs_from_dir(cfg.input_data_dir)
    save_cropped_protein_binding_site(
        smiles_and_pdb_id_list,
        cfg.input_data_dir,
        cfg.input_protein_structure_dir,
        protein_ligand_distance_threshold=cfg.protein_ligand_distance_threshold,
        num_buffer_residues=cfg.num_buffer_residues,
    )

    logger.info(f"Protein binding site cropping for dataset `{cfg.dataset}` complete.")


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
