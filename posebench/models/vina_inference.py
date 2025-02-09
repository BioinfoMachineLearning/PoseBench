# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import glob
import logging
import os
import shutil
import tempfile
from collections import defaultdict

import hydra
import numpy as np
import pandas as pd
import rootutils
from beartype.typing import Dict, List, Optional, Tuple
from biopandas.pdb import PandasPdb
from meeko import MoleculePreparation, PDBQTMolecule, PDBQTWriterLegacy, RDKitMolCreate
from omegaconf import DictConfig, open_dict
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import DBSCAN

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from posebench import register_custom_omegaconf_resolvers
from posebench.utils.data_utils import combine_molecules
from posebench.utils.utils import run_command_with_timeout

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BINDING_SITES_DICT = Dict[Tuple[str, ...], Dict[str, float]]
MEEKO_ATOM_TYPES_TO_ADD = (
    {"smarts": "[#19]", "atype": "K"},
    {"smarts": "[#27]", "atype": "Co"},
    {"smarts": "[#48]", "atype": "Cd"},
)

VINA_TIMEOUT_IN_SECONDS = 3600


def scatter_mean(indices: np.ndarray, values: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Scatter mean operation similar to PyTorch's scatter_mean.

    :param indices: Indices to scatter values.
    :param values: Values to scatter.
    :param shape: Shape of the resulting array.
    :return: Array with the mean of the scattered values.
    """
    result = np.zeros(shape, dtype=values.dtype)
    counts = np.zeros(shape, dtype=int)
    np.add.at(result, indices, values)
    np.add.at(counts, indices, 1)
    result = result / counts
    return result


def extract_binding_sites(
    protein_filepath: str,
    ligand_filepaths: List[str],
    ligand_smiles_strings: List[str],
    cfg: Dict,
) -> BINDING_SITES_DICT:
    """Extract binding sites from the input files.

    :param protein_filepath: Protein PDB filepath.
    :param ligand_filepaths: Ligand SDF filepaths.
    :param ligand_smiles_strings: Optional ligand SMILES strings.
    :param cfg: Configuration dictionary from the hydra YAML file.
    :return: Dictionary mapping ligand filepaths to a dictionary of
        binding site coordinate metadata.
    """
    binding_site_mapping = {}
    ligand_coords_list = []
    ligand_atom_molecule_mapping = []

    # Read ligand coordinates and metadata
    for ligand_index, ligand_filepath in enumerate(ligand_filepaths):
        ligand = Chem.SDMolSupplier(ligand_filepath, removeHs=True)[
            0
        ]  # NOTE: we only want to get heavy atoms here
        ligand_coords = Chem.RemoveHs(ligand).GetConformer().GetPositions()
        ligand_coords_list.append(ligand_coords)
        ligand_atom_molecule_mapping.extend([ligand_index] * len(ligand_coords))
    ligand_atom_molecule_mapping = np.array(ligand_atom_molecule_mapping)

    # Perform DBSCAN clustering to group ligands
    ligand_coords = np.concatenate(ligand_coords_list)
    dbscan = DBSCAN(eps=cfg.ligand_ligand_distance_threshold, min_samples=1, metric="euclidean")
    ligand_atom_labels = dbscan.fit_predict(ligand_coords)
    ligand_mol_labels = scatter_mean(
        ligand_atom_molecule_mapping, ligand_atom_labels, len(ligand_filepaths)
    ).astype(int)

    # Group ligand filepaths based on cluster labels
    ligand_filepath_indices = defaultdict(list)
    ligand_filepath_groups = defaultdict(list)
    ligand_smiles_string_groups = defaultdict(list)
    for label, (ligand_index, ligand_filepath) in zip(
        ligand_mol_labels, enumerate(ligand_filepaths)
    ):
        ligand_filepath_indices[label].append(ligand_index)
        ligand_filepath_groups[label].append(ligand_filepath)
        ligand_smiles_string_groups[label].append(ligand_smiles_strings[ligand_index])

    # Find the coordinates of any protein heavy atoms that are within e.g., 4 Å of any ligand heavy atom,
    # and report their average coordinates (i.e., centroid)
    if cfg.method == "p2rank":
        # Alternatively, use P2Rank to (arbitrarily) associate a binding site with each ligand group
        temp_p2rank_output_dir = tempfile.mkdtemp()
        command = [
            cfg.p2rank_exec_path,
            cfg.p2rank_exec_utility,
            "-f",
            protein_filepath,
            "-c",
            cfg.p2rank_config,
            "-o",
            temp_p2rank_output_dir,
        ]
        if not cfg.p2rank_enable_pymol_visualizations:
            command.extend(["-visualizations", "0"])
        os.system(" ".join(command))  # nosec
        p2rank_ligand_group_binding_sites = pd.read_csv(
            os.path.join(
                temp_p2rank_output_dir, f"{os.path.basename(protein_filepath)}_predictions.csv"
            ),
            skipinitialspace=True,
        )[["center_x", "center_y", "center_z"]].values
        shutil.rmtree(temp_p2rank_output_dir)
        assert len(p2rank_ligand_group_binding_sites) >= len(
            ligand_filepath_groups
        ), "P2Rank binding site predictions must be available for all ligand groups."
    else:
        protein_df = PandasPdb().read_pdb(protein_filepath).df["ATOM"]
        protein_coords = protein_df[protein_df["element_symbol"] != "H"][
            ["x_coord", "y_coord", "z_coord"]
        ].values
    for (
        ligand_filepaths_index,
        ligand_filepath_group_index,
        ligand_smiles_string_group_index,
    ) in zip(
        ligand_filepath_indices,
        ligand_filepath_groups,
        ligand_smiles_string_groups,
    ):
        ligand_indices = ligand_filepath_indices[ligand_filepaths_index]
        ligand_filepath_group = ligand_filepath_groups[ligand_filepath_group_index]
        ligand_smiles_string_group = ligand_smiles_string_groups[ligand_smiles_string_group_index]
        if cfg.method == "p2rank":
            binding_site_x, binding_site_y, binding_site_z = p2rank_ligand_group_binding_sites[
                ligand_filepath_group_index
            ]
        else:
            ligand_coords = np.concatenate(
                [
                    # NOTE: we only want to get heavy atoms here
                    Chem.SDMolSupplier(ligand_filepath, removeHs=True)[0]
                    .GetConformer()
                    .GetPositions()
                    for ligand_filepath in ligand_filepath_group
                ]
            )
            protein_ligand_dists = np.linalg.norm(
                protein_coords[..., None, :] - ligand_coords[None, ...], axis=-1
            )
            binding_site_mask = np.any(
                protein_ligand_dists < cfg.protein_ligand_distance_threshold, axis=1
            )
            binding_site_coords = protein_coords[binding_site_mask]
            binding_site_x, binding_site_y, binding_site_z = binding_site_coords.mean(axis=0)
        binding_site_mapping[tuple(ligand_filepath_group)] = {
            "ligand_indices": ligand_indices,
            "ligand_smiles_strings": ligand_smiles_string_group,
            "binding_site_x": binding_site_x.item(),
            "binding_site_y": binding_site_y.item(),
            "binding_site_z": binding_site_z.item(),
        }

    largest_binding_site_ligands = max(binding_site_mapping.keys(), key=len)
    largest_binding_site = binding_site_mapping[largest_binding_site_ligands]
    for ligand_filepaths, binding_site in binding_site_mapping.items():
        if (
            np.isnan(binding_site["binding_site_x"])
            or np.isnan(binding_site["binding_site_y"])
            or np.isnan(binding_site["binding_site_z"])
        ):
            binding_site["binding_site_x"] = largest_binding_site["binding_site_x"]
            binding_site["binding_site_y"] = largest_binding_site["binding_site_y"]
            binding_site["binding_site_z"] = largest_binding_site["binding_site_z"]
            logger.warning(
                f"Binding site not found for ligand filepath(s): {ligand_filepaths}. Substituting with largest protein-ligand binding site at ({largest_binding_site['binding_site_x']}, {largest_binding_site['binding_site_y']}, {largest_binding_site['binding_site_z']})."
            )

    return binding_site_mapping


def run_vina_inference(
    protein_filepath: str,
    ligand_binding_site_mapping: BINDING_SITES_DICT,
    item: str,
    cfg: DictConfig,
) -> Optional[List[str]]:
    """Run AutoDock Vina inference on the input protein structures and ligands.

    :param protein_filepath: Protein PDB filepath.
    :param ligand_filepaths: Ligand SDF filepaths.
    :param ligand_binding_site_mapping: Dictionary mapping ligand
        filepaths to a dictionary of binding site coordinate metadata.
    :param item: Name of the item.
    :param cfg: Configuration dictionary from the hydra YAML file.
    :return: Optional list of output filepaths.
    """
    output_dir = os.path.join(cfg.output_dir, item)
    os.makedirs(output_dir, exist_ok=True)
    ligand_filepath_groups = list(ligand_binding_site_mapping.keys())

    # prepare protein with hydrogens (using `reduce` and `obabel`) in PDBQT format (using the AFDR suite's `prepare_receptor4.py` script)
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as temp_protein:
        temp_protein_filepath = temp_protein.name
    shutil.copy(protein_filepath, temp_protein_filepath)
    command = [
        "reduce",
        temp_protein_filepath,
        ">",
        temp_protein_filepath.replace(".pdb", "_reduced.pdb"),
    ]
    os.system(" ".join(command))  # nosec
    command = [
        "obabel",
        temp_protein_filepath.replace(".pdb", "_reduced.pdb"),
        "-xr",
        "-O",
        temp_protein_filepath.replace(".pdb", "_reduced_prepped.pdb"),
        "-p",
        "7.4",
    ]
    os.system(" ".join(command))  # nosec
    prepared_protein_filepath = temp_protein_filepath.replace(
        ".pdb", "_reduced_prepped.pdb"
    ).replace(".pdb", ".pdbqt")
    command = [
        cfg.python2_exec_path,
        cfg.prepare_receptor_script_path,
        "-r",
        temp_protein_filepath.replace(".pdb", "_reduced_prepped.pdb"),
        "-o",
        prepared_protein_filepath,
    ]
    os.system(" ".join(command))  # nosec
    assert os.path.exists(
        prepared_protein_filepath
    ), f"Prepared protein file not found: {prepared_protein_filepath}"

    # prepare ligands with hydrogens in a random conformation of PDBQT format
    output_filepaths = []
    for ligand_filepath_group_index, ligand_filepaths in enumerate(ligand_filepath_groups):
        prepared_ligand_filepaths = []
        for ligand_filepath in ligand_filepaths:
            # following the PoseBusters supplementary materials, protonate the ligand (using `obabel`) and minimize its (random conformer's) energy
            command = [
                "obabel",
                "-isdf",
                ligand_filepath,
                "-O",
                ligand_filepath.replace(".sdf", "_prepped.sdf"),
                "-p",
                "7.4",
            ]
            os.system(" ".join(command))  # nosec
            temp_ligand_mol = Chem.SDMolSupplier(
                ligand_filepath.replace(".sdf", "_prepped.sdf"), removeHs=False
            )[0]
            if temp_ligand_mol is None:
                temp_ligand_mol = Chem.SDMolSupplier(
                    ligand_filepath.replace(".sdf", "_prepped.sdf"), removeHs=False, sanitize=False
                )[0]
            ps = Chem.rdDistGeom.ETKDGv3()
            try:
                Chem.rdDistGeom.EmbedMolecule(
                    temp_ligand_mol, ps
                )  # generate a random conformation using `ETKDGv3`
            except Exception as e:
                logger.warning(
                    f"RDKit failed to embed molecule: {ligand_filepath.replace('.sdf', '_prepped.sdf')} due to: {e}. Skipping..."
                )
                return None
            try:
                AllChem.UFFOptimizeMolecule(
                    temp_ligand_mol,
                    maxIters=5000,  # run more iterations than the default 200 in case of convergence issues
                )  # minimize the energy of the random conformation using `UFF`
            except Exception as e:
                logger.warning(
                    f"RDKit failed to minimize molecule: {ligand_filepath.replace('.sdf', '_prepped.sdf')} due to: {e}. Skipping..."
                )
                return None
            preparator = MoleculePreparation(add_atom_types=MEEKO_ATOM_TYPES_TO_ADD)
            mol_setups = preparator.prepare(temp_ligand_mol)
            assert len(mol_setups) == 1, "Only one molecule setup per SDF file is supported."
            for setup in mol_setups:
                pdbqt_string, _, _ = PDBQTWriterLegacy.write_string(
                    setup, set_bad_charges_to_zero=True
                )
            with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as temp_ligand:
                prepared_ligand_filepath = temp_ligand.name.replace(".sdf", ".pdbqt")
                if pdbqt_string:
                    with open(prepared_ligand_filepath, "w") as f:
                        f.write(pdbqt_string)
            try:
                assert os.path.exists(
                    prepared_ligand_filepath
                ), f"Prepared ligand file not found: {prepared_ligand_filepath}"
                prepared_ligand_filepaths.append(prepared_ligand_filepath)
            except AssertionError as e:
                logger.warning(
                    f"Prepared ligand file not found: {prepared_ligand_filepath}. Skipping..."
                )
                return None

        # run AutoDock Vina
        binding_site = ligand_binding_site_mapping[ligand_filepaths]
        output_filepath = os.path.join(
            output_dir,
            f"{os.path.splitext(os.path.basename(protein_filepath))[0]}_group_{ligand_filepath_group_index}.pdbqt",
        )
        command = [
            "vina",
            "--receptor",
            prepared_protein_filepath,
            "--ligand",
            " ".join(prepared_ligand_filepaths),
            "--center_x",
            str(binding_site["binding_site_x"]),
            "--center_y",
            str(binding_site["binding_site_y"]),
            "--center_z",
            str(binding_site["binding_site_z"]),
            "--size_x",
            str(cfg.binding_site_size_x),
            "--size_y",
            str(cfg.binding_site_size_y),
            "--size_z",
            str(cfg.binding_site_size_z),
            "--spacing",
            str(cfg.binding_site_spacing),
            "--num_modes",
            str(cfg.num_modes),
            "--out",
            output_filepath,
        ]
        if cfg.cpu is not None:
            command.extend(["--cpu", str(cfg.cpu)])
        if cfg.seed is not None:
            command.extend(["--seed", str(cfg.seed)])
        if cfg.exhaustiveness is not None:
            command.extend(["--exhaustiveness", str(cfg.exhaustiveness)])
        return_code = run_command_with_timeout(" ".join(command), timeout=VINA_TIMEOUT_IN_SECONDS)

        if return_code != 0:
            raise Exception(
                f"AutoDock Vina failed for {ligand_filepaths} with return code: {return_code}."
            )

        assert os.path.exists(output_filepath), f"Vina output file not found: {output_filepath}"
        output_filepaths.append(output_filepath)

    return output_filepaths


def write_vina_outputs(
    output_filepaths: List[str],
    ligand_binding_site_mapping: BINDING_SITES_DICT,
    item: str,
    num_ligands: int,
    cfg: DictConfig,
    remove_hs: bool = True,
):
    """Write AutoDock Vina inference outputs to the output directory.

    :param output_filepaths: List of output filepaths.
    :param ligand_binding_site_mapping: Dictionary mapping ligand
        filepaths to a dictionary of binding site coordinate metadata.
    :param item: Name of the item.
    :param num_ligands: Number of ligands.
    :param cfg: Configuration dictionary from the hydra YAML file.
    :param remove_hs: Whether to remove hydrogens from the combined
        ligand.
    """
    group_ligands_list = [None for _ in range(num_ligands)]
    for output_filepath, group_filepaths in zip(output_filepaths, ligand_binding_site_mapping):
        pdbqt_mol = PDBQTMolecule.from_file(output_filepath, skip_typing=True)
        group_ligands = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
        assert all(
            ligand is not None for ligand in group_ligands
        ), "Group ligands list contains `None` values."
        group_mapping = ligand_binding_site_mapping[group_filepaths]
        for i, (ligand_index, ligand_smiles_string) in enumerate(
            zip(group_mapping["ligand_indices"], group_mapping["ligand_smiles_strings"])
        ):
            template_lig = (
                Chem.RemoveHs(Chem.MolFromSmiles(ligand_smiles_string))
                if remove_hs
                else Chem.MolFromSmiles(ligand_smiles_string)
            )
            group_ligand = (
                Chem.RemoveHs(group_ligands[i]) if remove_hs else group_ligands[i]
            )  # ascertain global (i.e., inter-group) ligand index
            group_ligand_ = Chem.Mol(group_ligand)
            group_ligand_conformers = [conf for conf in group_ligand.GetConformers()]
            group_ligand_.RemoveAllConformers()
            group_ligand_.AddConformer(
                group_ligand_conformers[0]
            )  # retain only the first (e.g., top-ranked) conformer while working around RDKit bug: https://github.com/rdkit/rdkit/issues/3817
            group_ligands_list[ligand_index] = AllChem.AssignBondOrdersFromTemplate(
                template_lig, group_ligand
            )
    assert all(
        group_ligand is not None for group_ligand in group_ligands_list
    ), "Group ligands list contains `None` values."
    combined_ligand = combine_molecules(group_ligands_list)
    Chem.MolToMolFile(combined_ligand, os.path.join(cfg.output_dir, item, f"{item}.sdf"))


@hydra.main(
    version_base="1.3",
    config_path="../../configs/model",
    config_name="vina_inference.yaml",
)
def main(cfg: DictConfig):
    """Run inference using AutoDock Vina on predicted protein structures and
    binding sites.

    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    with open_dict(cfg):
        if cfg.pocket_only_baseline:
            cfg.input_protein_structure_dir += "_bs_cropped"
            cfg.output_dir = cfg.output_dir.replace(
                f"vina_{cfg.method}", f"vina_pocket_only_{cfg.method}"
            )

        if cfg.max_num_inputs:
            cfg.output_dir = cfg.output_dir.replace(
                f"_{cfg.method}", f"_{cfg.method}_first_{cfg.max_num_inputs}"
            )

    if cfg.protein_filepath and cfg.ligand_filepaths and cfg.apo_protein_filepath:
        # support ensemble inference
        logger.info("Beginning AutoDock Vina inference...")
        ligand_filepaths, ligand_smiles_strings = [], []
        ligand_mol_frags = [
            Chem.SDMolSupplier(path, removeHs=False)[0] for path in cfg.ligand_filepaths
        ]
        for ligand_mol_frag in ligand_mol_frags:
            with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as temp_pred:
                Chem.MolToMolFile(ligand_mol_frag, temp_pred.name)
                ligand_filepaths.append(temp_pred.name)
            ligand_smiles_strings.append(Chem.MolToSmiles(ligand_mol_frag))
        try:
            ligand_binding_site_mapping = extract_binding_sites(
                protein_filepath=cfg.protein_filepath,
                ligand_filepaths=ligand_filepaths,
                ligand_smiles_strings=ligand_smiles_strings,
                cfg=cfg,
            )
            group_output_filepaths = run_vina_inference(
                cfg.apo_protein_filepath, ligand_binding_site_mapping, cfg.input_id, cfg
            )
            if group_output_filepaths is None:
                logger.warning(
                    f"AutoDock Vina inference failed for {cfg.ligand_filepaths}. Skipping..."
                )
                return
            write_vina_outputs(
                group_output_filepaths,
                ligand_binding_site_mapping,
                cfg.input_id,
                len(ligand_filepaths),
                cfg,
            )
        except Exception as e:
            logger.warning(f"AutoDock Vina inference failed for {cfg.input_id} due to: {e}")
            raise e
        logger.info(f"AutoDock Vina inference outputs written to `{cfg.output_dir}`.")
        return
    elif cfg.protein_filepath and not (cfg.ligand_filepaths or cfg.apo_protein_filepath):
        raise ValueError(
            "AutoDock Vina inference requires protein, ligand, and apo protein files as inputs."
        )
    elif cfg.ligand_filepaths and not (cfg.protein_filepath or cfg.apo_protein_filepath):
        raise ValueError(
            "AutoDock Vina inference requires protein, ligand, and apo protein files as inputs."
        )
    elif cfg.apo_protein_filepath and not (cfg.protein_filepath or cfg.ligand_filepaths):
        raise ValueError(
            "AutoDock Vina inference requires protein, ligand, and apo protein files as inputs."
        )

    assert os.path.exists(
        cfg.input_protein_structure_dir
    ), f"Input protein structure directory not found: {cfg.input_protein_structure_dir}"

    if cfg.method == "p2rank":
        # support P2Rank input parsing
        pocket_only_suffix = "_pocket_only" if cfg.pocket_only_baseline else ""
        with open_dict(cfg):
            cfg.input_dir = os.path.join(
                "forks",
                "DiffDock",
                "inference",
                f"diffdock{pocket_only_suffix}_{cfg.dataset}_output_{cfg.repeat_index}",
            )
            assert os.path.exists(
                cfg.input_dir
            ), f"DiffDock directory must exist to enable P2Rank input parsing: {cfg.input_dir}"
    if not os.path.exists(cfg.input_dir):
        raise FileNotFoundError(f"Input directory not found: {cfg.input_dir}")
    input_dirs = (
        [os.path.basename(item) for item in glob.glob(cfg.input_dir + "*")]
        if cfg.method == "dynamicbind"
        else os.listdir(cfg.input_dir)
    )
    input_dirs = [
        item for item in input_dirs if "relaxed" not in item
    ]  # NOTE: Vina docking starts with a random conformation, so pre-relaxation is unnecessary

    num_dir_items_found = 0
    for item in input_dirs:
        item_path = (
            os.path.join(os.path.dirname(cfg.input_dir), item)
            if cfg.method == "dynamicbind"
            else os.path.join(cfg.input_dir, item)
        )
        if os.path.isdir(item_path):
            num_dir_items_found += 1
            if cfg.max_num_inputs and num_dir_items_found > cfg.max_num_inputs:
                logger.info(
                    f"Maximum number of input directories reached ({cfg.max_num_inputs}). Exiting inference loop."
                )
                break
            apo_protein_filepaths = glob.glob(
                os.path.join(
                    cfg.input_protein_structure_dir,
                    f"{item.replace('casp15_', '')}*holo_aligned_predicted_protein.pdb",
                )
            )
            if not apo_protein_filepaths:
                logger.warning(
                    f"Apo protein file not found: {apo_protein_filepaths}. Skipping {item}..."
                )
                continue
            apo_protein_filepath = apo_protein_filepaths[0]
            if cfg.method == "diffdock":
                protein_filepaths = glob.glob(
                    os.path.join(
                        cfg.input_protein_structure_dir,
                        f"{item}*holo_aligned_predicted_protein.pdb",
                    )
                )
                ligand_filepath = os.path.join(item_path, "rank1.sdf")
                if not protein_filepaths or not os.path.exists(ligand_filepath):
                    logger.warning(
                        f"DiffDock protein or ligand file not found: {protein_filepaths}, {ligand_filepath}. Skipping {item}..."
                    )
                    continue
                protein_filepath = protein_filepaths[0]
            elif cfg.method == "dynamicbind":
                protein_filepaths = glob.glob(
                    os.path.join(item_path, "index0_idx_0", "rank1_ligand*.sdf")
                )
                ligand_filepaths = glob.glob(
                    os.path.join(item_path, "index0_idx_0", "rank1_receptor*.pdb")
                )
                protein_filepath = protein_filepaths[0]
                ligand_filepath = ligand_filepaths[0]
                if not os.path.exists(protein_filepath) or not os.path.exists(ligand_filepath):
                    logger.warning(
                        f"DynamicBind protein or ligand file not found: {protein_filepath}, {ligand_filepath}. Skipping {item}..."
                    )
                    continue
                assert (
                    len(protein_filepaths) == len(ligand_filepaths) == 1
                ), "A single rank-1 protein-ligand complex is expected from DynamicBind."
            elif cfg.method == "neuralplexer":
                protein_filepath = os.path.join(item_path, "prot_rank1_aligned.pdb")
                ligand_filepath = os.path.join(item_path, "lig_rank1_aligned.sdf")
                if not os.path.exists(protein_filepath) or not os.path.exists(ligand_filepath):
                    logger.warning(
                        f"NeuralPlexer protein or ligand file not found: {protein_filepath}, {ligand_filepath}. Skipping {item}..."
                    )
                    continue
            elif cfg.method == "rfaa":
                protein_filepaths = [
                    item
                    for item in glob.glob(os.path.join(item_path, f"{item}_protein.pdb"))
                    if "relaxed" not in os.path.basename(item)
                ]
                ligand_filepaths = [
                    item
                    for item in glob.glob(os.path.join(item_path, f"{item}_ligand.sdf"))
                    if "relaxed" not in os.path.basename(item)
                ]
                if not (len(protein_filepaths) == len(ligand_filepaths) == 1):
                    logger.warning(
                        f"A single rank-1 protein-ligand complex is expected from RFAA. Skipping {item}"
                    )
                    continue
                protein_filepath = protein_filepaths[0]
                ligand_filepath = ligand_filepaths[0]
                if not os.path.exists(protein_filepath) or not os.path.exists(ligand_filepath):
                    logger.warning(
                        f"RoseTTAFold-All-Atom protein or ligand file not found: {protein_filepath}, {ligand_filepath}. Skipping {item}..."
                    )
                    continue
            elif cfg.method == "p2rank":
                protein_filepaths = glob.glob(
                    os.path.join(
                        cfg.input_protein_structure_dir,
                        f"{item}*holo_aligned_predicted_protein.pdb",
                    )
                )
                ligand_filepath = os.path.join(item_path, "rank1.sdf")
                if not protein_filepaths or not os.path.exists(ligand_filepath):
                    logger.warning(
                        f"P2Rank (DiffDock-)surrogate protein or ligand file not found: {protein_filepaths}, {ligand_filepath}. Skipping {item}..."
                    )
                    continue
                protein_filepath = protein_filepaths[0]
            elif cfg.method == "ensemble":
                protein_filepaths = [
                    item
                    for item in glob.glob(os.path.join(item_path, "*rank1*.pdb"))
                    if "relaxed" not in os.path.basename(item)
                ]
                ligand_filepaths = [
                    item
                    for item in glob.glob(os.path.join(item_path, "*rank1*.sdf"))
                    if "relaxed" not in os.path.basename(item)
                ]
                protein_filepath = protein_filepaths[0]
                ligand_filepath = ligand_filepaths[0]
                if not os.path.exists(protein_filepath) or not os.path.exists(ligand_filepath):
                    logger.warning(
                        f"Ensemble protein or ligand file not found: {protein_filepath}, {ligand_filepath}. Skipping {item}..."
                    )
                    continue
                assert (
                    len(protein_filepaths) == len(ligand_filepaths) == 1
                ), "A single rank-1 protein-ligand complex is expected from ensembling."
            else:
                raise ValueError(f"Invalid method for Vina binding site predictions: {cfg.method}")

            output_filepaths = glob.glob(os.path.join(cfg.output_dir, item, f"{item}.sdf"))
            if cfg.skip_existing and len(output_filepaths):
                logger.info(
                    f"AutoDock Vina inference output(s) already exist(s): {output_filepaths}. Skipping..."
                )
                continue

            # isolate molecule fragments
            ligand_mols = Chem.SDMolSupplier(ligand_filepath, removeHs=False)
            assert len(ligand_mols) == 1, "Only one molecule per SDF file is supported."
            ligand_mol = ligand_mols[0]
            ligand_mol_frags = Chem.GetMolFrags(ligand_mol, asMols=True)

            ligand_filepaths, ligand_smiles_strings = [], []
            for ligand_mol_frag in ligand_mol_frags:
                with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as temp_pred:
                    Chem.MolToMolFile(ligand_mol_frag, temp_pred.name)
                    ligand_filepaths.append(temp_pred.name)
                ligand_smiles_strings.append(Chem.MolToSmiles(ligand_mol_frag))

            try:
                ligand_binding_site_mapping = extract_binding_sites(
                    protein_filepath,
                    ligand_filepaths,
                    ligand_smiles_strings,
                    cfg,
                )
                group_output_filepaths = run_vina_inference(
                    apo_protein_filepath, ligand_binding_site_mapping, item, cfg
                )
                if group_output_filepaths is None:
                    logger.warning(f"AutoDock Vina inference failed for {item}. Skipping...")
                    continue
                write_vina_outputs(
                    group_output_filepaths,
                    ligand_binding_site_mapping,
                    item,
                    len(ligand_filepaths),
                    cfg,
                )
            except Exception as e:
                logger.warning(f"AutoDock Vina inference failed for {item}. Skipping due to: {e}")
                continue
    logger.info(f"AutoDock Vina inference outputs written to `{cfg.output_dir}`.")


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
