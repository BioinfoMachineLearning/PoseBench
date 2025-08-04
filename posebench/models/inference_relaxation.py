# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import glob
import logging
import multiprocessing
import os
import subprocess  # nosec
from collections import defaultdict
from pathlib import Path

import hydra
import rootutils
from omegaconf import DictConfig, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from posebench import METHOD_TITLE_MAPPING, register_custom_omegaconf_resolvers
from posebench.utils.utils import find_ligand_files, find_protein_files

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def relax_inference_results(
    protein_file_dir: Path,
    ligand_file_dir: Path,
    output_file_dir: Path,
    temp_directory: Path,
    cfg: DictConfig,
):
    """Relax a method's inference results using the specified configuration.

    :param protein_file_dir: The directory containing the protein files.
    :param ligand_file_dir: The directory containing the ligand files.
    :param output_file_dir: The directory to save the output files.
    :param temp_directory: The temporary directory to use for intermediate files.
    :param cfg: The relaxation configuration `DictConfig`.
    """
    if not protein_file_dir.exists() or cfg.method == "dynamicbind":
        protein_filepaths = [
            file
            for dir in glob.glob(f"{protein_file_dir}_*_{cfg.repeat_index}")
            for file in find_protein_files(Path(dir))
            if "rank1_receptor" in file.stem and "relaxed" not in file.parent.stem
        ]
    else:
        protein_filepaths = find_protein_files(protein_file_dir)
        if any("rank" in filepath.stem for filepath in protein_filepaths):
            # NOTE: handle for ranked outputs such as those of DiffDock
            protein_filepaths = [
                filepath
                for filepath in protein_filepaths
                if (
                    "rank1.pdb" in filepath.name
                    or (
                        cfg.method != "diffdock"
                        and "rank1_" in filepath.name
                        and filepath.name.endswith(".pdb")
                    )
                )
                and "relaxed" not in filepath.parent.stem
            ]
        elif cfg.method == "rfaa":
            protein_filepaths = [
                filepath
                for filepath in protein_filepaths
                if "_protein.pdb" in filepath.name and "relaxed" not in filepath.parent.stem
            ]
        elif cfg.method == "chai-lab":
            protein_filepaths = [
                filepath
                for filepath in protein_filepaths
                if "model_idx_0_protein.pdb" in filepath.name
                and "relaxed" not in filepath.parent.stem
            ]
        elif cfg.method == "boltz":
            protein_filepaths = [
                filepath
                for filepath in protein_filepaths
                if "model_0_protein.pdb" in filepath.name and "relaxed" not in filepath.parent.stem
            ]
        elif cfg.method == "alphafold3":
            protein_filepaths = [
                filepath
                for filepath in protein_filepaths
                if "_model_protein.pdb" in filepath.name and "relaxed" not in filepath.parent.stem
            ]
    if not ligand_file_dir.exists() or cfg.method == "dynamicbind":
        ligand_filepaths = [
            file
            for dir in glob.glob(f"{ligand_file_dir}_*_{cfg.repeat_index}")
            for file in find_ligand_files(Path(dir))
            if "rank1_ligand" in file.stem and "relaxed" not in file.parent.stem
        ]
    else:
        ligand_filepaths = find_ligand_files(ligand_file_dir)
        if any("rank" in filepath.stem for filepath in ligand_filepaths):
            # NOTE: handle for ranked outputs such as those of DiffDock
            ligand_filepaths = [
                filepath
                for filepath in ligand_filepaths
                if (
                    "rank1.sdf" in filepath.name
                    or (
                        cfg.method != "diffdock"
                        and "rank1_" in filepath.name
                        and filepath.name.endswith(".sdf")
                    )
                )
                and "relaxed" not in filepath.stem
                and "relaxed" not in filepath.parent.stem
            ]
        elif cfg.method == "rfaa":
            ligand_filepaths = [
                filepath
                for filepath in ligand_filepaths
                if "_ligand.sdf" in filepath.name and "relaxed" not in filepath.parent.stem
            ]
        elif cfg.method == "chai-lab":
            ligand_filepaths = [
                filepath
                for filepath in ligand_filepaths
                if "model_idx_0_ligand.sdf" in filepath.name
                and "relaxed" not in filepath.parent.stem
            ]
        elif cfg.method == "boltz":
            ligand_filepaths = [
                filepath
                for filepath in ligand_filepaths
                if "model_0_ligand.sdf" in filepath.name and "relaxed" not in filepath.parent.stem
            ]
        elif cfg.method == "alphafold3":
            ligand_filepaths = [
                filepath
                for filepath in ligand_filepaths
                if "_model_ligand.sdf" in filepath.name and "relaxed" not in filepath.parent.stem
            ]
        elif cfg.method == "vina":
            ligand_filepaths = [
                filepath for filepath in ligand_filepaths if "relaxed" not in filepath.stem
            ]
    protein_filepaths = sorted(
        [
            fp
            for fp in protein_filepaths
            if not any(s in fp.stem for s in ("relaxed", "aligned")) or "holo_aligned" in fp.stem
        ]
    )
    ligand_filepaths = sorted(
        [fp for fp in ligand_filepaths if not any(s in fp.stem for s in ("relaxed", "aligned"))]
    )
    id_slice = slice(0, 4) if cfg.dataset == "dockgen" else slice(2)
    if len(protein_filepaths) < len(ligand_filepaths):
        if cfg.method == "dynamicbind":
            # NOTE: sometimes, DynamicBind mysteriously omits the protein output for a given complex
            ligand_filepaths = [
                ligand_filepath
                for ligand_filepath in ligand_filepaths
                if any(
                    "_".join(protein_filepath.parent.parent.stem.split("_")[-3:])
                    in ligand_filepath.parent.parent.stem
                    for protein_filepath in protein_filepaths
                )
            ]
    if len(ligand_filepaths) < len(protein_filepaths):
        # NOTE: the performance of these loops could likely be improved
        if cfg.method == "diffdock":
            protein_filepaths = [
                protein_filepath
                for protein_filepath in protein_filepaths
                if any(
                    "_".join(protein_filepath.stem.split("_")[id_slice]) in ligand_filepath.stem
                    for ligand_filepath in ligand_filepaths
                )
                or any(
                    "_".join(protein_filepath.stem.split("_")[id_slice])
                    in ligand_filepath.parent.stem
                    for ligand_filepath in ligand_filepaths
                )
            ]
        elif cfg.method == "dynamicbind":
            protein_filepaths = [
                protein_filepath
                for protein_filepath in protein_filepaths
                if any(
                    "_".join(protein_filepath.parent.parent.stem.split("_")[-3:])
                    in ligand_filepath.parent.parent.stem
                    for ligand_filepath in ligand_filepaths
                )
            ]
        elif cfg.method in ["chai-lab", "boltz", "alphafold3"]:
            raise NotImplementedError(
                "Cannot subset `chai-lab`, `boltz`, or `alphafold3` protein predictions at this time."
            )
        else:
            protein_filepaths = [
                protein_filepath
                for protein_filepath in protein_filepaths
                if (
                    any(
                        "_".join(protein_filepath.stem.split("_")[id_slice])
                        in ligand_filepath.stem
                        for ligand_filepath in ligand_filepaths
                    )
                    or any(
                        "_".join(protein_filepath.stem.split("_")[id_slice])
                        in ligand_filepath.parent.stem
                        for ligand_filepath in ligand_filepaths
                    )
                )
            ]
        if (
            cfg.dataset == "dockgen"
            and cfg.method == "diffdock"
            or (cfg.method == "vina" and cfg.vina_binding_site_method in ["diffdock", "p2rank"])
        ):
            # NOTE: due to its multi-ligand support, e.g., DiffDock groups complexes
            # by the first three parts of their `complex_names`, not the first four as expected
            grouped_protein_filepaths = defaultdict(list)
            for protein_filepath in protein_filepaths:
                protein_id = "_".join(protein_filepath.stem.split("_")[:3])
                grouped_protein_filepaths[protein_id].append(protein_filepath)
            protein_filepaths = [
                protein_filepaths[0] for protein_filepaths in grouped_protein_filepaths.values()
            ]
        if cfg.method in ["diffdock", "rfaa"]:
            ligand_filepaths = [
                ligand_filepath
                for ligand_filepath in ligand_filepaths
                if any(
                    "_".join(ligand_filepath.stem.split("_")[id_slice]) in protein_filepath.stem
                    or "_".join(ligand_filepath.parent.stem.split("_")[id_slice])
                    in protein_filepath.stem
                    for protein_filepath in protein_filepaths
                )
            ]
    assert len(protein_filepaths) == len(
        ligand_filepaths
    ), f"Number of protein ({len(protein_filepaths)}) and ligand ({len(ligand_filepaths)}) files must be equal."

    if cfg.method != "dynamicbind" and not output_file_dir.exists():
        output_file_dir.mkdir(parents=True, exist_ok=True)

    num_processes = cfg.num_processes
    pool = multiprocessing.Pool(processes=num_processes)

    for protein_filepath, ligand_filepath in zip(protein_filepaths, ligand_filepaths):
        assert (
            protein_filepath.stem.split("_")[0] == ligand_filepath.stem.split("_")[0]
            or protein_filepath.parent.stem.split("_")[0]
            == ligand_filepath.parent.stem.split("_")[0]
            or protein_filepath.stem.split("_")[0] == ligand_filepath.parent.stem.split("_")[0]
        ), f"Protein ({protein_filepath}) and ligand ({ligand_filepath}) files must have the same ID."
        if cfg.dataset == "dockgen":
            id_parts = (
                3
                if cfg.method == "diffdock"
                or (
                    cfg.method == "vina" and cfg.vina_binding_site_method in ["diffdock", "p2rank"]
                )
                else 4
            )
            assert (
                "_".join(protein_filepath.stem.split("_")[:id_parts])
                == "_".join(ligand_filepath.stem.split("_")[:id_parts])
                or "_".join(protein_filepath.parent.stem.split("_")[:id_parts])
                == "_".join(ligand_filepath.parent.stem.split("_")[:id_parts])
                or "_".join(protein_filepath.stem.split("_")[:id_parts])
                == "_".join(ligand_filepath.parent.stem.split("_")[:id_parts])
            ), f"Protein ({protein_filepath}) and ligand ({ligand_filepath}) files must have the same ID for DockGen."
        pool.apply_async(
            relax_single_filepair,
            args=(protein_filepath, ligand_filepath, output_file_dir, temp_directory, cfg),
        )

    pool.close()
    pool.join()

    logger.info("Relaxation process complete.")


def relax_single_filepair(
    protein_filepath: Path,
    ligand_filepath: Path,
    output_file_dir: Path,
    temp_directory: Path,
    cfg: DictConfig,
):
    """Relax a single protein-ligand file pair using the specified
    configuration.

    :param protein_filepath: The protein file `Path`.
    :param ligand_filepath: The ligand file `Path`.
    :param output_file_dir: The directory to which to save the output files.
    :param temp_directory: The temporary directory to use for intermediate files.
    :param cfg: The relaxation configuration `DictConfig`.
    """
    try:
        if cfg.method == "dynamicbind":
            output_filepath = Path(
                output_file_dir.parent,
                f"{ligand_filepath.parent.parent.stem}_relaxed",
                f"{ligand_filepath.stem}.sdf",
            )
            protein_output_filepath = Path(
                output_file_dir.parent,
                f"{ligand_filepath.parent.parent.stem}_relaxed",
                f"{protein_filepath.stem}.pdb",
            )
            os.makedirs(output_filepath.parent, exist_ok=True)
            os.makedirs(protein_output_filepath.parent, exist_ok=True)
        elif cfg.method in ["neuralplexer", "flowdock"]:
            output_filepath = Path(
                output_file_dir,
                ligand_filepath.parent.stem,
                f"{ligand_filepath.stem}_relaxed.sdf",
            )
            protein_output_filepath = Path(
                output_file_dir,
                protein_filepath.parent.stem,
                f"{protein_filepath.stem}_relaxed.pdb",
            )
        elif cfg.method == "ensemble":
            output_filepath = Path(
                output_file_dir,
                f"{ligand_filepath.stem}_ensemble_relaxed.sdf",
            )
            protein_output_filepath = Path(
                output_file_dir,
                f"{protein_filepath.stem}_protein_ensemble_relaxed.pdb",
            )
            os.makedirs(output_filepath.parent, exist_ok=True)
        elif cfg.method == "rfaa":
            output_filepath = Path(
                output_file_dir,
                ligand_filepath.stem.replace("_ligand", ""),
                f"{ligand_filepath.stem}_relaxed.sdf",
            )
            protein_output_filepath = Path(
                output_file_dir,
                protein_filepath.stem.replace("_protein", ""),
                f"{protein_filepath.stem}_relaxed.pdb",
            )
        elif cfg.method == "chai-lab":
            output_filepath = Path(
                output_file_dir,
                ligand_filepath.parent.stem,
                f"{ligand_filepath.stem}_relaxed.sdf",
            )
            protein_output_filepath = Path(
                output_file_dir,
                protein_filepath.parent.stem,
                f"{protein_filepath.stem}_relaxed.pdb",
            )
        elif cfg.method == "boltz":
            output_filepath = Path(
                output_file_dir,
                ligand_filepath.parent.stem,
                f"{ligand_filepath.stem}_relaxed.sdf",
            )
            protein_output_filepath = Path(
                output_file_dir,
                protein_filepath.parent.stem,
                f"{protein_filepath.stem}_relaxed.pdb",
            )
        elif cfg.method == "alphafold3":
            output_filepath = Path(
                output_file_dir,
                ligand_filepath.parent.stem,
                f"{ligand_filepath.stem}_relaxed.sdf",
            )
            protein_output_filepath = Path(
                output_file_dir,
                protein_filepath.parent.stem,
                f"{protein_filepath.stem}_relaxed.pdb",
            )
        elif cfg.method == "vina":
            output_filepath = Path(
                output_file_dir,
                ligand_filepath.stem,
                f"{ligand_filepath.stem}_relaxed.sdf",
            )
            protein_output_filepath = Path(
                output_file_dir,
                protein_filepath.stem,
                f"{protein_filepath.stem}_relaxed.pdb",
            )
        else:
            if "rank1" in ligand_filepath.stem:
                # handle for ranked outputs such as those of DiffDock
                output_filepath = Path(
                    output_file_dir,
                    ligand_filepath.parent.stem,
                    f"{ligand_filepath.parent.stem}_relaxed.sdf",
                )
                protein_output_filepath = Path(
                    output_file_dir,
                    ligand_filepath.parent.stem,
                    f"{'_'.join(protein_filepath.stem.split('_')[:2])}_relaxed.pdb",
                )
            else:
                output_filepath = Path(output_file_dir, f"{ligand_filepath.stem}_relaxed.sdf")
                protein_output_filepath = Path(
                    output_file_dir,
                    f"{protein_filepath.stem}_relaxed.pdb",
                )
        if cfg.skip_existing:
            if output_filepath.exists() and cfg.relax_protein and protein_output_filepath.exists():
                logger.info(
                    f"Relaxed protein file `{protein_filepath}` and ligand file `{ligand_filepath}` already exist. Skipping..."
                )
                return
            elif output_filepath.exists():
                logger.info(f"Relaxed ligand file `{output_filepath}` already exists. Skipping...")
                return
        logger.info(
            f"{METHOD_TITLE_MAPPING.get(cfg.method, cfg.method)} {'energy calculation' if cfg.report_initial_energy_only else 'relaxation'} for protein `{protein_filepath}` and ligand `{ligand_filepath}` underway."
        )
        subprocess.run(
            [
                "python",
                os.path.join("posebench", "models", "minimize_energy.py"),
                f"protein_file={protein_filepath}",
                f"ligand_file={ligand_filepath}",
                f"output_file={output_filepath}",
                f"protein_output_file={protein_output_filepath}",
                f"temp_dir={temp_directory}",
                f"add_solvent={cfg.add_solvent}",
                f"name={'null' if not cfg.name else cfg.name}",
                f"prep_only={cfg.prep_only}",
                f"platform={cfg.platform}",
                f"cuda_device_index={cfg.cuda_device_index}",
                f"log_level={cfg.log_level}",
                f"relax_protein={cfg.relax_protein}",
                f"remove_initial_protein_hydrogens={cfg.remove_initial_protein_hydrogens}",
                f"assign_each_ligand_unique_force={cfg.assign_each_ligand_unique_force}",
                f"model_ions={cfg.model_ions or cfg.dataset == 'casp15'}",
                f"cache_files={cfg.cache_files}",
                f"assign_partial_charges_manually={cfg.assign_partial_charges_manually}",
                f"report_initial_energy_only={cfg.report_initial_energy_only}",
                f"max_final_e_value={cfg.max_final_e_value}",
                f"max_num_attempts={cfg.max_num_attempts}",
            ],
            check=True,
        )  # nosec
    except Exception as e:
        raise e
    logger.info(
        f"{METHOD_TITLE_MAPPING.get(cfg.method, cfg.method)} {'energy calculation' if cfg.report_initial_energy_only else 'relaxation'} for protein `{protein_filepath}` and ligand `{ligand_filepath}` complete."
    )


@hydra.main(
    version_base="1.3",
    config_path="../../configs/model",
    config_name="inference_relaxation.yaml",
)
def main(cfg: DictConfig):
    """Run the relaxation inference process using the specified
    configuration."""
    logger.setLevel(cfg.log_level)

    if cfg.v1_baseline:
        with open_dict(cfg):
            cfg.temp_dir = cfg.temp_dir.replace(cfg.method, f"{cfg.method}v1")

    protein_file_dir = Path(cfg.protein_dir)
    ligand_file_dir = Path(cfg.ligand_dir)
    output_file_dir = Path(cfg.output_dir)
    temp_directory = Path(cfg.temp_dir)

    if not protein_file_dir.exists():
        if len(glob.glob(f"{protein_file_dir}_*")) == 0:
            raise FileNotFoundError(
                f"Protein directory (or directories) does (do) not exist: {protein_file_dir}"
            )
    if not ligand_file_dir.exists():
        if len(glob.glob(f"{ligand_file_dir}_*")) == 0:
            raise FileNotFoundError(
                f"Ligand directory (or directories) does (do) not exist: {ligand_file_dir}"
            )

    relax_inference_results(
        protein_file_dir=protein_file_dir,
        ligand_file_dir=ligand_file_dir,
        output_file_dir=output_file_dir,
        temp_directory=temp_directory,
        cfg=cfg,
    )


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
