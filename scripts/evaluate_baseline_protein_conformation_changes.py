# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import glob
import logging
import os

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import rootutils
from omegaconf import DictConfig
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from posebench import register_custom_omegaconf_resolvers, resolve_method_title

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.3",
    config_path="../configs/scripts",
    config_name="evaluate_baseline_protein_conformation_changes.yaml",
)
def main(cfg: DictConfig):
    """Main function for evaluating protein conformation changes of the
    baseline methods."""
    from pymol import cmd

    alignment_results = []
    for item in tqdm(
        glob.glob(os.path.join(cfg.input_protein_structure_dir, "*.pdb")),
        desc="Evaluating protein conformation changes",
    ):
        pdb_id = os.path.basename(item).split("_holo_aligned")[0]
        reference_protein_suffix = "_processed" if cfg.dataset == "dockgen" else ""
        reference_ligand_file_ext = ".pdb" if cfg.dataset == "dockgen" else ".sdf"

        # Parse filepaths
        input_protein_structure_filepath = os.path.join(
            cfg.input_protein_structure_dir,
            f"{pdb_id}_holo_aligned_predicted_protein.pdb",
        )
        reference_protein_structure_filepath = os.path.join(
            cfg.reference_structure_dir, pdb_id, f"{pdb_id}_protein{reference_protein_suffix}.pdb"
        )
        reference_ligand_structure_filepath = os.path.join(
            cfg.reference_structure_dir, pdb_id, f"{pdb_id}_ligand{reference_ligand_file_ext}"
        )

        if cfg.method == "dynamicbind":
            predicted_protein_structure_filepaths = glob.glob(
                os.path.join(
                    os.path.dirname(cfg.predicted_protein_structure_dir),
                    "outputs",
                    "results",
                    f"{cfg.dataset}_{pdb_id}_1",
                    "index0_idx_0",
                    "rank1_receptor_*.pdb",
                )
            )
        else:
            predicted_protein_structure_filepaths = glob.glob(
                os.path.join(
                    cfg.predicted_protein_structure_dir,
                    pdb_id,
                    "prot_rank1_*_aligned.pdb",
                )
            )

        if not predicted_protein_structure_filepaths or not all(
            os.path.exists(filepath)
            for filepath in [
                input_protein_structure_filepath,
                reference_protein_structure_filepath,
                reference_ligand_structure_filepath,
            ]
        ):
            logger.error(
                f"Not all required protein structure files found for {pdb_id}. Skipping this complex."
            )
            continue

        predicted_protein_structure_filepath = predicted_protein_structure_filepaths[0]

        # Find binding site residues in reference structure using PyMOL
        # Refresh PyMOL
        cmd.reinitialize()

        # Load structures
        cmd.load(input_protein_structure_filepath, "input_protein")
        cmd.load(predicted_protein_structure_filepath, "pred_protein")
        cmd.load(reference_protein_structure_filepath, "ref_protein")
        cmd.load(reference_ligand_structure_filepath, "ref_ligand")

        # Select heavy atoms in the reference protein
        cmd.select("ref_protein_heavy", "ref_protein and not elem H")

        # Select heavy atoms in the reference ligand(s)
        cmd.select("ref_ligand_heavy", "ref_ligand and not elem H")

        # Define the reference binding site(s) based on the reference ligand(s)
        cmd.select("binding_site", f"ref_protein_heavy within {cfg.cutoff} of ref_ligand_heavy")

        # Compare the predicted protein binding site to the reference binding site using binding site (pocket) RMSD
        align_cmd = cmd.super if cfg.dataset == "dockgen" else cmd.align
        # NOTE: Since with DockGen we are aligning full predicted bioassemblies
        # to primary interacting chains, we instead use the `super` command to align
        # since it is more robust to large quaternary sequence differences
        input_alignment_result = align_cmd("input_protein", "binding_site", cycles=0, transform=0)
        predicted_alignment_result = cmd.align(
            "pred_protein", "binding_site", cycles=0, transform=0
        )

        # Collect results
        result = {
            "pdb_id": pdb_id,
            "input_binding_site_rmsd": input_alignment_result[0],
            "predicted_binding_site_rmsd": predicted_alignment_result[0],
            "input_num_aligned_atoms": input_alignment_result[1],
            "predicted_num_aligned_atoms": predicted_alignment_result[1],
        }
        alignment_results.append(result)

    # Prepare results
    df = pd.DataFrame(alignment_results)

    # Create the scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(
        df["input_binding_site_rmsd"],
        df["input_binding_site_rmsd"],
        label="AlphaFold 3",
        alpha=0.7,
        color="orange",
        edgecolor="k",
    )
    plt.scatter(
        df["input_binding_site_rmsd"],
        df["predicted_binding_site_rmsd"],
        label=resolve_method_title(cfg.method),
        alpha=0.7,
        color="dodgerblue",
        edgecolor="k",
    )

    # Adding diagonal for reference
    plt.plot([0, 12], [0, 12], linestyle="--", color="gray")

    # Customize the plot
    plt.xlabel("Original pocket RMSD (Å)")
    plt.ylabel("Pocket RMSD (Å)")
    plt.xlim(0, 12)
    plt.ylim(0, 12)
    plt.legend()

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(cfg.output_plot_filepath)
    plt.show()
    plt.close()


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
