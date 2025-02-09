# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import glob
import logging
import os
import subprocess  # nosec

import hydra
import pandas as pd
import rootutils
from omegaconf import DictConfig, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from posebench import register_custom_omegaconf_resolvers

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.3",
    config_path="../../configs/model",
    config_name="neuralplexer_inference.yaml",
)
def main(cfg: DictConfig):
    """Run inference using a trained NeuralPLexer model checkpoint.

    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    input_csv_path = (
        cfg.input_csv_path.replace(".csv", f"_first_{cfg.max_num_inputs}.csv")
        if cfg.max_num_inputs
        else cfg.input_csv_path
    )

    with open_dict(cfg):
        if cfg.no_ilcl:
            cfg.frozen_prot = True
            cfg.model_checkpoint = os.path.join(
                os.path.dirname(cfg.model_checkpoint),
                "pdbbind_finetuned",
                "rigid_docking_base.ckpt",
            )
            cfg.out_path = os.path.join(
                os.path.dirname(cfg.out_path),
                os.path.basename(cfg.out_path).replace("neuralplexer", "neuralplexer_no_ilcl"),
            )
            assert os.path.exists(
                cfg.model_checkpoint
            ), f"Model checkpoint trained without an inter-ligand clash loss (ILCL) `{cfg.model_checkpoint}` not found."

        if cfg.pocket_only_baseline:
            cfg.out_path = os.path.join(
                os.path.dirname(cfg.out_path),
                os.path.basename(cfg.out_path).replace("neuralplexer", "neuralplexer_pocket_only"),
            )
            input_csv_path = cfg.input_csv_path.replace("neuralplexer", "neuralplexer_pocket_only")

        if cfg.max_num_inputs:
            cfg.out_path = os.path.join(
                os.path.dirname(cfg.out_path),
                os.path.basename(cfg.out_path).replace(
                    "neuralplexer", f"neuralplexer_first_{cfg.max_num_inputs}"
                ),
            )

    os.makedirs(cfg.out_path, exist_ok=True)
    assert os.path.exists(input_csv_path), f"Input CSV file `{input_csv_path}` not found."
    for _, row in pd.read_csv(input_csv_path).iterrows():
        out_dir = os.path.join(cfg.out_path, row.id)
        os.makedirs(out_dir, exist_ok=True)
        out_protein_filepath = os.path.join(out_dir, "prot_rank1_*.pdb")
        out_ligand_filepath = os.path.join(out_dir, "lig_rank1_*.sdf")
        out_protein_filepaths = [
            item
            for item in glob.glob(out_protein_filepath)
            if "_aligned" not in os.path.basename(item)
        ]
        out_ligand_filepaths = [
            item
            for item in glob.glob(out_ligand_filepath)
            if "_aligned" not in os.path.basename(item)
        ]
        if cfg.skip_existing and out_protein_filepaths and out_ligand_filepaths:
            assert (
                len(out_protein_filepaths) == 1
            ), f"Multiple protein files found for complex {row.id}."
            assert (
                len(out_ligand_filepaths) == 1
            ), f"Multiple ligand files found for complex {row.id}."
            out_protein_filepath = out_protein_filepaths[0]
            out_ligand_filepath = out_ligand_filepaths[0]
            logger.info(
                f"Skipping inference for completed complex with protein `{out_protein_filepath}` and ligand `{out_ligand_filepath}`."
            )
            continue
        try:
            subprocess_args = [
                str(cfg.python_exec_path),
                os.path.join(str(cfg.neuralplexer_exec_dir), "neuralplexer", "inference.py"),
                "--task",
                str(cfg.task),
                "--sample-id",
                str(cfg.sample_id),
                "--template-id",
                str(cfg.template_id),
                "--cuda-device-index",
                str(cfg.cuda_device_index),
                "--model-checkpoint",
                str(cfg.model_checkpoint),
                "--input-ligand",
                str(row.input_ligand),
                "--input-receptor",
                str(row.input_receptor),
                "--input-template",
                str(row.input_template),
                "--out-path",
                str(out_dir),
                "--n-samples",
                str(cfg.n_samples),
                "--chunk-size",
                str(cfg.chunk_size),
                "--num-steps",
                str(cfg.num_steps),
                "--sampler",
                str(cfg.sampler),
                "--start-time",
                str(cfg.start_time),
                "--max-chain-encoding-k",
                str(cfg.max_chain_encoding_k),
                "--plddt-ranking-type",
                str(cfg.plddt_ranking_type),
            ]
            if cfg.latent_model:
                subprocess_args.extend(["--latent-model", cfg.latent_model])
            if cfg.exact_prior:
                subprocess_args.extend(["--exact-prior"])
            if cfg.discard_ligand:
                subprocess_args.extend(["--discard-ligand"])
            if cfg.discard_sdf_coords:
                subprocess_args.extend(["--discard-sdf-coords"])
            if cfg.detect_covalent:
                subprocess_args.extend(["--detect-covalent"])
            if cfg.use_template:
                subprocess_args.extend(["--use-template"])
            if cfg.separate_pdb:
                subprocess_args.extend(["--separate-pdb"])
            if cfg.rank_outputs_by_confidence:
                subprocess_args.extend(["--rank-outputs-by-confidence"])
            if cfg.frozen_prot:
                subprocess_args.extend(["--frozen-prot"])
            if cfg.csv_path:
                subprocess_args.extend(["--csv-path", cfg.csv_path])
            subprocess.run(subprocess_args, check=True)  # nosec
        except Exception as e:
            logger.error(
                f"NeuralPLexer inference for complex with protein `{out_protein_filepath}` and ligand `{out_ligand_filepath}` failed with error: {e}. Skipping..."
            )
            continue
        out_protein_filepaths = list(glob.glob(out_protein_filepath))
        out_ligand_filepaths = list(glob.glob(out_ligand_filepath))
        assert (
            len(out_protein_filepaths) == 1
        ), f"Multiple protein files found for complex {row.id}."
        assert len(out_ligand_filepaths) == 1, f"Multiple ligand files found for complex {row.id}."
        out_protein_filepath = out_protein_filepaths[0]
        out_ligand_filepath = out_ligand_filepaths[0]
        logger.info(
            f"NeuralPLexer inference for complex with protein `{out_protein_filepath}` and ligand `{out_ligand_filepath}` complete."
        )


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
