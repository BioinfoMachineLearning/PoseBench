# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import glob
import logging
import os
import re
import shutil
import subprocess  # nosec

import hydra
import numpy as np
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
    config_name="flowdock_inference.yaml",
)
def main(cfg: DictConfig):
    """Run inference using a trained FlowDock model checkpoint.

    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    input_csv_path = (
        cfg.input_csv_path.replace(".csv", f"_first_{cfg.max_num_inputs}.csv")
        if cfg.max_num_inputs
        else cfg.input_csv_path
    )

    with open_dict(cfg):
        if cfg.pocket_only_baseline:
            cfg.out_path = os.path.join(
                os.path.dirname(cfg.out_path),
                os.path.basename(cfg.out_path).replace("flowdock", "flowdock_pocket_only"),
            )
            input_csv_path = cfg.input_csv_path.replace("flowdock", "flowdock_pocket_only")

        if cfg.max_num_inputs:
            cfg.out_path = os.path.join(
                os.path.dirname(cfg.out_path),
                os.path.basename(cfg.out_path).replace(
                    "flowdock", f"flowdock_first_{cfg.max_num_inputs}"
                ),
            )

    os.makedirs(cfg.out_path, exist_ok=True)
    assert os.path.exists(input_csv_path), f"Input CSV file `{input_csv_path}` not found."
    if cfg.input_ligand is not None and cfg.input_receptor is not None:
        out_dir = os.path.join(cfg.out_path, cfg.sample_id)
        os.makedirs(out_dir, exist_ok=True)
        try:
            subprocess_args = [
                str(cfg.python_exec_path),
                os.path.join(str(cfg.flowdock_exec_dir), "flowdock", "sample.py"),
                f"sampling_task={cfg.sampling_task}",
                f"sample_id='{cfg.sample_id if cfg.sample_id is not None else 0}'",
                f"input_ligand='{cfg.input_ligand}'",
                f"input_receptor='{cfg.input_receptor}'",
                f"trainer={'gpu' if cfg.cuda_device_index is not None else 'default'}",
                f"{f'trainer.devices=[{int(cfg.cuda_device_index)}]' if cfg.cuda_device_index is not None else ''}",
                f"ckpt_path={cfg.model_checkpoint}",
                f"out_path={out_dir}",
                f"n_samples={int(cfg.n_samples)}",
                f"chunk_size={int(cfg.chunk_size)}",
                f"num_steps={int(cfg.num_steps)}",
                f"sampler={cfg.sampler}",
                f"sampler_eta={cfg.sampler_eta}",
                f"start_time={str(cfg.start_time)}",
                f"max_chain_encoding_k={int(cfg.max_chain_encoding_k)}",
                f"exact_prior={cfg.exact_prior}",
                f"prior_type={cfg.prior_type}",
                f"discard_ligand={cfg.discard_ligand}",
                f"discard_sdf_coords={cfg.discard_sdf_coords}",
                f"detect_covalent={cfg.detect_covalent}",
                f"use_template={cfg.use_template}",
                f"separate_pdb={cfg.separate_pdb}",
                f"rank_outputs_by_confidence={cfg.rank_outputs_by_confidence}",
                f"plddt_ranking_type={cfg.plddt_ranking_type}",
                f"visualize_sample_trajectories={cfg.visualize_sample_trajectories}",
                f"auxiliary_estimation_only={cfg.auxiliary_estimation_only}",
            ]
            if cfg.input_template:
                subprocess_args.append(f"input_template='{cfg.input_template}'")
            if cfg.latent_model:
                subprocess_args.append(f"latent_model={cfg.latent_model}")
            if cfg.csv_path:
                subprocess_args.append(f"csv_path={cfg.csv_path}")
            if cfg.esmfold_chunk_size:
                subprocess_args.append(f"esmfold_chunk_size={int(cfg.esmfold_chunk_size)}")
            subprocess.run(subprocess_args, check=True)  # nosec
        except Exception as e:
            logger.error(
                f"FlowDock inference for complex with protein `{cfg.input_receptor}` and ligand `{cfg.input_ligand}` failed with error: {e}."
            )
            raise e
        logger.info(
            f"FlowDock inference for complex with protein `{cfg.input_receptor}` and ligand `{cfg.input_ligand}` complete."
        )
    else:
        auxiliary_estimation_only = (
            cfg.auxiliary_estimation_only and cfg.auxiliary_estimation_input_dir is not None
        )
        if auxiliary_estimation_only:
            auxiliary_estimation_combined_csv_filepath = os.path.join(
                cfg.auxiliary_estimation_input_dir, "auxiliary_estimation.csv"
            )
            if cfg.skip_existing and os.path.exists(auxiliary_estimation_combined_csv_filepath):
                logger.info(
                    f"Skipping completed auxiliary estimation inference for input directory `{cfg.auxiliary_estimation_input_dir}`."
                )
                return
            assert os.path.exists(
                cfg.auxiliary_estimation_input_dir
            ), "Auxiliary estimation input directory must exist."
            input_rows = []
            target_name = os.path.basename(cfg.auxiliary_estimation_input_dir)
            for item in os.listdir(cfg.auxiliary_estimation_input_dir):
                item_path = os.path.join(cfg.auxiliary_estimation_input_dir, item)
                if os.path.isfile(item_path) and item.endswith(".sdf"):
                    item_basename = os.path.splitext(os.path.basename(item))[0]
                    rank_match = re.search("_rank(.*?)_rmsd", item_basename)
                    assert rank_match, f"Rank index must be present in filename {item}."
                    rank = rank_match.group(1)
                    receptor_path = os.path.join(
                        cfg.auxiliary_estimation_input_dir,
                        f"{item_basename.split('_pbvalid')[0]}.pdb",
                    )
                    input_rows.append(
                        {
                            "id": f"{target_name}_rank{rank}",
                            "input_ligand": item_path,
                            "input_receptor": receptor_path,
                            "input_template": receptor_path,
                        }
                    )
            input_df = pd.DataFrame(input_rows)
        else:
            input_df = pd.read_csv(input_csv_path)
        auxiliary_estimation_csv_filepaths = []
        for _, row in input_df.iterrows():
            out_dir = os.path.join(cfg.out_path, row.id)
            os.makedirs(out_dir, exist_ok=True)
            auxiliary_estimation_csv_filepath = os.path.join(out_dir, "auxiliary_estimation.csv")
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
                if auxiliary_estimation_only:
                    auxiliary_estimation_csv_filepaths.append(auxiliary_estimation_csv_filepath)
                continue
            try:
                subprocess_args = [
                    str(cfg.python_exec_path),
                    os.path.join(str(cfg.flowdock_exec_dir), "flowdock", "sample.py"),
                    f"sampling_task={cfg.sampling_task}",
                    f"sample_id='{cfg.sample_id if cfg.sample_id is not None and not auxiliary_estimation_only else row.id}'",
                    f"input_ligand='{row.input_ligand}'",
                    f"input_receptor='{row.input_receptor}'",
                    f"input_template='{row.input_template}'",
                    f"trainer={'gpu' if cfg.cuda_device_index is not None else 'default'}",
                    f"{f'trainer.devices=[{int(cfg.cuda_device_index)}]' if cfg.cuda_device_index is not None else ''}",
                    f"ckpt_path={cfg.model_checkpoint}",
                    f"out_path={out_dir}",
                    f"n_samples={int(cfg.n_samples)}",
                    f"chunk_size={int(cfg.chunk_size)}",
                    f"num_steps={int(cfg.num_steps)}",
                    f"sampler={cfg.sampler}",
                    f"sampler_eta={cfg.sampler_eta}",
                    f"start_time={str(cfg.start_time)}",
                    f"max_chain_encoding_k={int(cfg.max_chain_encoding_k)}",
                    f"exact_prior={cfg.exact_prior}",
                    f"discard_ligand={cfg.discard_ligand}",
                    f"discard_sdf_coords={cfg.discard_sdf_coords}",
                    f"detect_covalent={cfg.detect_covalent}",
                    f"use_template={cfg.use_template}",
                    f"separate_pdb={cfg.separate_pdb}",
                    f"rank_outputs_by_confidence={cfg.rank_outputs_by_confidence}",
                    f"plddt_ranking_type={cfg.plddt_ranking_type}",
                    f"visualize_sample_trajectories={cfg.visualize_sample_trajectories}",
                    f"auxiliary_estimation_only={cfg.auxiliary_estimation_only}",
                ]
                if cfg.latent_model:
                    subprocess_args.append(f"latent_model={cfg.latent_model}")
                if cfg.csv_path:
                    subprocess_args.append(f"csv_path={cfg.csv_path}")
                if cfg.esmfold_chunk_size:
                    subprocess_args.append(f"esmfold_chunk_size={int(cfg.esmfold_chunk_size)}")
                subprocess.run(subprocess_args, check=True)  # nosec
                if auxiliary_estimation_only:
                    auxiliary_estimation_csv_filepaths.append(auxiliary_estimation_csv_filepath)
            except Exception as e:
                logger.error(
                    f"FlowDock inference for complex with protein `{out_protein_filepath}` and ligand `{out_ligand_filepath}` failed with error: {e}. Skipping..."
                )
                continue
            if not auxiliary_estimation_only:
                out_protein_filepaths = list(glob.glob(out_protein_filepath))
                out_ligand_filepaths = list(glob.glob(out_ligand_filepath))
                assert (
                    len(out_protein_filepaths) == 1
                ), f"Multiple protein files found for complex {row.id}."
                assert (
                    len(out_ligand_filepaths) == 1
                ), f"Multiple ligand files found for complex {row.id}."
                out_protein_filepath = out_protein_filepaths[0]
                out_ligand_filepath = out_ligand_filepaths[0]
            logger.info(
                f"FlowDock inference for complex with protein `{out_protein_filepath}` and ligand `{out_ligand_filepath}` complete."
            )
        if auxiliary_estimation_only:
            # combine auxiliary estimation CSV files
            combined_auxiliary_estimation_df = pd.concat(
                [pd.read_csv(item) for item in auxiliary_estimation_csv_filepaths]
            )
            # install integer rank values
            combined_auxiliary_estimation_df["rank"] = np.arange(
                1, len(combined_auxiliary_estimation_df) + 1
            )
            combined_auxiliary_estimation_df.to_csv(
                auxiliary_estimation_combined_csv_filepath, index=False
            )
            for row_id in input_df.id:
                # clean up (now-)redundant files
                shutil.rmtree(os.path.join(cfg.out_path, row_id))
            logger.info(
                f"FlowDock auxiliary estimation inference for input directory `{cfg.auxiliary_estimation_input_dir}` complete."
            )


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
