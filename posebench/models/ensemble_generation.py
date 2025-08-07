# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import ast
import copy
import glob
import logging
import multiprocessing
import os
import re
import shutil
import subprocess  # nosec
import tempfile
from io import StringIO
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import rootutils
from beartype.typing import Any, Dict, List, Literal, Optional, Tuple, Union
from Bio import PDB
from Bio.PDB import Structure
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.PDBParser import PDBParser
from omegaconf import DictConfig, OmegaConf, open_dict
from posebusters import PoseBusters
from rdkit import Chem
from rdkit.Chem import AllChem

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from posebench import register_custom_omegaconf_resolvers
from posebench.analysis.complex_alignment import align_complex_to_protein_only
from posebench.data.components.protein_apo_to_holo_alignment import read_molecule
from posebench.models.inference_relaxation import relax_single_filepair
from posebench.models.minimize_energy import minimize_energy
from posebench.utils.data_utils import (
    extract_sequences_from_protein_structure_file,
    renumber_biopython_structure_residues,
)
from posebench.utils.model_utils import calculate_rmsd

METHODS_PREDICTING_HOLO_PROTEIN_AB_INITIO = {
    "neuralplexer",
    "flowdock",
    "rfaa",
    "chai-lab",
    "boltz",
    "alphafold3",
}

ENSEMBLE_PREDICTIONS = Dict[str, List[Tuple[str, str]]]
RANKED_ENSEMBLE_PREDICTIONS = Dict[int, Tuple[str, str, str, float]]

# NOTE: the following sequence is derived from `5S8I_2LY.pdb` of the PoseBusters Benchmark set
LIGAND_ONLY_RECEPTOR_PLACEHOLDER_SEQUENCE = "DSLFAGLVGEYYGTNSQLNNISDFRALVDSKEADATFEAANISYGRGSSDVAKGTHLQEFLGSDASTLSTDPGDNTDGGIYLQGYVYLEAGTYNFKVTADDGYEITINGNPVATVDNNQSVYTVTHASFTISESGYQAIDMIWWDQGGDYVFQPTLSADGGSTYFVLDSAILSSTGETPY"


def create_temporary_fasta_file(protein_sequence: str, name: Optional[str] = None) -> str:
    """Create a temporary FASTA file for the input protein sequence.

    :param protein_sequence: Amino acid sequence of the protein.
    :param name: Optional name of the temporary FASTA file.
    :return: Path to the temporary FASTA file.
    """
    name = name if name else "temp"
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fasta") as fasta_file:
        fasta_file.write(f">{name}\n{protein_sequence}\n")
        temp_fasta_file = fasta_file.name
    return temp_fasta_file


def predict_protein_structure_from_sequence(
    python_exec_path: str,
    structure_prediction_script_path: str,
    fasta_filepath: str,
    output_pdb_dir: str,
    chunk_size: Optional[int] = None,
    cpu_only: bool = False,
    cpu_offload: bool = False,
    cuda_device_index: int = 0,
):
    """Predict protein structure from amino acid sequence.

    :param python_exec_path: Path to the Python executable with which to
        run Python scripts.
    :param structure_prediction_script_path: Path to the ESMFold
        structure prediction script to run.
    :param fasta_filepath: Path to the input FASTA file.
    :param output_pdb_dir: Path to the output PDB directory.
    :param chunk_size: Optional chunk size for structure prediction.
    :param cpu_only: Whether to use CPU only for structure prediction.
    :param cpu_offload: Whether to use CPU offloading for structure
        prediction.
    :param cuda_device_index: The optional index of the CUDA device to
        use for structure prediction.
    """
    cmd_inputs = [
        python_exec_path,
        structure_prediction_script_path,
        "--fasta",
        fasta_filepath,
        "--pdb",
        output_pdb_dir,
        "--cuda-device-index",
        str(cuda_device_index),
    ]
    if chunk_size:
        cmd_inputs.extend(["--chunk-size", str(chunk_size)])
    if cpu_only:
        cmd_inputs.append("--cpu-only")
    if cpu_offload:
        cmd_inputs.append("--cpu-offload")
    try:
        subprocess.run(cmd_inputs, check=True)  # nosec
    except Exception as e:
        raise e
    logger.info(
        f"ESMFold structure prediction for input protein sequence has been saved to {output_pdb_dir}."
    )


def insert_hpc_headers(
    method: str,
    gpu_partition: str = "chengji-lab-gpu",
    gpu_account: str = "chengji-lab",
    gpu_type: Literal["A100", "H100", ""] = "",
    cpu_memory_in_gb: int = 59,
    time_limit: str = "7-00:00:00",
) -> str:
    """Insert batch headers for SLURM job scheduling.

    :param method: Name of the method for which to generate a prediction
        script.
    :param gpu_partition: Name of the GPU partition to use.
    :param gpu_account: Name of the GPU account to use.
    :param cpu_memory_in_gb: Amount of CPU memory to request in GB.
    :param time_limit: Time limit for the job as a SLURM-compatible
        string.
    :return: Batch headers string for SLURM job scheduling.
    """
    return f"""######################### Batch Headers #########################
#SBATCH --partition {gpu_partition} # use reserved partition `chengji-lab-gpu`
#SBATCH --account {gpu_account}  # NOTE: this must be specified to use the reserved partition above
#SBATCH --nodes=1              # NOTE: this needs to match Lightning's `Trainer(num_nodes=...)`
#SBATCH --gres gpu:{f'{gpu_type}:' if gpu_type else ''}1      # request {gpu_type} GPU resource(s)
#SBATCH --ntasks-per-node=1    # NOTE: this needs to be `1` on SLURM clusters when using Lightning's `ddp_spawn` strategy`; otherwise, set to match Lightning's quantity of `Trainer(devices=...)`
#SBATCH --mem={cpu_memory_in_gb}G              # NOTE: use `--mem=0` to request all memory "available" on the assigned node
#SBATCH -t {time_limit}          # time limit for the job (up to two days: `2-00:00:00`)
#SBATCH -J posebench_{method}_ensembling # job name
#SBATCH --output=R-%x.%j.out   # output log file
#SBATCH --error=R-%x.%j.err    # error log file

module purge
module load cuda/11.8.0_gcc_9.5.0

# determine location of the project directory
use_private_project_dir=false # NOTE: customize as needed
if [ "$use_private_project_dir" = true ]; then
    project_dir="/home/$USER/data/Repositories/Lab_Repositories/PoseBench"
else
    project_dir="/cluster/pixstor/chengji-lab/$USER/Repositories/Lab_Repositories/PoseBench"
fi

# shellcheck source=/dev/null
source /home/$USER/mambaforge/etc/profile.d/conda.sh

cd "$project_dir" || exit"""


def create_diffdock_bash_script(
    protein_filepath: str,
    ligand_smiles: str,
    input_id: str,
    output_filepath: str,
    cfg: DictConfig,
    generate_hpc_scripts: bool = True,
):
    """Create a bash script to run DiffDock protein-ligand complex prediction.

    :param protein_filepath: Path to the input protein structure PDB
        file.
    :param ligand_smiles: SMILES string of the input ligand.
    :param input_id: Input ID.
    :param output_filepath: Path to the output bash script file.
    :param cfg: Configuration dictionary for runtime arguments.
    :param generate_hpc_scripts: Whether to generate HPC scripts for
        DiffDock.
    """
    bash_script_content = f"""#!/bin/bash -l
{insert_hpc_headers(method='diffdock') if generate_hpc_scripts else 'source /home/$USER/mambaforge/etc/profile.d/conda.sh'}
conda activate {"$project_dir/PoseBench/" if generate_hpc_scripts else 'PoseBench'}

# command to run diffdock_input_preparation.py
python posebench/data/diffdock_input_preparation.py \\
    output_csv_path={cfg.diffdock_input_csv_path} \\
    protein_filepath="{protein_filepath}" \\
    ligand_smiles='"{ligand_smiles}"' \\
    input_id="{input_id}"

# command to run diffdock_inference.py
echo "Calling diffdock_inference.py!"
{cfg.diffdock_python_exec_path} posebench/models/diffdock_inference.py \\
    cuda_device_index={cfg.cuda_device_index} \\
    python_exec_path={cfg.diffdock_python_exec_path} \\
    diffdock_exec_dir={cfg.diffdock_exec_dir} \\
    input_csv_path={cfg.diffdock_input_csv_path} \\
    output_dir={cfg.diffdock_output_dir} \\
    model_dir={cfg.diffdock_model_dir} \\
    confidence_model_dir={cfg.diffdock_confidence_model_dir} \\
    inference_steps={cfg.diffdock_inference_steps} \\
    samples_per_complex={min(cfg.diffdock_samples_per_complex, cfg.max_method_predictions)} \\
    batch_size={cfg.diffdock_batch_size} \\
    actual_steps={cfg.diffdock_actual_steps} \\
    no_final_step_noise={cfg.diffdock_no_final_step_noise} \\
    skip_existing={cfg.diffdock_skip_existing}

echo "Finished calling diffdock_inference.py!"
    """

    with open(output_filepath, "w") as file:
        file.write(bash_script_content)

    logger.info(f"Bash script '{output_filepath}' created successfully.")


def create_dynamicbind_bash_script(
    protein_filepath: str,
    ligand_smiles: str,
    output_filepath: str,
    cfg: DictConfig,
    generate_hpc_scripts: bool = True,
):
    """Create a bash script to run DynamicBind protein-ligand complex
    prediction.

    :param protein_filepath: Path to the input protein structure PDB
        file.
    :param ligand_smiles: SMILES string of the input ligand.
    :param output_filepath: Path to the output bash script file.
    :param cfg: Configuration dictionary for runtime arguments.
    :param generate_hpc_scripts: Whether to generate HPC scripts for
        DynamicBind.
    """
    bash_script_content = f"""#!/bin/bash
{insert_hpc_headers(method='dynamicbind') if generate_hpc_scripts else 'source /home/$USER/mambaforge/etc/profile.d/conda.sh'}
conda activate {"$project_dir/PoseBench/" if generate_hpc_scripts else 'PoseBench'}

# command to run dynamicbind_input_preparation.py
python posebench/data/dynamicbind_input_preparation.py \\
    output_csv_dir={cfg.dynamicbind_input_ligand_csv_dir} \\
    input_protein_data_dir={cfg.dynamicbind_input_protein_data_dir} \\
    protein_filepath="{protein_filepath}" \\
    ligand_smiles='"{ligand_smiles}"'

# command to run dynamicbind_inference.py
echo "Calling dynamicbind_inference.py!"
{cfg.dynamicbind_python_exec_path} posebench/models/dynamicbind_inference.py \\
    cuda_device_index={cfg.cuda_device_index} \\
    python_exec_path={cfg.dynamicbind_python_exec_path} \\
    dynamicbind_exec_dir={cfg.dynamicbind_exec_dir} \\
    dataset={cfg.dynamicbind_dataset} \\
    input_data_dir={cfg.dynamicbind_input_protein_data_dir} \\
    input_ligand_csv_dir={cfg.dynamicbind_input_ligand_csv_dir} \\
    samples_per_complex={min(cfg.dynamicbind_samples_per_complex, cfg.max_method_predictions)} \\
    savings_per_complex={cfg.dynamicbind_savings_per_complex} \\
    inference_steps={cfg.dynamicbind_inference_steps} \\
    batch_size={cfg.dynamicbind_batch_size} \\
    header={cfg.dynamicbind_header} \\
    num_workers={cfg.dynamicbind_num_workers} \\
    skip_existing={cfg.dynamicbind_skip_existing}

echo "Finished calling dynamicbind_inference.py!"
    """

    with open(output_filepath, "w") as file:
        file.write(bash_script_content)

    logger.info(f"Bash script '{output_filepath}' created successfully.")


def create_neuralplexer_bash_script(
    protein_filepath: str,
    ligand_smiles: str,
    input_id: str,
    output_filepath: str,
    cfg: DictConfig,
    generate_hpc_scripts: bool = True,
):
    """Create a bash script to run NeuralPLexer protein-ligand complex
    prediction.

    :param protein_filepath: Path to the input protein structure PDB
        file.
    :param ligand_smiles: SMILES string of the input ligand.
    :param input_id: Input ID.
    :param output_filepath: Path to the output bash script file.
    :param cfg: Configuration dictionary for runtime arguments.
    :param generate_hpc_scripts: Whether to generate HPC scripts for
        NeuralPLexer.
    """
    bash_script_content = f"""#!/bin/bash
{insert_hpc_headers(method='neuralplexer') if generate_hpc_scripts else 'source /home/$USER/mambaforge/etc/profile.d/conda.sh'}
conda activate {"$project_dir/PoseBench/" if generate_hpc_scripts else 'PoseBench'}

# command to run neuralplexer_input_preparation.py
python posebench/data/neuralplexer_input_preparation.py \\
    output_csv_path={cfg.neuralplexer_input_csv_path} \\
    input_receptor='{protein_filepath}' \\
    input_ligand='"{ligand_smiles}"' \\
    input_template='{protein_filepath}' \\
    input_id='{input_id}'

# command to run neuralplexer_inference.py
echo "Calling neuralplexer_inference.py!"
{cfg.neuralplexer_python_exec_path} posebench/models/neuralplexer_inference.py \\
    python_exec_path={cfg.neuralplexer_python_exec_path} \\
    neuralplexer_exec_dir={cfg.neuralplexer_exec_dir} \\
    input_csv_path={cfg.neuralplexer_input_csv_path} \\
    skip_existing={cfg.neuralplexer_skip_existing} \\
    task={cfg.neuralplexer_task} \\
    sample_id={cfg.neuralplexer_sample_id} \\
    template_id={cfg.neuralplexer_template_id} \\
    cuda_device_index={cfg.cuda_device_index} \\
    model_checkpoint={cfg.neuralplexer_model_checkpoint} \\
    out_path={cfg.neuralplexer_out_path} \\
    n_samples={min(cfg.neuralplexer_n_samples, cfg.max_method_predictions)} \\
    chunk_size={cfg.neuralplexer_chunk_size} \\
    num_steps={cfg.neuralplexer_num_steps} \\
    sampler={cfg.neuralplexer_sampler} \\
    start_time={cfg.neuralplexer_start_time} \\
    max_chain_encoding_k={cfg.neuralplexer_max_chain_encoding_k} \\
    exact_prior={cfg.neuralplexer_exact_prior} \\
    discard_ligand={cfg.neuralplexer_discard_ligand} \\
    discard_sdf_coords={cfg.neuralplexer_discard_sdf_coords} \\
    detect_covalent={cfg.neuralplexer_detect_covalent} \\
    use_template={cfg.neuralplexer_use_template} \\
    separate_pdb={cfg.neuralplexer_separate_pdb} \\
    rank_outputs_by_confidence={cfg.neuralplexer_rank_outputs_by_confidence} \\
    plddt_ranking_type={cfg.neuralplexer_plddt_ranking_type}

echo "Finished calling neuralplexer_inference.py!"
    """

    with open(output_filepath, "w") as file:
        file.write(bash_script_content)

    logger.info(f"Bash script '{output_filepath}' created successfully.")


def create_flowdock_bash_script(
    protein_filepath: str,
    ligand_smiles: str,
    input_id: str,
    output_filepath: str,
    cfg: DictConfig,
    generate_hpc_scripts: bool = True,
):
    """Create a bash script to run FlowDock protein-ligand complex prediction.

    :param protein_filepath: Path to the input protein structure PDB
        file.
    :param ligand_smiles: SMILES string of the input ligand.
    :param input_id: Input ID.
    :param output_filepath: Path to the output bash script file.
    :param cfg: Configuration dictionary for runtime arguments.
    :param generate_hpc_scripts: Whether to generate HPC scripts for
        FlowDock.
    """
    bash_script_content = f"""#!/bin/bash
{insert_hpc_headers(method='flowdock') if generate_hpc_scripts else 'source /home/$USER/mambaforge/etc/profile.d/conda.sh'}
conda activate {"$project_dir/PoseBench/" if generate_hpc_scripts else 'PoseBench'}

# command to run flowdock_input_preparation.py
python posebench/data/flowdock_input_preparation.py \\
    output_csv_path={cfg.flowdock_input_csv_path} \\
    input_receptor='{protein_filepath}' \\
    input_ligand='"{ligand_smiles}"' \\
    input_template='{protein_filepath}' \\
    input_id='{input_id}'

# command to run flowdock_inference.py
echo "Calling flowdock_inference.py!"
{cfg.flowdock_python_exec_path} posebench/models/flowdock_inference.py \\
    python_exec_path={cfg.flowdock_python_exec_path} \\
    flowdock_exec_dir={cfg.flowdock_exec_dir} \\
    input_data_dir={cfg.flowdock_input_data_dir} \\
    input_receptor_structure_dir={cfg.flowdock_input_receptor_structure_dir} \\
    input_csv_path={cfg.flowdock_input_csv_path} \\
    skip_existing={cfg.flowdock_skip_existing} \\
    sampling_task={cfg.flowdock_sampling_task} \\
    sample_id={cfg.flowdock_sample_id} \\
    input_receptor={cfg.flowdock_input_receptor} \\
    input_ligand={cfg.flowdock_input_ligand} \\
    input_template={cfg.flowdock_input_template} \\
    model_checkpoint={cfg.flowdock_model_checkpoint} \\
    out_path={cfg.flowdock_out_path} \\
    n_samples={min(cfg.flowdock_n_samples, cfg.max_method_predictions)} \\
    chunk_size={cfg.flowdock_chunk_size} \\
    num_steps={cfg.flowdock_num_steps} \\
    latent_model={cfg.flowdock_latent_model} \\
    sampler={cfg.flowdock_sampler} \\
    sampler_eta={cfg.flowdock_sampler_eta} \\
    start_time={cfg.flowdock_start_time} \\
    max_chain_encoding_k={cfg.flowdock_max_chain_encoding_k} \\
    exact_prior={cfg.flowdock_exact_prior} \\
    prior_type={cfg.flowdock_prior_type} \\
    discard_ligand={cfg.flowdock_discard_ligand} \\
    discard_sdf_coords={cfg.flowdock_discard_sdf_coords} \\
    detect_covalent={cfg.flowdock_detect_covalent} \\
    use_template={cfg.flowdock_use_template} \\
    separate_pdb={cfg.flowdock_separate_pdb} \\
    rank_outputs_by_confidence={cfg.flowdock_rank_outputs_by_confidence} \\
    plddt_ranking_type={cfg.flowdock_plddt_ranking_type} \\
    visualize_sample_trajectories={cfg.flowdock_visualize_sample_trajectories} \\
    auxiliary_estimation_only={cfg.flowdock_auxiliary_estimation_only} \\
    auxiliary_estimation_input_dir={cfg.flowdock_auxiliary_estimation_input_dir} \\
    csv_path={cfg.flowdock_csv_path} \\
    esmfold_chunk_size={cfg.flowdock_esmfold_chunk_size}

echo "Finished calling flowdock_inference.py!"
    """

    with open(output_filepath, "w") as file:
        file.write(bash_script_content)

    logger.info(f"Bash script '{output_filepath}' created successfully.")


def rfaa_get_chain_letter(index: int) -> str:
    """Get the RFAA chain letter based on index."""
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    num_chars = len(alphabet)
    num_single_chars = num_chars - 1  # Exclude 'z'
    if index <= num_single_chars:
        return alphabet[index - 1]
    else:
        # Use multiple characters for chain IDs beyond 'z', such as 'aa', 'ab', 'ac', etc.
        num_multiple_chars = index - num_single_chars
        num_full_sets = num_multiple_chars // num_chars
        remainder = num_multiple_chars % num_chars
        if remainder == 0:
            return alphabet[-1] * num_full_sets
        else:
            return alphabet[-1] * num_full_sets + alphabet[remainder - 1]


def dynamically_build_rfaa_input_config(
    fasta_filepaths: List[str],
    sdf_filepaths: Optional[List[str]],
    input_id: str,
    cfg: DictConfig,
    smiles_strings: Optional[List[str]] = None,
) -> str:
    """Dynamically build the RoseTTAFold-All-Atom inference configuration file
    for input proteins and ligands.

    :param fasta_filepaths: List of FASTA filepaths.
    :param sdf_filepaths: List of optional SDF filepaths.
    :param input_id: Input ID.
    :param cfg: Configuration dictionary for runtime arguments.
    :param smiles_strings: Optional list of SMILES strings of the input
        ligands to use directly.
    :return: Path to the dynamically built configuration file.
    """
    # Build protein_inputs section dynamically
    protein_inputs_content = ""
    for i, fasta_filepath in enumerate(fasta_filepaths, start=1):
        chain_letter = rfaa_get_chain_letter(i)
        protein_inputs_content += f"""  {chain_letter}:
    fasta_file: {fasta_filepath}
"""

    # Build sm_inputs section dynamically
    sm_inputs_content = ""
    if smiles_strings is None:
        assert (
            sdf_filepaths is not None
        ), "No SMILES strings or SDF filepaths found for input ligands."
        smiles_strings = [
            Chem.MolToSmiles(Chem.MolFromMolFile(sdf_filepath)) for sdf_filepath in sdf_filepaths
        ]
    assert len(smiles_strings) > 0, "No SMILES strings found for input ligands."
    for i, smiles in enumerate(
        smiles_strings, start=len(fasta_filepaths) + 1
    ):  # Starting index from len(fasta_filepaths)+1
        chain_letter = rfaa_get_chain_letter(i)
        sm_inputs_content += f"""  {chain_letter}:
    input: '{smiles}'
    input_type: "smiles"
"""

    config_file_content = f"""# RoseTTAFold-All-Atom inference configuration file for input '{input_id}'
defaults:
  - base
job_name: "{input_id}"

protein_inputs:
  {protein_inputs_content.strip()}

sm_inputs:
  {sm_inputs_content.strip()}
"""

    config_filepath = os.path.join(
        (cfg.rfaa_config_dir if cfg.get("rfaa_config_dir") else cfg.config_dir),
        f"{input_id}_rfaa_inference.yaml",
    )
    with open(config_filepath, "w") as file:
        file.write(config_file_content)

    return config_filepath


def create_rfaa_bash_script(
    fasta_filepaths: List[str],
    sdf_filepaths: Optional[List[str]],
    input_id: str,
    cfg: DictConfig,
    output_filepath: Optional[str] = None,
    smiles_strings: Optional[List[str]] = None,
    generate_hpc_scripts: bool = True,
):
    """Create a bash script to run RoseTTAFold-All-Atom protein-ligand complex
    prediction.

    :param fasta_filepaths: List of FASTA filepaths.
    :param sdf_filepaths: List of optional SDF filepaths.
    :param input_id: Input ID.
    :param cfg: Configuration dictionary for runtime arguments.
    :param output_filepath: Optional path to the output bash script
        file.
    :param smiles_strings: Optional list of SMILES strings of the input
        ligands to use directly.
    :param generate_hpc_scripts: Whether to generate HPC scripts for
        RoseTTAFold-All-Atom.
    """

    if output_filepath is None:
        output_filepath = os.path.join(cfg.output_dir, input_id, f"{input_id}_rfaa_inference.sh")
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    # Dynamically build the RoseTTAFold-All-Atom inference configuration file
    config_filepath = dynamically_build_rfaa_input_config(
        fasta_filepaths=fasta_filepaths,
        sdf_filepaths=sdf_filepaths,
        input_id=input_id,
        cfg=cfg,
        smiles_strings=smiles_strings,
    )

    bash_script_content = f"""#!/bin/bash -l
{insert_hpc_headers(method='rfaa', time_limit='0-12:00:00') if generate_hpc_scripts else 'source /home/$USER/mambaforge/etc/profile.d/conda.sh'}
conda activate {"$project_dir/forks/RoseTTAFold-All-Atom/RFAA/" if generate_hpc_scripts else 'forks/RoseTTAFold-All-Atom/RFAA/'}
echo "Beginning RoseTTAFold-All-Atom inference for input '{input_id}'!"

# command to run RoseTTAFold-All-Atom inference
cd {cfg.rfaa_exec_dir}
{cfg.rfaa_python_exec_path} -m rf2aa.run_inference --config-name {os.path.basename(config_filepath)} \\
    loader_params.MAXCYCLE={cfg.rfaa_max_cycles} \\
    output_path={os.path.dirname(output_filepath)}

# clean up temporary config file
rm {os.path.join(cfg.rfaa_config_dir, f"{input_id}_rfaa_inference.yaml")}
rm -r {os.path.join(cfg.rfaa_config_dir, input_id)}
echo "Finished RoseTTAFold-All-Atom inference for input '{input_id}'!"
"""

    with open(output_filepath, "w") as file:
        file.write(bash_script_content)

    logger.info(f"Bash script '{output_filepath}' created successfully.")


def create_chai_bash_script(
    protein_filepath: str,
    ligand_smiles: str,
    input_id: str,
    cfg: DictConfig,
    output_filepath: Optional[str] = None,
    generate_hpc_scripts: bool = True,
):
    """Create a bash script to run Chai-1 protein-ligand complex prediction.

    :param protein_filepath: Path to the input protein structure PDB
        file.
    :param ligand_smiles: SMILES string of the input ligand.
    :param input_id: Input ID.
    :param cfg: Configuration dictionary for runtime arguments.
    :param output_filepath: Optional path to the output bash script
        file.
    :param generate_hpc_scripts: Whether to generate HPC scripts for
        RoseTTAFold-All-Atom.
    """

    if output_filepath is None:
        output_filepath = os.path.join(cfg.output_dir, input_id, f"{input_id}_rfaa_inference.sh")
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    bash_script_content = f"""#!/bin/bash -l
{insert_hpc_headers(method='chai-lab', time_limit='0-12:00:00') if generate_hpc_scripts else 'source /home/$USER/mambaforge/etc/profile.d/conda.sh'}
conda activate {"$project_dir/forks/chai-lab/chai-lab/" if generate_hpc_scripts else 'forks/chai-lab/chai-lab/'}
echo "Beginning Chai-1 inference for input '{input_id}'!"

# command to run chai_input_preparation.py
python posebench/data/chai_input_preparation.py \\
    dataset=ensemble \\
    protein_filepath='{protein_filepath}' \\
    ligand_smiles='"{ligand_smiles}"' \\
    input_id='{input_id}'

# command to run chai_inference.py
echo "Calling chai_inference.py!"
python posebench/models/chai_inference.py \\
    dataset=ensemble \\
    cuda_device_index={cfg.cuda_device_index} \\
    skip_existing={cfg.chai_skip_existing}

echo "Finished calling chai_inference.py!"
    """

    with open(output_filepath, "w") as file:
        file.write(bash_script_content)

    logger.info(f"Bash script '{output_filepath}' created successfully.")


def create_boltz_bash_script(
    protein_filepath: str,
    ligand_smiles: str,
    input_id: str,
    cfg: DictConfig,
    output_filepath: Optional[str] = None,
    generate_hpc_scripts: bool = True,
):
    """Create a bash script to run Boltz protein-ligand complex prediction.

    :param protein_filepath: Path to the input protein structure PDB
        file.
    :param ligand_smiles: SMILES string of the input ligand.
    :param input_id: Input ID.
    :param cfg: Configuration dictionary for runtime arguments.
    :param output_filepath: Optional path to the output bash script
        file.
    :param generate_hpc_scripts: Whether to generate HPC scripts for
        RoseTTAFold-All-Atom.
    """

    if output_filepath is None:
        output_filepath = os.path.join(cfg.output_dir, input_id, f"{input_id}_rfaa_inference.sh")
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    bash_script_content = f"""#!/bin/bash -l
{insert_hpc_headers(method='boltz', time_limit='0-24:00:00') if generate_hpc_scripts else 'source /home/$USER/mambaforge/etc/profile.d/conda.sh'}
conda activate {"$project_dir/forks/boltz/boltz/" if generate_hpc_scripts else 'forks/boltz/boltz/'}
echo "Beginning Boltz inference for input '{input_id}'!"

# command to run boltz_input_preparation.py
python posebench/data/boltz_input_preparation.py \\
    dataset=ensemble \\
    protein_filepath='{protein_filepath}' \\
    ligand_smiles='"{ligand_smiles}"' \\
    input_id='{input_id}'

# command to run boltz_inference.py
echo "Calling boltz_inference.py!"
python posebench/models/boltz_inference.py \\
    dataset=ensemble \\
    cuda_device_index={cfg.cuda_device_index} \\
    skip_existing={cfg.boltz_skip_existing}

echo "Finished calling boltz_inference.py!"
    """

    with open(output_filepath, "w") as file:
        file.write(bash_script_content)

    logger.info(f"Bash script '{output_filepath}' created successfully.")


def create_vina_bash_script(
    binding_site_method: Literal[
        "diffdock", "fabind", "dynamicbind", "neuralplexer", "flowdock", "rfaa"
    ],
    protein_filepath: str,
    ligand_filepath: str,
    apo_protein_filepath: str,
    input_id: str,
    output_filepath: str,
    cfg: DictConfig,
    generate_hpc_scripts: bool = True,
):
    """Create a bash script to run Vina-based protein-ligand complex
    prediction.

    :param binding_site_method: Name of the method used to predict the
        binding site.
    :param protein_filepath: Path to the input protein structure PDB
        file.
    :param ligand_filepath: Path to the input ligand structure SDF file.
    :param apo_protein_filepath: Path to the predicted apo protein
        structure PDB file.
    :param input_id: Input ID.
    :param output_filepath: Path to the output bash script file.
    :param cfg: Configuration dictionary for runtime arguments.
    :param generate_hpc_scripts: Whether to generate HPC scripts for
        Vina.
    """
    bash_script_content = f"""#!/bin/bash -l
{insert_hpc_headers(method='vina') if generate_hpc_scripts else 'source /home/$USER/mambaforge/etc/profile.d/conda.sh'}
conda activate {"$project_dir/PoseBench/" if generate_hpc_scripts else 'PoseBench'}

# command to run vina_inference.py
echo "Calling vina_inference.py!"
python3 posebench/models/vina_inference.py \\
    method={binding_site_method} \\
    python2_exec_path={cfg.vina_python2_exec_path} \\
    prepare_receptor_script_path={cfg.vina_prepare_receptor_script_path} \\
    output_dir={cfg.vina_output_dir} \\
    cpu={cfg.vina_cpu} \\
    seed={cfg.vina_seed} \\
    exhaustiveness={cfg.vina_exhaustiveness} \\
    ligand_ligand_distance_threshold={cfg.vina_ligand_ligand_distance_threshold} \\
    protein_ligand_distance_threshold={cfg.vina_protein_ligand_distance_threshold} \\
    binding_site_size_x={cfg.vina_binding_site_size_x} \\
    binding_site_size_y={cfg.vina_binding_site_size_y} \\
    binding_site_size_z={cfg.vina_binding_site_size_z} \\
    binding_site_spacing={cfg.vina_binding_site_spacing} \\
    num_modes={cfg.vina_num_modes} \\
    skip_existing={cfg.vina_skip_existing} \\
    protein_filepath="{protein_filepath}" \\
    ligand_filepath="{ligand_filepath}" \\
    apo_protein_filepath="{apo_protein_filepath}" \\
    input_id="{input_id}" \\
    p2rank_exec_utility={cfg.vina_p2rank_exec_utility} \\
    p2rank_config={cfg.vina_p2rank_config} \\
    p2rank_enable_pymol_visualizations={cfg.vina_p2rank_enable_pymol_visualizations} \\

echo "Finished calling vina_inference.py!"
    """

    with open(output_filepath, "w") as file:
        file.write(bash_script_content)

    logger.info(f"Bash script '{output_filepath}' created successfully.")


def generate_method_prediction_script(
    method: str,
    protein_filepath: str,
    ligand_smiles: str,
    input_id: str,
    output_filepath: str,
    cfg: DictConfig,
    generate_hpc_scripts: bool,
    method_filepaths_mapping: Optional[Dict[str, List[Tuple[str, str]]]] = None,
):
    """Generate a script to run the method's protein-ligand complex prediction.

    :param method: Name of the method to generate a prediction script
        for.
    :param protein_filepath: Path to the input protein structure PDB
        file.
    :param ligand_smiles: SMILES string of the input ligand.
    :param input_id: Input ID.
    :param output_filepath: Path to the output Bash script file.
    :param cfg: Configuration dictionary for runtime arguments.
    :param generate_hpc_scripts: Whether to generate HPC scripts for the
        method.
    :param method_filepaths_mapping: Optional mapping of method names to
        a list of tuples of protein and ligand filepaths.
    """

    def extract_protein_chains_to_fasta_files(protein_filepath: str) -> List[str]:
        """Extract individual chains from a protein file and save them as
        separate FASTA files.

        :param protein_filepath: Path to the protein file.
        :return: List of paths to the extracted FASTA files.
        """
        # Parse the protein file using Bio.PDB
        parser = PDBParser()
        structure = parser.get_structure("structure", protein_filepath)
        sequences = extract_sequences_from_protein_structure_file(
            protein_filepath, structure=structure
        )

        # Create a temporary directory to store the FASTA files
        temp_dir = tempfile.mkdtemp()

        # Iterate over each chain and save it as a separate FASTA file
        fasta_filepaths = []
        for model in structure:
            model_chains = [chain for chain in model]
            assert len(model_chains) == len(
                sequences
            ), "For RFAA, numbers of Biopython chains and parsed sequences do not match."
            for chain_index, chain in enumerate(model_chains):
                fasta_filename = f"{Path(protein_filepath).stem}_{chain.id}.fasta"
                fasta_filepath = os.path.join(temp_dir, fasta_filename)
                chain_sequence = sequences[chain_index]
                with open(fasta_filepath, "w") as f:
                    f.write(f">{chain.id}\n{chain_sequence}\n")
                fasta_filepaths.append(fasta_filepath)

        return fasta_filepaths

    if method == "diffdock":
        create_diffdock_bash_script(
            protein_filepath,
            ligand_smiles.replace(
                ":", "."
            ),  # NOTE: DiffDock supports multi-ligands using the separator "."
            input_id,
            output_filepath,
            cfg,
            generate_hpc_scripts=generate_hpc_scripts,
        )
    elif method == "dynamicbind":
        create_dynamicbind_bash_script(
            protein_filepath,
            ligand_smiles.replace(
                ":", "."
            ),  # NOTE: DynamicBind supports multi-ligands using the separator "."
            output_filepath,
            cfg,
            generate_hpc_scripts=generate_hpc_scripts,
        )
    elif method == "neuralplexer":
        create_neuralplexer_bash_script(
            protein_filepath,
            ligand_smiles.replace(
                ":", "."
            ),  # NOTE: NeuralPLexer supports multi-ligands using the separator "."
            input_id,
            output_filepath,
            cfg,
            generate_hpc_scripts=generate_hpc_scripts,
        )
    elif method == "flowdock":
        create_flowdock_bash_script(
            protein_filepath,
            ligand_smiles.replace(
                ":", "."
            ),  # NOTE: FlowDock supports multi-ligands using the separator "."
            input_id,
            output_filepath,
            cfg,
            generate_hpc_scripts=generate_hpc_scripts,
        )
    elif method == "rfaa":
        fasta_filepaths = extract_protein_chains_to_fasta_files(protein_filepath)
        smiles_strings = ligand_smiles.split(":")
        create_rfaa_bash_script(
            fasta_filepaths=fasta_filepaths,
            sdf_filepaths=None,
            input_id=input_id,
            cfg=cfg,
            output_filepath=output_filepath,
            smiles_strings=smiles_strings,
            generate_hpc_scripts=generate_hpc_scripts,
        )
    elif method == "chai-lab":
        create_chai_bash_script(
            protein_filepath=protein_filepath,
            ligand_smiles=ligand_smiles,
            input_id=input_id,
            cfg=cfg,
            output_filepath=output_filepath,
            generate_hpc_scripts=generate_hpc_scripts,
        )
    elif method == "boltz":
        create_boltz_bash_script(
            protein_filepath=protein_filepath,
            ligand_smiles=ligand_smiles,
            input_id=input_id,
            cfg=cfg,
            output_filepath=output_filepath,
            generate_hpc_scripts=generate_hpc_scripts,
        )
    elif method == "alphafold3":
        logger.info(
            "AlphaFold-3 ensemble prediction Bash scripts are not supported. Skipping script creation."
        )
    elif method == "vina":
        assert (
            cfg.generate_vina_scripts and cfg.resume
        ), "Vina predictions must be resumed from prior method predictions."
        for binding_site_method in cfg.vina_binding_site_methods:
            # NOTE: these correspond to selecting the top-ranked method predictions (since the method predictions are already rank-ordered)
            method_protein_filepath = method_filepaths_mapping[binding_site_method][0][0]
            method_ligand_filepath = method_filepaths_mapping[binding_site_method][0][1]
            vina_output_filepath = os.path.join(
                cfg.output_bash_file_dir,
                os.path.basename(output_filepath).replace("vina_", f"vina_{binding_site_method}_"),
            )
            create_vina_bash_script(
                binding_site_method,
                method_protein_filepath,
                method_ligand_filepath,
                protein_filepath,
                input_id,
                vina_output_filepath,
                cfg,
                generate_hpc_scripts=generate_hpc_scripts,
            )
    else:
        raise ValueError(f"Method {method} is not supported.")


def rank_key(file_path: str) -> float:
    """Define a custom key for ranking the predictions.

    :param file_path: Path to the file to rank.
    :return: The rank key for the file.
    """
    filename = os.path.basename(file_path)
    match = re.search(r"rank(\d+)", filename)
    if match:
        return int(match.group(1))
    else:
        return float("inf")  # NOTE: if no match found, put it at the end


def get_method_predictions(
    method: str,
    target: str,
    cfg: DictConfig,
    binding_site_method: Optional[str] = None,
    input_protein_filepath: Optional[str] = None,
    is_ss_method: bool = False,
) -> List[Tuple[str, str]]:
    """Get the predictions generated by the method.

    :param method: Name of the method to get predictions for.
    :param target: Name of the target protein-ligand pair.
    :param cfg: Configuration dictionary for runtime arguments.
    :param binding_site_method: Optional name of the method used to
        predict AutoDock Vina's binding sites.
    :param input_protein_filepath: Optional path to the input protein
        structure PDB file.
    :param is_ss_method: Whether the method is a single-sequence method.
    :return: List of method predictions, each as a tuple of the output
        protein filepath and the output ligand filepath.
    """
    pocket_only_suffix = "_pocket_only" if cfg.pocket_only_baseline else ""
    no_ilcl_suffix = "_no_ilcl" if cfg.neuralplexer_no_ilcl else ""
    single_seq_suffix = "_ss" if is_ss_method else ""

    if method == "diffdock":
        ensemble_benchmarking_output_dir = (
            Path(cfg.diffdock_output_dir).parent
            / f"diffdock{pocket_only_suffix}_{cfg.ensemble_benchmarking_dataset}_output_{cfg.ensemble_benchmarking_repeat_index}"
            if cfg.ensemble_benchmarking
            else cfg.diffdock_output_dir
        )
        if cfg.ensemble_benchmarking:
            protein_output_files = list(
                Path(cfg.ensemble_benchmarking_apo_protein_dir).rglob(f"*{target}*.pdb")
            )
            if len(protein_output_files) == 0:
                logger.warning(
                    f"No apo protein structure found for target {target} in directory {cfg.ensemble_benchmarking_apo_protein_dir}. Skipping this target..."
                )
                return []
            assert (
                len(protein_output_files) == 1
            ), "The DiffDock ensemble benchmarking dataset must contain one apo protein input structure per target."
            protein_output_files = [
                str(protein_output_files[0]) for _ in range(cfg.method_top_n_to_select)
            ]
        else:
            diffdock_input_csv = pd.read_csv(cfg.diffdock_input_csv_path)
            protein_output_files = [
                diffdock_input_csv[diffdock_input_csv.complex_name == target].protein_path.item()
                for _ in range(cfg.method_top_n_to_select)
            ]
        diffdock_target = (
            "_".join(target.split("_")[:3])
            if cfg.ensemble_benchmarking and cfg.ensemble_benchmarking_dataset == "dockgen"
            else target
        )
        ligand_output_files = sorted(
            [
                file
                for file in map(
                    str,
                    Path(os.path.join(ensemble_benchmarking_output_dir, diffdock_target)).rglob(
                        "*.sdf"
                    ),
                )
                if "rank" in os.path.basename(file)
                and "confidence" in os.path.basename(file)
                and "relaxed" not in os.path.basename(file)
            ],
            key=rank_key,
        )[: cfg.method_top_n_to_select]
        for protein_output_file, ligand_output_file in zip(
            protein_output_files, ligand_output_files
        ):
            assert (
                os.path.splitext(os.path.basename(protein_output_file))[0].split("_holo")[0]
                == target
            ), "Protein files must be for the designated target."
            assert (
                Path(ligand_output_file).parent.stem == diffdock_target
            ), "Ligand files must be for the designated target."
    elif method == "dynamicbind":
        target_dir_name = (
            f"{cfg.ensemble_benchmarking_dataset}{pocket_only_suffix}_{target}_{cfg.ensemble_benchmarking_repeat_index}"
            if cfg.ensemble_benchmarking
            else f"{cfg.dynamicbind_header}{pocket_only_suffix}_{target}"
        )
        protein_output_files = list(
            map(
                str,
                Path(cfg.dynamicbind_input_ligand_csv_dir).parent.rglob(
                    os.path.join(
                        "outputs",
                        "results",
                        target_dir_name,
                        "index0_idx_0",
                        "rank*_receptor_*.pdb",
                    )
                ),
            )
        )
        ligand_output_files = list(
            map(
                str,
                Path(cfg.dynamicbind_input_ligand_csv_dir).parent.rglob(
                    os.path.join(
                        "outputs",
                        "results",
                        target_dir_name,
                        "index0_idx_0",
                        "rank*_ligand_*.sdf",
                    )
                ),
            )
        )
        protein_output_files = sorted(
            [file for file in protein_output_files if "relaxed" not in os.path.basename(file)],
            key=rank_key,
        )[: cfg.method_top_n_to_select]
        ligand_output_files = sorted(
            [file for file in ligand_output_files if "relaxed" not in os.path.basename(file)],
            key=rank_key,
        )[: cfg.method_top_n_to_select]
        if len(protein_output_files) < len(ligand_output_files):
            ligand_output_files = [
                file
                for file in ligand_output_files
                if Path(file).stem.replace("_ligand_", "_receptor_")
                in {Path(protein_file).stem for protein_file in protein_output_files}
            ]
        if len(ligand_output_files) < len(protein_output_files):
            protein_output_files = [
                file
                for file in protein_output_files
                if Path(file).stem.replace("_receptor_", "_ligand_")
                in {Path(ligand_file).stem for ligand_file in ligand_output_files}
            ]
        if len(protein_output_files) < len(ligand_output_files):
            ligand_output_files = [
                file
                for file in ligand_output_files
                if Path(file).stem.replace("_ligand_", "_receptor_")
                in {Path(protein_file).stem for protein_file in protein_output_files}
            ]
        assert len(protein_output_files) == len(
            ligand_output_files
        ), "The number of DynamicBind protein and ligand files must match."
    elif method == "neuralplexer":
        ensemble_benchmarking_output_dir = (
            Path(cfg.input_dir if cfg.input_dir else cfg.neuralplexer_out_path).parent
            / f"neuralplexer{single_seq_suffix}{pocket_only_suffix}{no_ilcl_suffix}_{cfg.ensemble_benchmarking_dataset}_outputs_{cfg.ensemble_benchmarking_repeat_index}"
            if cfg.ensemble_benchmarking
            else (cfg.input_dir if cfg.input_dir else cfg.neuralplexer_out_path)
        )
        protein_output_files = sorted(
            [
                file
                for file in map(
                    str,
                    Path(os.path.join(ensemble_benchmarking_output_dir, target)).rglob("*.pdb"),
                )
                if "rank" in os.path.basename(file)
                and "relaxed" not in os.path.basename(file)
                and "aligned" not in os.path.basename(file)
            ],
            key=rank_key,
        )[: cfg.method_top_n_to_select]
        ligand_output_files = sorted(
            [
                file
                for file in map(
                    str,
                    Path(os.path.join(ensemble_benchmarking_output_dir, target)).rglob("*.sdf"),
                )
                if "rank" in os.path.basename(file)
                and "relaxed" not in os.path.basename(file)
                and "aligned" not in os.path.basename(file)
            ],
            key=rank_key,
        )[: cfg.method_top_n_to_select]
    elif method == "flowdock":
        ensemble_benchmarking_output_dir = (
            Path(cfg.input_dir if cfg.input_dir else cfg.flowdock_out_path).parent
            / f"flowdock_{cfg.ensemble_benchmarking_dataset}_outputs_{cfg.ensemble_benchmarking_repeat_index}"
            if cfg.ensemble_benchmarking
            else (cfg.input_dir if cfg.input_dir else cfg.flowdock_out_path)
        )
        protein_output_files = sorted(
            [
                file
                for file in map(
                    str,
                    Path(os.path.join(ensemble_benchmarking_output_dir, target)).rglob("*.pdb"),
                )
                if "rank" in os.path.basename(file)
                and "relaxed" not in os.path.basename(file)
                and "aligned" not in os.path.basename(file)
            ],
            key=rank_key,
        )[: cfg.method_top_n_to_select]
        ligand_output_files = sorted(
            [
                file
                for file in map(
                    str,
                    Path(os.path.join(ensemble_benchmarking_output_dir, target)).rglob("*.sdf"),
                )
                if "rank" in os.path.basename(file)
                and "relaxed" not in os.path.basename(file)
                and "aligned" not in os.path.basename(file)
            ],
            key=rank_key,
        )[: cfg.method_top_n_to_select]
    elif method == "rfaa":
        ensemble_benchmarking_output_dir = (
            Path(cfg.rfaa_output_dir).parent
            / f"rfaa{pocket_only_suffix}_{cfg.ensemble_benchmarking_dataset}_outputs_{cfg.ensemble_benchmarking_repeat_index}"
            if cfg.ensemble_benchmarking
            else cfg.rfaa_output_dir
        )
        protein_output_files = sorted(
            [
                file
                for file in map(
                    str,
                    Path(os.path.join(ensemble_benchmarking_output_dir, target)).rglob(
                        f"{target}_protein.pdb"
                    ),
                )
            ],
            key=rank_key,
        )[: cfg.method_top_n_to_select]
        ligand_output_files = sorted(
            [
                file
                for file in map(
                    str,
                    Path(os.path.join(ensemble_benchmarking_output_dir, target)).rglob(
                        f"{target}_ligand.sdf"
                    ),
                )
            ],
            key=rank_key,
        )[: cfg.method_top_n_to_select]
    elif method == "chai-lab":
        ensemble_benchmarking_output_dir = (
            Path(cfg.input_dir if cfg.input_dir else cfg.chai_out_path).parent
            / f"chai-lab{single_seq_suffix}{pocket_only_suffix}_{cfg.ensemble_benchmarking_dataset}_outputs_{cfg.ensemble_benchmarking_repeat_index}"
            if cfg.ensemble_benchmarking
            else (cfg.input_dir if cfg.input_dir else cfg.chai_out_path)
        )
        protein_output_files = sorted(
            [
                file
                for file in map(
                    str,
                    Path(os.path.join(ensemble_benchmarking_output_dir, target)).rglob("*.pdb"),
                )
                if "model_idx" in os.path.basename(file)
                and "relaxed" not in os.path.basename(file)
                and "aligned" not in os.path.basename(file)
            ],
            key=rank_key,
        )[: cfg.method_top_n_to_select]
        ligand_output_files = sorted(
            [
                file
                for file in map(
                    str,
                    Path(os.path.join(ensemble_benchmarking_output_dir, target)).rglob("*.sdf"),
                )
                if "model_idx" in os.path.basename(file)
                and "relaxed" not in os.path.basename(file)
                and "aligned" not in os.path.basename(file)
                and "_LIG_" not in os.path.basename(file)
            ],
            key=rank_key,
        )[: cfg.method_top_n_to_select]
    elif method == "boltz":
        ensemble_benchmarking_output_dir = (
            Path(cfg.input_dir if cfg.input_dir else cfg.boltz_out_path).parent
            / f"boltz{single_seq_suffix}{pocket_only_suffix}_{cfg.ensemble_benchmarking_dataset}_outputs_{cfg.ensemble_benchmarking_repeat_index}"
            if cfg.ensemble_benchmarking
            else (cfg.input_dir if cfg.input_dir else cfg.boltz_out_path)
        )
        protein_output_files = sorted(
            [
                file
                for file in map(
                    str,
                    Path(os.path.join(ensemble_benchmarking_output_dir, target)).rglob("*.pdb"),
                )
                if "model_" in os.path.basename(file)
                and "relaxed" not in os.path.basename(file)
                and "aligned" not in os.path.basename(file)
            ],
            key=rank_key,
        )[: cfg.method_top_n_to_select]
        ligand_output_files = sorted(
            [
                file
                for file in map(
                    str,
                    Path(os.path.join(ensemble_benchmarking_output_dir, target)).rglob("*.sdf"),
                )
                if "model_" in os.path.basename(file)
                and "relaxed" not in os.path.basename(file)
                and "aligned" not in os.path.basename(file)
                and "_LIG" not in os.path.basename(file)
            ],
            key=rank_key,
        )[: cfg.method_top_n_to_select]
    elif method == "alphafold3":
        ensemble_benchmarking_output_dir = (
            Path(cfg.input_dir if cfg.input_dir else cfg.alphafold3_out_path).parent
            / f"alphafold3{single_seq_suffix}{pocket_only_suffix}_{cfg.ensemble_benchmarking_dataset}_outputs_{cfg.ensemble_benchmarking_repeat_index}"
            if cfg.ensemble_benchmarking
            else (cfg.input_dir if cfg.input_dir else cfg.alphafold3_out_path)
        )
        protein_output_files = sorted(
            [
                file
                for file in map(
                    str,
                    Path(os.path.join(ensemble_benchmarking_output_dir, target)).rglob("*.pdb"),
                )
                if "model_protein" in os.path.basename(file)
                and "relaxed" not in os.path.basename(file)
                and "aligned" not in os.path.basename(file)
            ],
            key=rank_key,
        )[: cfg.method_top_n_to_select]
        ligand_output_files = sorted(
            [
                file
                for file in map(
                    str,
                    Path(os.path.join(ensemble_benchmarking_output_dir, target)).rglob("*.sdf"),
                )
                if "model_ligand" in os.path.basename(file)
                and "relaxed" not in os.path.basename(file)
                and "aligned" not in os.path.basename(file)
                and "_LIG_" not in os.path.basename(file)
            ],
            key=rank_key,
        )[: cfg.method_top_n_to_select]
    elif method == "vina":
        assert binding_site_method, "Binding site method must be provided for Vina predictions."
        ensemble_benchmarking_output_dir = (
            Path(cfg.vina_output_dir).parent
            / f"vina{pocket_only_suffix}_{binding_site_method}_{cfg.ensemble_benchmarking_dataset}_outputs_{cfg.ensemble_benchmarking_repeat_index}"
            if cfg.ensemble_benchmarking
            else cfg.vina_output_dir.replace("vina_", f"vina_{binding_site_method}_")
        )
        if cfg.ensemble_benchmarking:
            protein_output_files = list(
                Path(cfg.ensemble_benchmarking_apo_protein_dir).rglob(f"*{target}*.pdb")
            )
            if len(protein_output_files) == 0:
                logger.warning(
                    f"No apo protein structure found for target {target} in directory {cfg.ensemble_benchmarking_apo_protein_dir}. Skipping this target..."
                )
                return []
            assert (
                len(protein_output_files) == 1
            ), "The DiffDock ensemble benchmarking dataset must contain one apo protein input structure per target."
            protein_output_files = [str(protein_output_files[0])]
        else:
            assert (
                input_protein_filepath
            ), "Input protein filepath must be provided for Vina predictions when not ensemble-benchmarking."
            protein_output_files = [input_protein_filepath]
        # NOTE: Vina saves only the top-ranked ligand prediction per target
        vina_target = (
            "_".join(target.split("_")[:3])
            if cfg.ensemble_benchmarking and cfg.ensemble_benchmarking_dataset == "dockgen"
            else target
        )
        ligand_output_files = sorted(
            [
                file
                for file in map(
                    str,
                    Path(os.path.join(ensemble_benchmarking_output_dir, vina_target)).rglob(
                        f"{vina_target}*.sdf"
                    ),
                )
                if "relaxed" not in os.path.basename(file)
                and "group" not in os.path.basename(file)
            ],
            key=rank_key,
        )[: cfg.method_top_n_to_select]
        if not len(ligand_output_files):
            # NOTE: when predicting e.g., with DiffDock-Vina, if no ligand predictions are found, skip the protein prediction as well
            protein_output_files = []
        assert len(protein_output_files) == len(
            ligand_output_files
        ), "Number of Vina protein and ligand files must match."
    elif method == "tulip":
        assert input_protein_filepath is not None and os.path.exists(
            input_protein_filepath
        ), "A valid input protein filepath must be provided to analyze TULIP's results."
        ensemble_benchmarking_output_dir = (
            Path(cfg.tulip_output_dir).parent
            / f"tulip_{cfg.ensemble_benchmarking_dataset}_outputs_{cfg.ensemble_benchmarking_repeat_index}"
            if cfg.ensemble_benchmarking
            else cfg.tulip_output_dir
        )
        ligand_output_files = sorted(
            [
                file
                for file in map(
                    str,
                    Path(os.path.join(ensemble_benchmarking_output_dir, target)).rglob(
                        "rank*.sdf"
                    ),
                )
                if "relaxed" not in os.path.basename(file)
            ],
            key=rank_key,
        )[: cfg.method_top_n_to_select]
        protein_output_files = [input_protein_filepath for _ in range(len(ligand_output_files))]
    else:
        raise ValueError(f"Method {method} is not supported.")

    protein_output_files = sorted(protein_output_files, key=rank_key)
    ligand_output_files = sorted(ligand_output_files, key=rank_key)

    if cfg.relax_method_ligands_pre_ranking:
        # relax ligand structures prior to ranking as desired;
        # NOTE: relaxation is practically necessary when making
        # multi-ligand predictions with DiffDock and DynamicBind,
        # and is optional (yet highly recommended) for all other predictions
        pool = multiprocessing.Pool(processes=cfg.relax_num_processes)
        for protein_output_file, ligand_output_file in zip(
            protein_output_files, ligand_output_files
        ):
            if (
                not os.path.exists(ligand_output_file.replace(".sdf", "_ensemble_relaxed.sdf"))
                or not cfg.skip_existing
            ):
                with tempfile.TemporaryDirectory(
                    suffix=f"_{method}_ensemble_cache_dir"
                ) as temp_directory:
                    pool.apply_async(
                        relax_single_filepair,
                        args=(
                            Path(protein_output_file),
                            Path(ligand_output_file),
                            Path(ligand_output_file).parent,
                            temp_directory,
                            OmegaConf.create(
                                {
                                    "method": "ensemble",
                                    "add_solvent": cfg.relax_add_solvent,
                                    "name": target,
                                    "prep_only": cfg.relax_prep_only,
                                    "platform": cfg.relax_platform,
                                    "cuda_device_index": cfg.cuda_device_index,
                                    "log_level": cfg.relax_log_level,
                                    "relax_protein": cfg.relax_protein,
                                    "remove_initial_protein_hydrogens": cfg.relax_remove_initial_protein_hydrogens
                                    or method == "rfaa",
                                    "assign_each_ligand_unique_force": cfg.relax_assign_each_ligand_unique_force,
                                    "model_ions": cfg.relax_model_ions,
                                    "cache_files": cfg.relax_cache_files,
                                    "assign_partial_charges_manually": cfg.relax_assign_partial_charges_manually,
                                    "report_initial_energy_only": False,
                                    "max_final_e_value": cfg.relax_max_final_e_value,
                                    "max_num_attempts": cfg.relax_max_num_attempts,
                                    "skip_existing": cfg.relax_skip_existing,
                                }
                            ),
                        ),
                    )
        pool.close()
        pool.join()
        relaxed_ligand_output_files = [
            file.replace(".sdf", "_ensemble_relaxed.sdf") for file in ligand_output_files
        ]
        assert all(
            os.path.exists(file) for file in relaxed_ligand_output_files
        ), f"Ligand relaxation failed for file(s): {[file for file in relaxed_ligand_output_files if not os.path.exists(file)]}"
        if cfg.relax_protein:
            relaxed_protein_output_files = [
                file.replace(".pdb", "_protein_ensemble_relaxed.pdb")
                for file in protein_output_files
            ]
            assert all(
                os.path.exists(file) for file in relaxed_protein_output_files
            ), f"Protein relaxation failed for file(s): {[file for file in relaxed_protein_output_files if not os.path.exists(file)]}"
            protein_output_files = relaxed_protein_output_files
        ligand_output_files = relaxed_ligand_output_files
    return list(zip(protein_output_files, ligand_output_files))


def generate_ensemble_predictions(
    protein_filepath: str,
    ligand_smiles: str,
    input_id: str,
    cfg: DictConfig,
    generate_hpc_scripts: bool = True,
    method_filepaths_mapping: Optional[Dict[str, List[Tuple[str, str]]]] = None,
) -> Tuple[Optional[ENSEMBLE_PREDICTIONS], bool]:
    """Generate bound complex predictions using an ensemble of methods.

    :param protein_filepath: Path to the input protein structure PDB
        file.
    :param ligand_input: Path to the input ligand SMILES string.
    :param input_id: Input ID.
    :param target: Name of the target protein-ligand pair.
    :param cfg: Configuration dictionary for runtime arguments.
    :param generate_hpc_scripts: Whether to generate HPC scripts for the
        ensemble predictions.
    :param method_filepaths_mapping: Optional mapping of method names to
        a list of tuples of protein and ligand filepaths.
    :return: Dictionary of method names and their corresponding
        predictions as well as whether the prediction scripts were
        generated and now need to be run.
    """
    os.makedirs(cfg.output_bash_file_dir, exist_ok=True)

    generating_script = False
    for method in cfg.ensemble_methods:
        output_filepath = os.path.join(
            cfg.output_bash_file_dir, f"{method}_{input_id}_inference.sh"
        )
        generating_script = not cfg.resume or (
            method == "vina" and cfg.generate_vina_scripts and cfg.resume
        )
        if generating_script:
            generate_method_prediction_script(
                method,
                protein_filepath,
                ligand_smiles,
                input_id,
                output_filepath,
                cfg,
                generate_hpc_scripts,
                method_filepaths_mapping=method_filepaths_mapping,
            )
    if generating_script:
        return None, generating_script

    if cfg.resume:
        ensemble_predictions_dict = {}
        for method, is_ss_method in zip(cfg.ensemble_methods, cfg.is_ss_ensemble_method):
            if method == "vina":
                for binding_site_method in cfg.vina_binding_site_methods:
                    method_predictions = get_method_predictions(
                        method,
                        input_id,
                        cfg,
                        binding_site_method=binding_site_method,
                        input_protein_filepath=protein_filepath,
                        is_ss_method=is_ss_method,
                    )
                    ensemble_predictions_dict[f"vina_{binding_site_method}"] = method_predictions
            else:
                method_predictions = get_method_predictions(
                    method,
                    input_id,
                    cfg,
                    input_protein_filepath=protein_filepath,
                    is_ss_method=is_ss_method,
                )
                ensemble_predictions_dict[method] = method_predictions

    return ensemble_predictions_dict, generating_script


def consensus_rank_ensemble_predictions(
    cfg: DictConfig,
    method_ligand_positions: List[np.ndarray],
    ensemble_predictions_list: List[Tuple[str, str, str]],
) -> RANKED_ENSEMBLE_PREDICTIONS:
    """Consensus-rank the predictions to select the top prediction(s).

    :param cfg: Configuration dictionary for runtime arguments.
    :param method_ligand_positions: List of ligand positions from each
        method's predictions.
    :param ensemble_predictions_list: List of tuples of method name,
        output protein filepath, and output ligand filepath.
    :return: Dictionary of consensus-ranked predictions indexed by each
        prediction's consensus ranking and valued as its method name,
        output protein filepath, output ligand filepath, and average
        pairwise RMSD.
    """
    if len(cfg.ensemble_methods) > 1 or (
        len(cfg.ensemble_methods) == 1 and not cfg.rank_single_method_intrinsically
    ):
        # calculate RMSD values between each pair of method predictions
        rmsd_values_list = []
        for positions1 in method_ligand_positions:
            rmsd_values = []
            for positions2 in method_ligand_positions:
                rmsd = calculate_rmsd(positions1, positions2)
                if rmsd > 0.0:  # avoid self-comparison
                    rmsd_values.append(rmsd)
            rmsd_values_list.append(rmsd_values)
        avg_rmsd_values_array = np.array(rmsd_values_list).mean(-1)

        # rank the predictions by their average RMSD values
        sorted_indices = np.argsort(avg_rmsd_values_array)
        ranked_predictions = {
            rank + 1: (*ensemble_predictions_list[index], avg_rmsd_values_array[index])
            for rank, index in enumerate(sorted_indices)
        }
    else:
        # rank the predictions by their explicit rank assignments
        ranked_predictions = {
            rank + 1: (*ensemble_predictions_list[index], 0.0)
            for rank, index in enumerate(range(len(ensemble_predictions_list)))
        }

    return ranked_predictions


def ff_rank_ensemble_predictions(
    cfg: DictConfig,
    ensemble_predictions_list: List[Tuple[str, str, str]],
) -> RANKED_ENSEMBLE_PREDICTIONS:
    """Rank the predictions using an OpenMM force field (FF) to select the top
    prediction(s) according to the criterion of minimum energy.

    :param cfg: Configuration dictionary for runtime arguments.
    :param ensemble_predictions_list: List of tuples of method name,
        output protein filepath, and output ligand filepath.
    :return: Dictionary of Vina-ranked predictions indexed by each
        prediction's consensus ranking and valued as its method name,
        output protein filepath, output ligand filepath, and Vina energy
        score.
    """
    if len(cfg.ensemble_methods) > 1 or (
        len(cfg.ensemble_methods) == 1 and not cfg.rank_single_method_intrinsically
    ):
        # calculate Vina energy scores of the (intrinsically) top-ranked method predictions
        vina_energy_scores_list = []
        new_ensemble_predictions_list = []
        for prediction in ensemble_predictions_list:
            try:
                method, protein_filepath, ligand_filepath = prediction
                name = "_".join(Path(protein_filepath).stem.split("_")[:2])
                with tempfile.TemporaryDirectory(
                    suffix=f"_{method}_ensemble_cache_dir"
                ) as temp_directory:
                    result = minimize_energy(
                        OmegaConf.create(
                            {
                                "protein_file": protein_filepath,
                                "ligand_file": ligand_filepath,
                                "output_file": str(
                                    Path(
                                        Path(ligand_filepath).parent,
                                        f"{Path(ligand_filepath).stem}_ensemble_relaxed.sdf",
                                    )
                                ),
                                "protein_output_file": str(
                                    Path(
                                        Path(ligand_filepath).parent,
                                        f"{Path(protein_filepath).stem}_protein_ensemble_relaxed.pdb",
                                    )
                                ),
                                "complex_output_file": None,
                                "temp_dir": temp_directory,
                                "add_solvent": cfg.relax_add_solvent,
                                "name": name,
                                "prep_only": cfg.relax_prep_only,
                                "platform": cfg.relax_platform,
                                "cuda_device_index": cfg.cuda_device_index,
                                "log_level": cfg.relax_log_level,
                                "relax_protein": cfg.relax_protein,
                                "remove_initial_protein_hydrogens": cfg.relax_remove_initial_protein_hydrogens
                                or method == "rfaa",
                                "assign_each_ligand_unique_force": cfg.relax_assign_each_ligand_unique_force,
                                "model_ions": cfg.relax_model_ions,
                                "cache_files": cfg.relax_cache_files,
                                "assign_partial_charges_manually": cfg.relax_assign_partial_charges_manually,
                                "report_initial_energy_only": True,
                                "max_final_e_value": cfg.relax_max_final_e_value,
                                "max_num_attempts": cfg.relax_max_num_attempts,
                            }
                        )
                    )
                    new_ensemble_predictions_list.append(prediction)
                    vina_energy_scores_list.append(result["e_init"]._value)

            except Exception as e:
                logger.warning(
                    f"Failed to calculate Vina energy score for prediction {prediction}. Skipping due to: {e}"
                )
                continue
        ensemble_predictions_list = new_ensemble_predictions_list
        vina_energy_scores_array = np.array(vina_energy_scores_list)

        # rank the predictions by their Vina energy scores (NOTE: lower is better)
        sorted_indices = np.argsort(vina_energy_scores_array)
        ranked_predictions = {
            rank + 1: (*ensemble_predictions_list[index], vina_energy_scores_array[index])
            for rank, index in enumerate(sorted_indices)
        }
    else:
        # rank the predictions by their intrinsic (explicit) rank assignments
        ranked_predictions = {
            rank + 1: (*ensemble_predictions_list[index], 0.0)
            for rank, index in enumerate(range(len(ensemble_predictions_list)))
        }

    return ranked_predictions


def rank_ensemble_predictions(
    ensemble_predictions_dict: ENSEMBLE_PREDICTIONS,
    name: str,
    cfg: DictConfig,
) -> RANKED_ENSEMBLE_PREDICTIONS:
    """Rank the predictions to select the top prediction(s).

    :param ensemble_predictions_dict: Dictionary of method names and
        their corresponding predictions.
    :param name: Name of the target protein-ligand pair.
    :param cfg: Configuration dictionary for runtime arguments.
    :return: Dictionary of consensus-ranked predictions indexed by each
        prediction's consensus ranking and valued as its method name,
        output protein filepath, output ligand filepath, and average
        pairwise RMSD or Vina energy score.
    """
    # cache filepath to predicted apo protein structure from a structure predictor e.g., ESMFold
    if cfg.ensemble_benchmarking:
        apo_reference_protein_filepaths = list(
            Path(cfg.ensemble_benchmarking_apo_protein_dir).rglob(f"*{name}*.pdb")
        )
        if len(apo_reference_protein_filepaths) == 0:
            logger.warning(
                f"No apo protein structure found for target {name} in directory {cfg.ensemble_benchmarking_apo_protein_dir}. Skipping this target..."
            )
            return {}
        assert (
            len(apo_reference_protein_filepaths) == 1
        ), f"Expected one apo protein structure for target {name}."
        apo_reference_protein_filepath = apo_reference_protein_filepaths[0]
    elif ensemble_predictions_dict.get("diffdock"):
        predictions = ensemble_predictions_dict.get("diffdock")
        assert len(predictions) > 0, "No predictions found for DiffDock."
        assert (
            len(predictions[0]) == 2
        ), "Expected both a protein and ligand prediction filepath for DiffDock."
        apo_reference_protein_filepath = predictions[0][0]
    else:
        apo_reference_protein_filepath = None

    # determine reference ligand molecule for prediction sanity checking
    if cfg.ensemble_benchmarking:
        if ensemble_predictions_dict.get("neuralplexer"):
            reference_ligand_filepath = ensemble_predictions_dict["neuralplexer"][0][1]
        elif ensemble_predictions_dict.get("diffdock"):
            reference_ligand_filepath = ensemble_predictions_dict["diffdock"][0][1]
        else:
            reference_ligand_filepath = None
    elif ensemble_predictions_dict.get("neuralplexer"):
        reference_ligand_filepath = ensemble_predictions_dict["neuralplexer"][0][1]
    else:
        reference_ligand_filepath = None

    if reference_ligand_filepath is not None:
        assert os.path.exists(
            reference_ligand_filepath
        ), f"Reference ligand not found at {reference_ligand_filepath}."
        reference_ligand = read_molecule(reference_ligand_filepath)
        assert (
            reference_ligand is not None
        ), f"Failed to read reference ligand structure from {reference_ligand_filepath}."
    else:
        reference_ligand = None

    # collect ligand positions from each method's predictions
    unique_ligand_positions = {}
    method_ligand_positions, ensemble_predictions_list = [], []
    for method in ensemble_predictions_dict:
        for protein_filepath, ligand_filepath in ensemble_predictions_dict[method]:
            if (
                method in METHODS_PREDICTING_HOLO_PROTEIN_AB_INITIO
                and apo_reference_protein_filepath is not None
            ):
                try:
                    alignment_return_code = align_complex_to_protein_only(
                        protein_filepath, ligand_filepath, apo_reference_protein_filepath
                    )
                    if alignment_return_code == 0:
                        protein_filepath = protein_filepath.replace(".pdb", "_aligned.pdb")
                        ligand_filepath = ligand_filepath.replace(".sdf", "_aligned.sdf")
                    else:
                        logger.warning(
                            f"Failed to align predicted complex structure {protein_filepath} and ligand structure {ligand_filepath} to the apo protein structure {apo_reference_protein_filepath} from method {method}. Skipping alignment..."
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to align protein-ligand complex {protein_filepath} and {ligand_filepath} to apo protein structure {apo_reference_protein_filepath}. Skipping alignment due to: {e}"
                    )
            try:
                ligand = read_molecule(ligand_filepath)
            except Exception as e:
                logger.warning(
                    f"Failed to read predicted ligand positions of file {ligand_filepath} from method {method}. Skipping due to: {e}"
                )
                continue
            if ligand is None:
                logger.warning(
                    f"Failed to read predicted ligand positions of file {ligand_filepath} from method {method}. Skipping..."
                )
                continue
            # check whether predicted structure matches the expected number of atoms;
            # NOTE: occasionally, for the same SMILES input string, e.g., DynamicBind
            # may predict (multi-)ligand structures with a different number of atoms
            if (
                reference_ligand is not None
                and ligand.GetNumAtoms() != reference_ligand.GetNumAtoms()
            ):
                logger.warning(
                    f"Number of atoms in predicted ligand structure {ligand_filepath} from method {method} ({ligand.GetNumAtoms()}) does not match that of the reference ligand ({reference_ligand.GetNumAtoms()}). Skipping {method}'s prediction..."
                )
                continue
            # ensure only unique ligand position predictions are considered (e.g., as DiffDock may predict the same ligand structure multiple times due to numerical instability)
            ligand_positions = ligand.GetConformer().GetPositions()
            ligand_positions_tuple = tuple(map(tuple, ligand_positions))
            if ligand_positions_tuple not in unique_ligand_positions:
                unique_ligand_positions[ligand_positions_tuple] = len(unique_ligand_positions)
                method_ligand_positions.append(ligand_positions)
                ensemble_predictions_list.append((method, protein_filepath, ligand_filepath))
            else:
                logger.info(f"Skipping duplicate ligand position prediction from method {method}.")

    # rank the predictions using the requested ranking method
    if cfg.ensemble_ranking_method == "consensus":
        ranked_predictions = consensus_rank_ensemble_predictions(
            cfg,
            method_ligand_positions,
            ensemble_predictions_list,
        )
    elif cfg.ensemble_ranking_method == "ff":
        ranked_predictions = ff_rank_ensemble_predictions(
            cfg,
            ensemble_predictions_list,
        )

    return ranked_predictions


def assign_reference_residue_b_factors(
    protein_output_files: List[str], protein_reference_filepath: str
) -> List[str]:
    """If `b_factor` columns values are not already present, assign the
    reference protein structure's per-residue confidence scores to each output
    protein.

    :param protein_output_files: List of output protein structure PDB
        filepaths.
    :param protein_reference_filepath: Path to the input protein
        structure PDB file.
    :return: List of output protein structure PDB filepaths with the
        input protein's per-residue confidence scores.
    """
    new_protein_output_files = []

    def set_residue_b_factors(
        structure: Structure, residue_chain_b_factors: Dict[Tuple[int, str], float]
    ):
        """Set the per-residue confidence scores for each residue in the
        protein structure.

        :param structure: Structure object of the protein.
        :param: residue_chain_b_factors: Dictionary of residue keys to
            their confidence scores.
        """
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        residue_key = (residue.id[1], chain.id)
                        atom.set_bfactor(residue_chain_b_factors.get(residue_key, 0.0))

    def set_atom_b_factors(structure: Structure, ref_atom_records: List[Any]):
        """Set the per-atom confidence scores for each atom in the protein
        structure.

        :param structure: Structure object of the protein.
        :param ref_atom_records: List of Atom objects from the reference
            protein structure.
        """
        for atom_index, atom in enumerate(structure.get_atoms()):
            ref_atom = ref_atom_records[atom_index]
            atom.set_bfactor(ref_atom.get_bfactor())

    parser = PDBParser()

    for protein_output_file in protein_output_files:
        structure = parser.get_structure("protein", protein_output_file)
        assert structure, f"Failed to parse structure from {protein_output_file}."

        atom_records = [atom for atom in structure.get_atoms()]
        b_factors = [atom.get_bfactor() for atom in atom_records]

        if len(set(b_factors)) == 1:
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".pdb"
            ) as temp_output_protein_file:
                logger.warning(
                    f"All per-residue confidence scores in the input protein structure {protein_output_file} are identical. Replacing them with the confidence values stored in the predicted structure {protein_reference_filepath}."
                )
                ref_structure = parser.get_structure("input_protein", protein_reference_filepath)
                assert (
                    ref_structure
                ), f"Failed to parse structure from {protein_reference_filepath}."

                input_atom_records = [atom for atom in structure.get_atoms()]
                ref_atom_records = [atom for atom in ref_structure.get_atoms()]
                same_num_atoms = len(input_atom_records) == len(ref_atom_records)
                all_atom_records_match = same_num_atoms and all(
                    input_atom.name == ref_atom.name
                    for input_atom, ref_atom in zip(input_atom_records, ref_atom_records)
                )
                if all_atom_records_match:
                    # assign per-atom confidence scores
                    set_atom_b_factors(structure, ref_atom_records)
                else:
                    # assign per-residue confidence scores
                    residue_chain_b_factors = {}
                    for atom in ref_atom_records:
                        residue_key = (atom.get_parent().id[1], atom.get_parent().get_parent().id)
                        if residue_key not in residue_chain_b_factors:
                            residue_chain_b_factors[residue_key] = atom.get_bfactor()
                    set_residue_b_factors(structure, residue_chain_b_factors)

                io = PDBIO()
                io.set_structure(structure)
                io.save(temp_output_protein_file.name)
                new_protein_output_files.append(temp_output_protein_file.name)
        else:
            new_protein_output_files.append(protein_output_file)

    return new_protein_output_files


def export_proteins_in_casp_format(
    output_protein_filepaths: List[str],
    output_protein_pdb_file: str,
    pdb_header: str,
    append: bool = False,
    export_casp15_format: bool = False,
    model_index: Optional[int] = None,
    gap_insertion_point: Optional[int] = None,
):
    """Export the predicted protein structures in CASP format.

    :param output_protein_filepaths: List of output protein structure
        PDB filepaths.
    :param output_protein_pdb_file: Path to the output protein structure
        PDB file.
    :param pdb_header: Header string for the PDB file.
    :param append: Whether to append the predicted protein structures to
        the output file.
    :param export_casp15_format: Whether to format the output file for
        CASP15 benchmarking.
    :param model_index: Optional index of the model to write to the PDB
        file.
    :param gap_insertion_point: Optional `:`-separated string
        representing the chain-residue pair index of the residue at
        which to insert a single index gap.
    """
    with open(output_protein_pdb_file, "a" if append else "w") as f:
        if (
            export_casp15_format
            or (append and model_index is not None and model_index == 1)
            or not append
        ):
            f.write(pdb_header)
        for i, pdb_file in enumerate(output_protein_filepaths, start=1):
            structure = PDB.PDBParser().get_structure("pdb_structure", pdb_file)
            structure = renumber_biopython_structure_residues(
                structure, gap_insertion_point=gap_insertion_point
            )
            assert len(list(structure.get_models())) == 1, f"Expected one model in {pdb_file}."
            f.write(
                f"MODEL {model_index if model_index is not None else i}\nREMARK N/A\nPARENT N/A\n"
            )
            io = PDB.PDBIO()
            io.set_structure(structure)
            io.save(f, write_end=False)
            if not append:
                f.write("END\n")

    logger.info(f"CASP protein submission file saved to {output_protein_pdb_file}.")


def export_ligands_in_casp15_format(
    output_ligand_filepaths: List[str],
    output_ligand_sdf_file: str,
    sdf_header: str,
    method: str,
    append: bool = False,
    model_index: Optional[int] = None,
    ligand_numbers: Optional[Union[str, List[int]]] = None,
    ligand_names: Optional[Union[str, List[str]]] = None,
):
    """Export the predicted ligand structures in CASP15 format.

    Note that for the sake of consistency when evaluating deep learning docking methods
    on the CASP15 benchmark, we only report a single pose per protein-ligand model
    (i.e., 5 submitted protein-ligand models with 1 pose per model vs. 5 poses per model).

    :param output_ligand_filepaths: List of output ligand structure SDF filepaths.
    :param ligand_output_filepath: Path to the output ligand structure SDF file.
    :param sdf_header: Header string for the SDF file.
    :param method: Method name.
    :param append: Whether to append the predicted ligand structures to the output file.
    :param model_index: Optional index of the model to write to the SDF file.
    :param ligand_numbers: Optional list of ligand numbers represented as a `_`-delimited string or a list of integers.
    :param ligand_names: Optional list of ligand names represented as a `_`-delimited string or a list of strings.
    """
    sdf_content = StringIO()
    sdf_content.write(sdf_header)

    for i, sdf_file in enumerate(output_ligand_filepaths, start=1):
        mol = Chem.MolFromMolFile(sdf_file, sanitize=False)
        mol_frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)

        ligand_numbers_list = (
            (
                ligand_numbers.split("_")
                if isinstance(ligand_numbers, str)
                else [f"{num:03}" for num in ligand_numbers]
            )
            if ligand_numbers
            else [f"{i:03}" for i in range(1, len(mol_frags) + 1)]
        )
        ligand_names_list = (
            (ligand_names.split("_") if isinstance(ligand_names, str) else ligand_names)
            if ligand_names
            else ["LIG"] * len(mol_frags)
        )

        if method == "rfaa":
            if not (len(ligand_numbers_list) == len(ligand_names_list) == len(mol_frags)):
                logger.warning(
                    "Number of RFAA ligand numbers, names, and molecule fragments do not match. Note that this means it did not predict for all input ligands and that manual adjustments to the resulting CASP15 submission file may need to be made (e.g., to make sure ligand names are correctly aligned with listed molecular fragments)."
                )
        else:
            if not (len(ligand_numbers_list) == len(ligand_names_list) == len(mol_frags)):
                logger.warning(
                    f"Number of ligand numbers, names, and molecule fragments must match. Skipping model {i}..."
                )
                continue

        sdf_content.write(f"MODEL {model_index if model_index is not None else i}\n")

        for mol_frag_index, mol_frag in enumerate(mol_frags):
            sdf_content.write(
                f"LIGAND {ligand_numbers_list[mol_frag_index]} {ligand_names_list[mol_frag_index]}\nPOSE 1\n{ligand_names_list[mol_frag_index]}"
            )
            AllChem.ComputeGasteigerCharges(mol_frag)
            sdf_content.write(Chem.MolToMolBlock(mol_frag))

        sdf_content.write("END\n")

    with open(output_ligand_sdf_file, "a" if append else "w") as f:
        f.write(sdf_content.getvalue())

    logger.info(f"CASP15 ligand submission file saved to {output_ligand_sdf_file}.")


def save_ranked_predictions(
    ranked_predictions: RANKED_ENSEMBLE_PREDICTIONS,
    protein_input_filepath: str,
    name: str,
    ligand_numbers: Optional[Union[str, List[int]]],
    ligand_names: Optional[Union[str, List[str]]],
    ligand_tasks: Optional[str],
    cfg: DictConfig,
):
    """Save the top-ranked predictions to the output directory.

    :param ranked_predictions: Dictionary of ranked predictions indexed by each
        prediction's ranking and valued as its method name, output protein filepath,
        output ligand filepath, and average pairwise RMSD or Vina energy score.
    :param protein_input_filepath: Path to the input protein structure PDB file.
    :param name: Name of the target protein-ligand pair.
    :param ligand_numbers: Optional list of ligand numbers represented as a `_`-delimited string or a list of integers.
    :param ligand_names: Optional list of ligand names represented as a `_`-delimited string or a list of strings.
    :param ligand_tasks: Optional ligand tasks specification.
    :param cfg: Configuration dictionary for runtime arguments.
    """
    if ligand_tasks is not None:
        assert ligand_tasks == "P", "Only protein-ligand docking tasks are supported."
    ranking_metric = "ff" if cfg.ensemble_ranking_method == "ff" else "rmsd"
    relax_complex = cfg.relax_method_ligands_pre_ranking or cfg.relax_method_ligands_post_ranking
    ligand_relaxed_suffix = "_relaxed" if relax_complex else ""
    protein_relaxed_suffix = ligand_relaxed_suffix if cfg.relax_protein else ""

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, name + ligand_relaxed_suffix), exist_ok=True)

    relaxation_success_list = []
    if cfg.relax_method_ligands_post_ranking:
        # relax ligand structures after ranking as desired;
        # NOTE: relaxation is practically necessary when making
        # multi-ligand predictions with DiffDock and DynamicBind,
        # and is optional (yet highly recommended) for all other predictions
        pool = multiprocessing.Pool(processes=cfg.relax_num_processes)
        for index, (
            rank,
            (
                method,
                protein_filepath,
                ligand_filepath,
                _,
            ),
        ) in enumerate(ranked_predictions.items()):
            # only relax the top-N predictions as specified
            if cfg.export_top_n is not None:
                if (index + 1) > cfg.export_top_n:
                    break
            if (
                not os.path.exists(ligand_filepath.replace(".sdf", "_ensemble_relaxed.sdf"))
                or not cfg.skip_existing
            ):
                with tempfile.TemporaryDirectory(
                    suffix=f"_{method}_ensemble_cache_dir"
                ) as temp_directory:
                    try:
                        result = pool.apply_async(
                            relax_single_filepair,
                            args=(
                                Path(protein_filepath),
                                Path(ligand_filepath),
                                Path(ligand_filepath).parent,
                                temp_directory,
                                OmegaConf.create(
                                    {
                                        "method": "ensemble",
                                        "add_solvent": cfg.relax_add_solvent,
                                        "name": name,
                                        "prep_only": cfg.relax_prep_only,
                                        "platform": cfg.relax_platform,
                                        "cuda_device_index": cfg.cuda_device_index,
                                        "log_level": cfg.relax_log_level,
                                        "relax_protein": cfg.relax_protein,
                                        "remove_initial_protein_hydrogens": cfg.relax_remove_initial_protein_hydrogens
                                        or method == "rfaa",
                                        "assign_each_ligand_unique_force": cfg.relax_assign_each_ligand_unique_force,
                                        "model_ions": cfg.relax_model_ions,
                                        "cache_files": cfg.relax_cache_files,
                                        "assign_partial_charges_manually": cfg.relax_assign_partial_charges_manually,
                                        "report_initial_energy_only": False,
                                        "max_final_e_value": cfg.relax_max_final_e_value,
                                        "max_num_attempts": cfg.relax_max_num_attempts,
                                        "skip_existing": cfg.relax_skip_existing,
                                    }
                                ),
                            ),
                        )
                        result.get()
                        relaxation_success_list.append(True)
                    except Exception as e:
                        relaxation_success_list.append(False)
                        logger.warning(
                            f"Failed to relax ligand structure {ligand_filepath} from method {method} due to: {e}. Skipping relaxation..."
                        )
            else:
                relaxation_success_list.append(True)

        pool.close()
        pool.join()

        for index, (
            rank,
            (
                method,
                protein_filepath,
                ligand_filepath,
                ranking_metric_value,
            ),
        ) in enumerate(ranked_predictions.items()):
            # only relax the top-N predictions as specified
            if cfg.export_top_n is not None:
                if (index + 1) > cfg.export_top_n:
                    break
            if (
                0 < len(relaxation_success_list) <= len(ranked_predictions)
                and relaxation_success_list[index]
            ):
                assert os.path.exists(
                    ligand_filepath.replace(".sdf", "_ensemble_relaxed.sdf")
                ), f"Ligand relaxation failed for file: {ligand_filepath.replace('.sdf', '_ensemble_relaxed.sdf')}"
                if cfg.relax_protein:
                    maybe_relaxed_protein_filepath = protein_filepath.replace(
                        ".pdb", "_protein_ensemble_relaxed.pdb"
                    )
                    assert os.path.exists(
                        maybe_relaxed_protein_filepath
                    ), f"Protein relaxation failed for file: {maybe_relaxed_protein_filepath}"
                else:
                    maybe_relaxed_protein_filepath = protein_filepath
                ranked_predictions[rank] = (
                    method,
                    maybe_relaxed_protein_filepath,
                    ligand_filepath.replace(".sdf", "_ensemble_relaxed.sdf"),
                    ranking_metric_value,
                )

    # NOTE: we use the `dock` mode here since with each method we implicitly perform cognate (e.g., apo or ab initio) docking,
    # yet for arbitrary input data we do not have access to the ground-truth ligand structures in SDF format
    buster = PoseBusters(config="dock", top_n=None)
    output_ligand_filepaths, output_protein_filepaths = [], []
    for index, (
        rank,
        (
            method,
            protein_filepath,
            ligand_filepath,
            ranking_metric_value,
        ),
    ) in enumerate(ranked_predictions.items()):
        # only save the top-N predictions as specified
        if cfg.export_top_n is not None:
            if (index + 1) > cfg.export_top_n:
                break
        ligand_plddt_match = re.search(r"plddt(\d+\.\d+)", os.path.basename(ligand_filepath))
        ligand_affinity_match = re.search(r"affinity(\d+\.\d+)", os.path.basename(ligand_filepath))
        ligand_plddt_value = float(ligand_plddt_match.group(1)) if ligand_plddt_match else None
        ligand_affinity_value = (
            float(ligand_affinity_match.group(1)) if ligand_affinity_match else None
        )
        ligand_plddt_suffix = f"_plddt{ligand_plddt_value:.7f}" if ligand_plddt_value else ""
        ligand_affinity_suffix = (
            f"_affinity{ligand_affinity_value:.7f}" if ligand_affinity_value else ""
        )
        ligand_output_filepath = os.path.join(
            cfg.output_dir,
            name + ligand_relaxed_suffix,
            f"{method}_rank{rank}_{ranking_metric}{ranking_metric_value:.2e}{ligand_plddt_suffix}{ligand_affinity_suffix}{ligand_relaxed_suffix if 0 < len(relaxation_success_list) <= len(ranked_predictions) and relaxation_success_list[index] else ''}.sdf",
        )
        protein_output_filepath = protein_filepath
        if (
            protein_filepath is not None
            and os.path.basename(protein_filepath) != "ligand_only.pdb"
        ):
            protein_plddt_match = re.search(r"plddt(\d+\.\d+)", os.path.basename(protein_filepath))
            protein_affinity_match = re.search(
                r"affinity(\d+\.\d+)", os.path.basename(protein_filepath)
            )
            protein_plddt_value = (
                float(protein_plddt_match.group(1)) if protein_plddt_match else None
            )
            protein_affinity_value = (
                float(protein_affinity_match.group(1)) if protein_affinity_match else None
            )
            protein_plddt_suffix = (
                f"_plddt{protein_plddt_value:.7f}" if protein_plddt_value else ""
            )
            protein_affinity_suffix = (
                f"_affinity{protein_affinity_value:.7f}" if protein_affinity_value else ""
            )
            protein_output_filepath = os.path.join(
                cfg.output_dir,
                name + ligand_relaxed_suffix,
                f"{method}_rank{rank}_{ranking_metric}{ranking_metric_value:.2e}{protein_plddt_suffix}{protein_affinity_suffix}{protein_relaxed_suffix if 0 < len(relaxation_success_list) <= len(ranked_predictions) and relaxation_success_list[index] else ''}.pdb",
            )

        if not os.path.exists(ligand_output_filepath.replace(".sdf", "_bust_results.csv")):
            # skip PoseBusters validation if the results have previously been saved
            mol_table = pd.DataFrame(
                {
                    "mol_pred": [ligand_filepath],
                    "mol_true": None,
                    "mol_cond": [protein_filepath],
                }
            )
            try:
                bust_results = buster.bust_table(mol_table, full_report=True)
                bust_results["valid"] = (
                    bust_results["mol_pred_loaded"].astype(bool)
                    & bust_results["mol_cond_loaded"].astype(bool)
                    & bust_results["sanitization"].astype(bool)
                    & bust_results["all_atoms_connected"].astype(bool)
                    & bust_results["bond_lengths"].astype(bool)
                    & bust_results["bond_angles"].astype(bool)
                    & bust_results["internal_steric_clash"].astype(bool)
                    & bust_results["aromatic_ring_flatness"].astype(bool)
                    & bust_results["double_bond_flatness"].astype(bool)
                    & bust_results["internal_energy"].astype(bool)
                    & bust_results["protein-ligand_maximum_distance"].astype(bool)
                    & bust_results["minimum_distance_to_protein"].astype(bool)
                    & bust_results["minimum_distance_to_organic_cofactors"].astype(bool)
                    & bust_results["minimum_distance_to_inorganic_cofactors"].astype(bool)
                    & bust_results["minimum_distance_to_waters"].astype(bool)
                    & bust_results["volume_overlap_with_protein"].astype(bool)
                    & bust_results["volume_overlap_with_organic_cofactors"].astype(bool)
                    & bust_results["volume_overlap_with_inorganic_cofactors"].astype(bool)
                    & bust_results["volume_overlap_with_waters"].astype(bool)
                )
                relaxed_mol_is_valid = bust_results["valid"].item()
                bust_results.to_csv(
                    ligand_output_filepath.replace(".sdf", "_bust_results.csv"), index=False
                )
            except Exception as e:
                relaxed_mol_is_valid = False
                logger.warning(
                    f"Failed to PoseBusters-validate ligand structure {ligand_filepath} from method {method} due to: {e}. Skipping validation..."
                )
            shutil.copy(
                ligand_filepath,
                ligand_output_filepath.replace(".sdf", f"_pbvalid={relaxed_mol_is_valid}.sdf"),
            )
            shutil.copy(protein_filepath, protein_output_filepath)

        if (
            relax_complex
            and 0 < len(relaxation_success_list) <= len(ranked_predictions)
            and relaxation_success_list[index]
        ):
            # if applicable, save a copy of the unrelaxed ligand structure as well
            unrelaxed_ligand_filepath = ligand_filepath.replace("_ensemble_relaxed.sdf", ".sdf")
            unrelaxed_protein_filepath = (
                protein_filepath.replace("_ensemble_protein_relaxed.sdf", ".sdf")
                if cfg.relax_protein
                else protein_filepath
            )
            if not os.path.exists(
                ligand_output_filepath.replace("_relaxed", "").replace(".sdf", "_bust_results.csv")
            ):
                # skip (relaxed) PoseBusters validation if the results have previously been saved
                mol_table = pd.DataFrame(
                    {
                        "mol_pred": [unrelaxed_ligand_filepath],
                        "mol_true": None,
                        "mol_cond": [unrelaxed_protein_filepath],
                    }
                )
                try:
                    bust_results = buster.bust_table(mol_table, full_report=True)
                    bust_results["valid"] = (
                        bust_results["mol_pred_loaded"].astype(bool)
                        & bust_results["mol_cond_loaded"].astype(bool)
                        & bust_results["sanitization"].astype(bool)
                        & bust_results["all_atoms_connected"].astype(bool)
                        & bust_results["bond_lengths"].astype(bool)
                        & bust_results["bond_angles"].astype(bool)
                        & bust_results["internal_steric_clash"].astype(bool)
                        & bust_results["aromatic_ring_flatness"].astype(bool)
                        & bust_results["double_bond_flatness"].astype(bool)
                        & bust_results["internal_energy"].astype(bool)
                        & bust_results["protein-ligand_maximum_distance"].astype(bool)
                        & bust_results["minimum_distance_to_protein"].astype(bool)
                        & bust_results["minimum_distance_to_organic_cofactors"].astype(bool)
                        & bust_results["minimum_distance_to_inorganic_cofactors"].astype(bool)
                        & bust_results["minimum_distance_to_waters"].astype(bool)
                        & bust_results["volume_overlap_with_protein"].astype(bool)
                        & bust_results["volume_overlap_with_organic_cofactors"].astype(bool)
                        & bust_results["volume_overlap_with_inorganic_cofactors"].astype(bool)
                        & bust_results["volume_overlap_with_waters"].astype(bool)
                    )
                    unrelaxed_mol_is_valid = bust_results["valid"].item()
                    bust_results.to_csv(
                        ligand_output_filepath.replace("_relaxed", "").replace(
                            ".sdf", "_bust_results.csv"
                        ),
                        index=False,
                    )
                except Exception as e:
                    unrelaxed_mol_is_valid = False
                    logger.warning(
                        f"Failed to PoseBusters-validate ligand structure {unrelaxed_ligand_filepath} from method {method} due to: {e}. Skipping validation..."
                    )
                shutil.copy(
                    unrelaxed_ligand_filepath,
                    ligand_output_filepath.replace("_relaxed", "").replace(
                        ".sdf", f"_pbvalid={unrelaxed_mol_is_valid}.sdf"
                    ),
                )
            if (
                cfg.relax_protein
                and 0 < len(relaxation_success_list) <= len(ranked_predictions)
                and relaxation_success_list[index]
            ):
                # if applicable, save a copy of the unrelaxed protein structure as well
                shutil.copy(
                    unrelaxed_protein_filepath,
                    protein_output_filepath.replace("_relaxed", ""),
                )

        pb_validated_ligand_output_filepaths = glob.glob(
            ligand_output_filepath.replace(".sdf", "_pbvalid=*.sdf")
        )
        assert (
            len(pb_validated_ligand_output_filepaths) == 1
        ), f"Expected one PoseBusters-validated ligand structure file to match {ligand_output_filepath.replace('.sdf', f'_pbvalid=*.sdf')}, but found {pb_validated_ligand_output_filepaths}."
        output_ligand_filepaths.append(pb_validated_ligand_output_filepaths[0])
        output_protein_filepaths.append(protein_output_filepath)

    if cfg.export_file_format is not None and "casp" in cfg.export_file_format:
        # NOTE: relaxed ligand (and potentially protein) files are used for CASP submission when `relax_complex=True`
        pdb_header = (
            f"PFRMAT TS\nTARGET {name}\nAUTHOR {cfg.casp_author}\nMETHOD {cfg.casp_method}\n"
        )
        sdf_header = (
            f"PFRMAT LG\nTARGET {name}\nAUTHOR {cfg.casp_author}\nMETHOD {cfg.casp_method}\n"
        )

        output_protein_filepaths = assign_reference_residue_b_factors(
            output_protein_filepaths, protein_input_filepath
        )

        for i, (protein_output_filepath, ligand_output_filepath) in enumerate(
            zip(output_protein_filepaths, output_ligand_filepaths), start=1
        ):
            output_file = (
                os.path.join(
                    os.path.dirname(ligand_output_filepath),
                    f"{name}LG{cfg.casp_author}_{i}",
                )
                if cfg.combine_casp_output_files
                else os.path.join(
                    os.path.dirname(ligand_output_filepath),
                    f"{name}LG{cfg.casp_author}_protein_{i}",
                )
            )
            method = os.path.basename(ligand_output_filepath).split("_")[0]
            gap_insertion_point = (
                # NOTE: for target `T1124` from CASP15, we have to insert a one-step gap starting at
                # residue `243` in chain `B` for methods that predict holo protein PDB files ab initio
                # to properly score these predictions
                "B:243"
                if name == "T1124" and method in METHODS_PREDICTING_HOLO_PROTEIN_AB_INITIO
                else None
            )
            export_proteins_in_casp_format(
                [protein_output_filepath],
                output_file,
                pdb_header,
                append=cfg.combine_casp_output_files,
                export_casp15_format=cfg.export_file_format == "casp15",
                model_index=i,
                gap_insertion_point=gap_insertion_point,
            )

            output_file = (
                os.path.join(
                    os.path.dirname(ligand_output_filepath),
                    f"{name}LG{cfg.casp_author}_{i}",
                )
                if cfg.combine_casp_output_files
                else os.path.join(
                    os.path.dirname(ligand_output_filepath),
                    f"{name}LG{cfg.casp_author}_ligand_{i}",
                )
            )
            export_ligands_in_casp15_format(
                [ligand_output_filepath],
                output_file,
                sdf_header,
                method,
                append=cfg.combine_casp_output_files,
                model_index=i,
                ligand_numbers=ligand_numbers,
                ligand_names=ligand_names,
            )


@hydra.main(
    version_base="1.3",
    config_path="../../configs/model",
    config_name="ensemble_generation.yaml",
)
def main(cfg: DictConfig):
    """Generate predictions for a protein-ligand target pair using an ensemble
    of methods."""
    os.makedirs(cfg.temp_protein_dir, exist_ok=True)

    with open_dict(cfg):
        # NOTE: besides their output directories, single-sequence baselines are treated like their multi-sequence counterparts
        output_dir = copy.deepcopy(cfg.output_dir)
        cfg.is_ss_ensemble_method = [s.endswith("_ss") for s in cfg.ensemble_methods]
        cfg.ensemble_methods = [s.removesuffix("_ss") for s in cfg.ensemble_methods]
        cfg.output_dir = output_dir

    if list(cfg.ensemble_methods) == ["neuralplexer"] and cfg.neuralplexer_no_ilcl:
        with open_dict(cfg):
            cfg.output_dir = cfg.output_dir.replace(
                "top_neuralplexer",
                "top_neuralplexer_no_ilcl",
            )

    if cfg.diffdock_v1_baseline:
        with open_dict(cfg):
            cfg.output_dir = cfg.output_dir.replace(
                "top_diffdock",
                "top_diffdockv1",
            )
            cfg.diffdock_exec_dir = cfg.diffdock_exec_dir.replace("DiffDock", "DiffDockv1")
            cfg.diffdock_input_csv_path = cfg.diffdock_input_csv_path.replace(
                "DiffDock", "DiffDockv1"
            )
            cfg.diffdock_model_dir = cfg.diffdock_model_dir.replace(
                "forks/DiffDock/workdir/v1.1/score_model",
                "forks/DiffDockv1/workdir/paper_score_model",
            )
            cfg.diffdock_confidence_model_dir = cfg.diffdock_confidence_model_dir.replace(
                "forks/DiffDock/workdir/v1.1/confidence_model",
                "forks/DiffDockv1/workdir/paper_confidence_model",
            )
            cfg.diffdock_output_dir = cfg.diffdock_output_dir.replace("DiffDock", "DiffDockv1")
            cfg.diffdock_actual_steps = 18
            cfg.diffdock_no_final_step_noise = True

    if cfg.pocket_only_baseline:
        with open_dict(cfg):
            cfg.input_csv_filepath = cfg.input_csv_filepath.replace(
                "ensemble_inputs.csv", "ensemble_pocket_only_inputs.csv"
            )
            cfg.output_dir = cfg.output_dir.replace(
                f"top_{cfg.ensemble_ranking_method}",
                f"top_{cfg.ensemble_ranking_method}_pocket_only",
            )

    input_csv_df = pd.read_csv(cfg.input_csv_filepath)
    assert len(input_csv_df.name.unique()) == len(
        input_csv_df
    ), "Names in input CSV must all be unique."
    assert cfg.ensemble_ranking_method in [
        "consensus",
        "ff",
    ], "`ensemble_ranking_method` must be either `consensus` or `ff`."

    # ensure the input CSV contains columns for ligand numbers and names
    if "ligand_numbers" not in input_csv_df.columns:
        input_csv_df["ligand_numbers"] = np.nan
    if "ligand_names" not in input_csv_df.columns:
        input_csv_df["ligand_names"] = np.nan
    if "ligand_tasks" not in input_csv_df.columns:
        input_csv_df["ligand_tasks"] = "P"
    input_csv_df["ligand_numbers"] = input_csv_df["ligand_numbers"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    input_csv_df["ligand_names"] = input_csv_df["ligand_names"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    if cfg.relax_method_ligands_pre_ranking and cfg.relax_method_ligands_post_ranking:
        raise ValueError(
            "Only one of `relax_method_ligands_pre_ranking` and `relax_method_ligands_post_ranking` can be set to `true`."
        )

    if cfg.ensemble_benchmarking:
        if cfg.pocket_only_baseline:
            # NOTE: this is necessary to support protein pocket-based experiments
            with open_dict(cfg):
                cfg.ensemble_benchmarking_apo_protein_dir += "_bs_cropped"
            assert os.path.exists(
                cfg.ensemble_benchmarking_apo_protein_dir
            ), "Ensemble benchmarking for protein pocket-based experiments requires `ensemble_benchmarking_apo_protein_dir` to be set to a valid directory."

        if not os.path.exists(cfg.ensemble_benchmarking_apo_protein_dir):
            # NOTE: this may be necessary to support e.g., CASP15 ensemble benchmarking
            with open_dict(cfg):
                cfg.ensemble_benchmarking_apo_protein_dir = os.path.join(
                    Path(cfg.ensemble_benchmarking_apo_protein_dir).parent,
                    "predicted_structures",
                )
            assert os.path.exists(
                cfg.ensemble_benchmarking_apo_protein_dir
            ), "Ensemble benchmarking requires `ensemble_benchmarking_apo_protein_dir` to be set to a valid directory."
        assert cfg.resume is True, "Ensemble benchmarking requires `resume=True`."

    for row in input_csv_df.itertuples():
        if (
            cfg.ensemble_benchmarking
            and cfg.ensemble_benchmarking_dataset not in row.protein_input.split(os.sep)[-3]
        ):
            raise ValueError(
                f"Row {row.Index} does not belong to the ensemble benchmarking dataset {cfg.ensemble_benchmarking_dataset}."
            )
        if row.ligand_tasks != "P":
            raise ValueError(
                f"Row {row.Index} has an invalid `ligand_tasks` value of {row.ligand_tasks}. Should be `P`."
            )

        config = (
            "_relaxed"
            if cfg.relax_method_ligands_pre_ranking or cfg.relax_method_ligands_post_ranking
            else ""
        )
        if (
            cfg.skip_existing
            and os.path.exists(os.path.join(cfg.output_dir, row.name + config))
            and len(list(glob.glob(os.path.join(cfg.output_dir, row.name + config, "*.sdf"))))
        ):
            logger.info(
                f"Skipping target {row.name + config} as its predictions already exist in {cfg.output_dir}."
            )
            continue

        # ensure an input protein structure is available
        if isinstance(row.protein_input, str) and os.path.exists(row.protein_input):
            temp_protein_filepath = row.protein_input
        else:
            if cfg.ensemble_benchmarking:
                logging.warning(
                    f"The input (e.g., predicted) protein structure ({row.protein_input}) must be locally available for ensemble benchmarking. Skipping target {row.name + config}."
                )
                continue
            # NOTE: a placeholder protein sequence is used when making ligand-only predictions
            row_protein_input = (
                row.protein_input
                if isinstance(row.protein_input, str) and len(row.protein_input) > 0
                else LIGAND_ONLY_RECEPTOR_PLACEHOLDER_SEQUENCE
            )
            row_name = (
                "ligand_only"
                if row_protein_input == LIGAND_ONLY_RECEPTOR_PLACEHOLDER_SEQUENCE
                else row.name
            )
            temp_protein_filepath = os.path.join(cfg.temp_protein_dir, f"{row_name}.pdb")
            if not os.path.exists(temp_protein_filepath):
                temp_fasta_filepath = create_temporary_fasta_file(row_protein_input, name=row_name)
                predict_protein_structure_from_sequence(
                    cfg.diffdock_python_exec_path,
                    cfg.structure_prediction_script_path,
                    temp_fasta_filepath,
                    cfg.temp_protein_dir,
                    chunk_size=cfg.structure_prediction_chunk_size,
                    cpu_only=cfg.structure_prediction_cpu_only,
                    cpu_offload=cfg.structure_prediction_cpu_offload,
                    cuda_device_index=cfg.cuda_device_index,
                )
            if not os.path.exists(temp_protein_filepath):
                raise FileNotFoundError(
                    f"Predicted protein structure not found at {temp_protein_filepath}."
                )

        # generate bound protein-ligand complex predictions using all selected methods
        method_filepaths_mapping = (
            {
                method: get_method_predictions(method, row.name, cfg)
                for method in cfg.vina_binding_site_methods
            }
            if cfg.generate_vina_scripts and cfg.resume
            else None
        )
        ensemble_predictions_dict, generating_scripts = generate_ensemble_predictions(
            temp_protein_filepath,
            row.ligand_smiles,
            row.name,
            cfg,
            method_filepaths_mapping=method_filepaths_mapping,
            generate_hpc_scripts=cfg.generate_hpc_scripts,
        )

        # inform the user of next steps
        if generating_scripts:
            logger.info(
                f"All method prediction scripts for target {row.name} have been generated. Please run these scripts and continue this ensemble generation pipeline with `resume=True`."
            )
            continue

        # skip to the next target if no predictions from any method were found
        predictions_found = any(
            len(ensemble_predictions_dict[method]) for method in ensemble_predictions_dict
        )
        if not predictions_found:
            logger.warning(
                f"No predictions from any method found for target {row.name}. Skipping..."
            )
            continue

        # rank the predictions to select the top prediction(s)
        ranked_predictions = rank_ensemble_predictions(ensemble_predictions_dict, row.name, cfg)
        if not len(ranked_predictions):
            logger.warning(f"No ranked predictions found for target {row.name}. Skipping...")
            continue

        # save the top-ranked predictions to the output directory
        save_ranked_predictions(
            ranked_predictions,
            temp_protein_filepath,
            row.name,
            (
                None
                if isinstance(row, np.ndarray) and np.isnan(row.ligand_numbers).any()
                else row.ligand_numbers
            ),
            (
                None
                if isinstance(row, np.ndarray) and np.isnan(row.ligand_names).any()
                else row.ligand_names
            ),
            (
                None
                if isinstance(row, np.ndarray) and np.isnan(row.ligand_tasks).any()
                else row.ligand_tasks
            ),
            cfg,
        )
        logger.info(f"Ensemble generation for target {row.name} has been completed.")


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
