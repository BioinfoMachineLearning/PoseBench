# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import logging
import os
from typing import Literal

import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from posebench.models.ensemble_generation import insert_hpc_headers

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Commands dictionary
COMMANDS = {
    "diffdock": {
        "prepare_input": [
            "python3 posebench/data/diffdock_input_preparation.py dataset={dataset}",
        ],
        "run_inference": [
            "python3 posebench/models/diffdock_inference.py dataset={dataset} repeat_index={repeat_index}",
        ],
        "relax": [
            "python3 posebench/models/inference_relaxation.py method=diffdock dataset={dataset} remove_initial_protein_hydrogens=true assign_partial_charges_manually=true num_processes=1 repeat_index={repeat_index}",
        ],
        "analyze_results": [
            "python3 posebench/analysis/inference_analysis.py method=diffdock dataset={dataset} repeat_index={repeat_index}",
        ],
        "assemble_casp15": [
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[diffdock] input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_diffdock_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=false export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=40 method_top_n_to_select=40 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[diffdock] input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_diffdock_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=true export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=40 method_top_n_to_select=40 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
        ],
        "analyze_casp15": [
            "python3 posebench/analysis/inference_analysis_casp.py method=diffdock dataset=casp15 repeat_index={repeat_index}",
        ],
    },
    "fabind": {
        "prepare_input": [
            "python3 posebench/data/fabind_input_preparation.py dataset={dataset}",
        ],
        "run_inference": [
            "python3 posebench/models/fabind_inference.py dataset={dataset} repeat_index={repeat_index}",
        ],
        "relax": [
            "python3 posebench/models/inference_relaxation.py method=fabind dataset={dataset} remove_initial_protein_hydrogens=true assign_partial_charges_manually=true num_processes=1 repeat_index={repeat_index}",
        ],
        "analyze_results": [
            "python3 posebench/analysis/inference_analysis.py method=fabind dataset={dataset} repeat_index={repeat_index}",
        ],
    },
    "dynamicbind": {
        "prepare_input": [
            "python3 posebench/data/dynamicbind_input_preparation.py dataset={dataset}",
            'python3 posebench/data/dynamicbind_input_preparation.py dataset=casp15 input_data_dir="$PWD"/data/casp15_set/targets',
        ],
        "run_inference": [
            "python3 posebench/models/dynamicbind_inference.py dataset={dataset} repeat_index={repeat_index}",
            'python3 posebench/models/dynamicbind_inference.py dataset=casp15 batch_size=1 input_data_dir="$PWD"/data/casp15_set/predicted_structures repeat_index={repeat_index}',
        ],
        "relax": [
            "python3 posebench/models/inference_relaxation.py method=dynamicbind dataset={dataset} remove_initial_protein_hydrogens=true assign_partial_charges_manually=true num_processes=1 repeat_index={repeat_index}",
        ],
        "analyze_results": [
            "python3 posebench/analysis/inference_analysis.py method=dynamicbind dataset={dataset} repeat_index={repeat_index}",
        ],
        "assemble_casp15": [
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[dynamicbind] input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_dynamicbind_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=false export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=40 method_top_n_to_select=40 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[dynamicbind] input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_dynamicbind_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=true export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=40 method_top_n_to_select=40 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
        ],
        "analyze_casp15": [
            "python3 posebench/analysis/inference_analysis_casp.py method=dynamicbind dataset=casp15 repeat_index={repeat_index}",
        ],
    },
    "neuralplexer": {
        "prepare_input": [
            "python3 posebench/data/neuralplexer_input_preparation.py dataset={dataset}",
            'python3 posebench/data/neuralplexer_input_preparation.py dataset=casp15 input_data_dir="$PWD"/data/casp15_set/targets input_receptor_structure_dir="$PWD"/data/casp15_set/predicted_structures',
        ],
        "run_inference": [
            "python3 posebench/models/neuralplexer_inference.py dataset={dataset} repeat_index={repeat_index}",
            "python3 posebench/models/neuralplexer_inference.py dataset=casp15 repeat_index={repeat_index}",
        ],
        "relax": [
            "python3 posebench/models/inference_relaxation.py method=neuralplexer dataset={dataset} num_processes=1 remove_initial_protein_hydrogens=true assign_partial_charges_manually=true cache_files=false repeat_index={repeat_index}",
        ],
        "align_complexes": [
            "python3 posebench/analysis/complex_alignment.py method=neuralplexer dataset={dataset} repeat_index={repeat_index}",
        ],
        "analyze_results": [
            "python3 posebench/analysis/inference_analysis.py method=neuralplexer dataset={dataset} repeat_index={repeat_index}",
        ],
        "assemble_casp15": [
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[neuralplexer] input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_neuralplexer_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=false export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=40 method_top_n_to_select=40 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[neuralplexer] input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_neuralplexer_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=true export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=40 method_top_n_to_select=40 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
        ],
        "analyze_casp15": [
            "python3 posebench/analysis/inference_analysis_casp.py method=neuralplexer dataset=casp15 repeat_index={repeat_index}",
        ],
    },
    "rfaa": {
        "prepare_input": [
            "python3 posebench/data/rfaa_input_preparation.py dataset={dataset}",
            'python3 posebench/data/rfaa_input_preparation.py dataset=casp15 input_data_dir="$PWD"/data/casp15_set/targets',
        ],
        "run_inference": [
            "conda activate forks/RoseTTAFold-All-Atom/RFAA/",
            "python3 posebench/models/rfaa_inference.py dataset={dataset} run_inference_directly=true",
            "python3 posebench/models/rfaa_inference.py dataset=casp15 run_inference_directly=true",
            "conda deactivate",
        ],
        "extract_outputs": [
            "python3 posebench/data/rfaa_output_extraction.py dataset={dataset}",
        ],
        "relax": [
            "python3 posebench/models/inference_relaxation.py method=rfaa dataset={dataset} num_processes=1 remove_initial_protein_hydrogens=true",
        ],
        "align_complexes": [
            "python3 posebench/analysis/complex_alignment.py method=rfaa dataset={dataset}",
        ],
        "analyze_results": [
            "python3 posebench/analysis/inference_analysis.py method=rfaa dataset={dataset}",
        ],
        "assemble_casp15": [
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[rfaa] input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_rfaa_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=false export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=40 method_top_n_to_select=40 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[rfaa] input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_rfaa_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=true export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=40 method_top_n_to_select=40 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
        ],
        "analyze_casp15": [
            "python3 posebench/analysis/inference_analysis_casp.py method=rfaa dataset=casp15 repeat_index={repeat_index}",
        ],
    },
    "vina": {
        "prepare_input": [
            "cp forks/DiffDock/inference/diffdock_{dataset}_inputs.csv forks/Vina/inference/vina_{dataset}_inputs.csv",
        ],
        "run_inference": [
            "python3 posebench/models/vina_inference.py dataset={dataset} method={vina_binding_site_method} repeat_index={repeat_index}",
        ],
        "copy_predictions": [
            "mkdir -p forks/Vina/inference/vina_{vina_binding_site_method}_{dataset}_outputs_{repeat_index} && cp -r data/test_cases/{dataset}/vina_{vina_binding_site_method}_{dataset}_outputs_{repeat_index}/* forks/Vina/inference/vina_{vina_binding_site_method}_{dataset}_outputs_{repeat_index}",
        ],
        "relax": [
            "python3 posebench/models/inference_relaxation.py method=vina vina_binding_site_method={vina_binding_site_method} dataset={dataset} remove_initial_protein_hydrogens=true assign_partial_charges_manually=true num_processes=1 repeat_index={repeat_index}",
        ],
        "analyze_results": [
            "python3 posebench/analysis/inference_analysis.py method=vina vina_binding_site_method={vina_binding_site_method} dataset={dataset} repeat_index={repeat_index}",
        ],
        "analyze_casp15": [
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[vina] vina_binding_site_methods=[{vina_binding_site_method}] input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_vina_{vina_binding_site_method}_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=false export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=40 method_top_n_to_select=40 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset={dataset} cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[vina] vina_binding_site_methods=[{vina_binding_site_method}] input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_vina_{vina_binding_site_method}_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=true export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=40 method_top_n_to_select=40 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset={dataset} cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
            "python3 posebench/analysis/inference_analysis_casp.py method=vina vina_binding_site_method={vina_binding_site_method} dataset={dataset} repeat_index={repeat_index}",
        ],
    },
    "tulip": {
        "prepare_input": [
            "python3 posebench/data/tulip_output_extraction.py dataset={dataset}",
        ],
        "relax": [
            "python3 posebench/models/inference_relaxation.py method=tulip dataset={dataset} remove_initial_protein_hydrogens=true assign_partial_charges_manually=true num_processes=1 repeat_index={repeat_index}",
        ],
        "analyze_results": [
            "python3 posebench/analysis/inference_analysis.py method=tulip dataset={dataset} repeat_index={repeat_index}",
        ],
        "analyze_casp15": [
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[tulip] input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_tulip_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=false export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=40 method_top_n_to_select=40 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset={dataset} cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[tulip] input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_tulip_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=true export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=40 method_top_n_to_select=40 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset={dataset} cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
            "python3 posebench/analysis/inference_analysis_casp.py method=tulip dataset={dataset} repeat_index={repeat_index} targets='[H1135, H1171v1, H1171v2, H1172v1, H1172v2, H1172v3, H1172v4, T1124, T1127v2, T1152, T1158v1, T1158v2, T1158v3, T1158v4, T1186, T1187]'",
        ],
    },
}
VINA_BINDING_SITE_METHODS = ["diffdock", "p2rank"]
SINGLE_RUN_METHODS = ["fabind", "rfaa", "tulip"]
GPU_ENABLED_METHODS = ["diffdock", "fabind", "dynamicbind", "neuralplexer"]
DATASETS = ["posebusters_benchmark", "astex_diverse", "dockgen", "casp15"]


def build_inference_script(
    method: Literal["diffdock", "fabind", "dynamicbind", "neuralplexer", "rfaa", "vina", "tulip"],
    vina_binding_site_method: Literal[
        "diffdock", "fabind", "dynamicbind", "neuralplexer", "rfaa", "p2rank"
    ],
    dataset: Literal["posebusters_benchmark", "astex_diverse", "dockgen", "casp15"],
    repeat_index: int,
    cuda_device_index: int,
    output_script_dir: str,
    export_hpc_headers: bool = False,
    gpu_partition: str = "chengji-lab-gpu",
    gpu_account: str = "chengji-lab",
    gpu_type: Literal["A100", "H100"] = "A100",
    cpu_memory_in_gb: int = 59,
    time_limit: str = "2-00:00:00",
):
    """Build an inference script according to user arguments.

    :param method: Inference method to use.
    :param vina_binding_site_method: Vina binding site method to use.
    :param dataset: Dataset to use.
    :param repeat_index: Index of the repeat.
    :param cuda_device_index: Index of the CUDA device to use.
    :param output_script_dir: Output script directory.
    :param export_hpc_headers: Whether to export HPC headers.
    :param gpu_partition: GPU partition to use.
    :param gpu_account: GPU account to use.
    :param gpu_type: GPU type to use.
    :param cpu_memory_in_gb: CPU memory in GB.
    :param time_limit: Time limit.
    """
    commands = COMMANDS.get(method)
    if not commands:
        raise ValueError(f"Unsupported method: {method}")

    if method in SINGLE_RUN_METHODS and repeat_index > 1:
        logging.info(
            f"Method {method} does not support multiple repeats. Skipping repeat_index {repeat_index}."
        )
        return

    os.makedirs(output_script_dir, exist_ok=True)
    output_script = os.path.join(
        output_script_dir,
        f"{method}_{dataset}{'_hpc' if export_hpc_headers else ''}_inference_{repeat_index}.sh",
    )

    with open(output_script, "w") as f:
        if export_hpc_headers:
            f.write(
                insert_hpc_headers(
                    method=method,
                    gpu_partition=gpu_partition,
                    gpu_account=gpu_account,
                    gpu_type=gpu_type,
                    cpu_memory_in_gb=cpu_memory_in_gb,
                    time_limit=time_limit,
                )
            )
            f.write("\nconda activate PoseBench\n\n")

        # Prepare input files
        f.write("# Prepare input files\n")
        for cmd in commands.get("prepare_input", []):
            f.write(cmd.format(dataset=dataset) + "\n")
        f.write("\n")

        # Run inference
        f.write("# Run inference\n")
        run_inference_cmds = commands.get("run_inference", [])
        for cmd in run_inference_cmds:
            f.write(
                cmd.format(
                    dataset=dataset,
                    repeat_index=repeat_index,
                    cuda_device_index=cuda_device_index,
                    vina_binding_site_method=vina_binding_site_method,
                )
                + "\n"
            )
        f.write("\n")

        # Relax generated ligand structures
        f.write("# Relax generated ligand structures\n")
        for cmd in commands.get("relax", []):
            f.write(
                cmd.format(
                    dataset=dataset,
                    repeat_index=repeat_index,
                    vina_binding_site_method=vina_binding_site_method,
                )
                + "\n"
            )
        f.write("\n")

        # Analyze inference results
        f.write("# Analyze inference results\n")
        for cmd in commands.get("analyze_results", []):
            f.write(
                cmd.format(
                    dataset=dataset,
                    repeat_index=repeat_index,
                    vina_binding_site_method=vina_binding_site_method,
                )
                + "\n"
            )
        f.write("\n")

        # Assemble CASP15 (if applicable)
        if dataset == "casp15" and "assemble_casp15" in commands:
            f.write("# Assemble CASP15 results\n")
            for cmd in commands.get("assemble_casp15", []):
                f.write(
                    cmd.format(
                        dataset=dataset,
                        repeat_index=repeat_index,
                        cuda_device_index=cuda_device_index,
                    )
                    + "\n"
                )
            f.write("\n")

        # Analyze CASP15 results (if applicable)
        if dataset == "casp15" and "analyze_casp15" in commands:
            f.write("# Analyze CASP15 results\n")
            for cmd in commands.get("analyze_casp15", []):
                f.write(
                    cmd.format(
                        dataset=dataset,
                        repeat_index=repeat_index,
                        vina_binding_site_method=vina_binding_site_method,
                        cuda_device_index=cuda_device_index,
                    )
                    + "\n"
                )
            f.write("\n")

    logging.info(f"Script {output_script} created successfully.")


def build_inference_scripts(
    num_repeats: int,
    cuda_device_index: int,
    output_script_dir: str,
    export_hpc_headers: bool = False,
):
    """Build inference scripts according to user arguments.

    :param num_repeats: Number of repeats total.
    :param cuda_device_index: Index of the CUDA device to use.
    :param output_script: Output script file.
    :param export_hpc_headers: Whether to export HPC headers.
    """
    for method in COMMANDS:
        for vina_binding_site_method in VINA_BINDING_SITE_METHODS:
            for dataset in DATASETS:
                for repeat_index in range(1, num_repeats + 1):
                    build_inference_script(
                        method=method,
                        vina_binding_site_method=vina_binding_site_method,
                        dataset=dataset,
                        repeat_index=repeat_index,
                        cuda_device_index=cuda_device_index,
                        output_script_dir=output_script_dir,
                        export_hpc_headers=export_hpc_headers,
                    )


@hydra.main(
    version_base="1.3",
    config_path="../configs/scripts",
    config_name="build_inference_script.yaml",
)
def main(cfg: DictConfig):
    """Build an inference script according to user arguments."""
    if cfg.build_all_scripts:
        build_inference_scripts(
            num_repeats=cfg.num_repeats,
            cuda_device_index=cfg.cuda_device_index,
            output_script_dir=cfg.output_script_dir,
            export_hpc_headers=cfg.export_hpc_headers,
        )
    else:
        build_inference_script(
            method=cfg.method,
            dataset=cfg.dataset,
            repeat_index=cfg.repeat_index,
            cuda_device_index=cfg.cuda_device_index,
            output_script_dir=cfg.output_script_dir,
            export_hpc_headers=cfg.export_hpc_headers,
        )


if __name__ == "__main__":
    main()