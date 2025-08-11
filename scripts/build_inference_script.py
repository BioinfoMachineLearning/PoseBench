# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import logging
import os
from typing import List, Literal, Optional

import hydra
import rootutils
from beartype import beartype
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from posebench.models.ensemble_generation import insert_hpc_headers

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Constants
COMMANDS = {
    "diffdock": {
        "prepare_input": [
            "python3 posebench/data/diffdock_input_preparation.py dataset={dataset} pocket_only_baseline={pocket_only_baseline}",
        ],
        "run_inference": [
            "python3 posebench/models/diffdock_inference.py dataset={dataset} cuda_device_index={cuda_device_index} pocket_only_baseline={pocket_only_baseline} v1_baseline={v1_baseline} repeat_index={repeat_index}",
        ],
        "relax": [
            "python3 posebench/models/inference_relaxation.py method=diffdock dataset={dataset} cuda_device_index={cuda_device_index} pocket_only_baseline={pocket_only_baseline} v1_baseline={v1_baseline} relax_protein={relax_protein} remove_initial_protein_hydrogens=true assign_partial_charges_manually=true repeat_index={repeat_index}",
        ],
        "analyze_results": [
            "python3 posebench/analysis/inference_analysis.py method=diffdock dataset={dataset} pocket_only_baseline={pocket_only_baseline} v1_baseline={v1_baseline} relax_protein={relax_protein} repeat_index={repeat_index}",
        ],
        "assemble_casp15": [
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[diffdock] ensemble_ranking_method={ensemble_ranking_method} diffdock_v1_baseline={v1_baseline} input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_diffdock_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=false relax_protein={relax_protein} export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=5 method_top_n_to_select=5 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[diffdock] ensemble_ranking_method={ensemble_ranking_method} diffdock_v1_baseline={v1_baseline} input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_diffdock_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=true relax_protein={relax_protein} export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=5 method_top_n_to_select=5 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
        ],
        "analyze_casp15": [
            "python3 posebench/analysis/inference_analysis_casp.py method=diffdock dataset=casp15 relax_protein={relax_protein} v1_baseline={v1_baseline} repeat_index={repeat_index}",
        ],
    },
    "fabind": {
        "prepare_input": [
            "python3 posebench/data/fabind_input_preparation.py dataset={dataset} pocket_only_baseline={pocket_only_baseline}",
        ],
        "run_inference": [
            "python3 posebench/models/fabind_inference.py dataset={dataset} cuda_device_index={cuda_device_index} pocket_only_baseline={pocket_only_baseline} repeat_index={repeat_index}",
        ],
        "relax": [
            "python3 posebench/models/inference_relaxation.py method=fabind dataset={dataset} cuda_device_index={cuda_device_index} pocket_only_baseline={pocket_only_baseline} relax_protein={relax_protein} remove_initial_protein_hydrogens=true assign_partial_charges_manually=true repeat_index={repeat_index}",
        ],
        "analyze_results": [
            "python3 posebench/analysis/inference_analysis.py method=fabind dataset={dataset} pocket_only_baseline={pocket_only_baseline} relax_protein={relax_protein} repeat_index={repeat_index}",
        ],
    },
    "dynamicbind": {
        "prepare_input": [
            "python3 posebench/data/dynamicbind_input_preparation.py dataset={dataset} pocket_only_baseline={pocket_only_baseline}",
        ],
        "run_inference": [
            "python3 posebench/models/dynamicbind_inference.py dataset={dataset} cuda_device_index={cuda_device_index} pocket_only_baseline={pocket_only_baseline} repeat_index={repeat_index}",
        ],
        "relax": [
            "python3 posebench/models/inference_relaxation.py method=dynamicbind dataset={dataset} cuda_device_index={cuda_device_index} pocket_only_baseline={pocket_only_baseline} relax_protein={relax_protein} remove_initial_protein_hydrogens=true assign_partial_charges_manually=true repeat_index={repeat_index}",
        ],
        "analyze_results": [
            "python3 posebench/analysis/inference_analysis.py method=dynamicbind dataset={dataset} pocket_only_baseline={pocket_only_baseline} relax_protein={relax_protein} repeat_index={repeat_index}",
        ],
        "assemble_casp15": [
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[dynamicbind] ensemble_ranking_method={ensemble_ranking_method} input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_dynamicbind_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=false relax_protein={relax_protein} export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=5 method_top_n_to_select=5 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[dynamicbind] ensemble_ranking_method={ensemble_ranking_method} input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_dynamicbind_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=true relax_protein={relax_protein} export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=5 method_top_n_to_select=5 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
        ],
        "analyze_casp15": [
            "python3 posebench/analysis/inference_analysis_casp.py method=dynamicbind dataset=casp15 relax_protein={relax_protein} repeat_index={repeat_index}",
        ],
    },
    "neuralplexer": {
        "prepare_input": [
            "python3 posebench/data/neuralplexer_input_preparation.py dataset={dataset} pocket_only_baseline={pocket_only_baseline}",
        ],
        "run_inference": [
            "python3 posebench/models/neuralplexer_inference.py dataset={dataset} cuda_device_index={cuda_device_index} pocket_only_baseline={pocket_only_baseline} no_ilcl={no_ilcl} repeat_index={repeat_index}",
        ],
        "relax": [
            "python3 posebench/models/inference_relaxation.py method=neuralplexer dataset={dataset} cuda_device_index={cuda_device_index} pocket_only_baseline={pocket_only_baseline} relax_protein={relax_protein} remove_initial_protein_hydrogens=true assign_partial_charges_manually=true repeat_index={repeat_index}",
        ],
        "align_complexes": [
            "conda activate PyMOL-PoseBench",
            "python3 posebench/analysis/complex_alignment.py method=neuralplexer dataset={dataset} pocket_only_baseline={pocket_only_baseline} repeat_index={repeat_index}",
            "conda deactivate",
        ],
        "analyze_results": [
            "python3 posebench/analysis/inference_analysis.py method=neuralplexer dataset={dataset} pocket_only_baseline={pocket_only_baseline} relax_protein={relax_protein} repeat_index={repeat_index}",
        ],
        "assemble_casp15": [
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[neuralplexer] ensemble_ranking_method={ensemble_ranking_method} neuralplexer_no_ilcl={no_ilcl} input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_neuralplexer_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=false relax_protein={relax_protein} export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=5 method_top_n_to_select=5 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[neuralplexer] ensemble_ranking_method={ensemble_ranking_method} neuralplexer_no_ilcl={no_ilcl} input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_neuralplexer_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=true relax_protein={relax_protein} export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=5 method_top_n_to_select=5 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
        ],
        "analyze_casp15": [
            "python3 posebench/analysis/inference_analysis_casp.py method=neuralplexer dataset=casp15 no_ilcl={no_ilcl} relax_protein={relax_protein} repeat_index={repeat_index}",
        ],
    },
    "flowdock": {
        "prepare_input": [
            "python3 posebench/data/flowdock_input_preparation.py dataset={dataset} pocket_only_baseline={pocket_only_baseline}",
        ],
        "run_inference": [
            "python3 posebench/models/flowdock_inference.py dataset={dataset} cuda_device_index={cuda_device_index} pocket_only_baseline={pocket_only_baseline} repeat_index={repeat_index}",
        ],
        "relax": [
            "python3 posebench/models/inference_relaxation.py method=flowdock dataset={dataset} cuda_device_index={cuda_device_index} pocket_only_baseline={pocket_only_baseline} relax_protein={relax_protein} remove_initial_protein_hydrogens=true assign_partial_charges_manually=true repeat_index={repeat_index}",
        ],
        "align_complexes": [
            "conda activate PyMOL-PoseBench",
            "python3 posebench/analysis/complex_alignment.py method=flowdock dataset={dataset} pocket_only_baseline={pocket_only_baseline} repeat_index={repeat_index}",
            "conda deactivate",
        ],
        "analyze_results": [
            "python3 posebench/analysis/inference_analysis.py method=flowdock dataset={dataset} pocket_only_baseline={pocket_only_baseline} relax_protein={relax_protein} repeat_index={repeat_index}",
        ],
        "assemble_casp15": [
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[flowdock] ensemble_ranking_method={ensemble_ranking_method} input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_flowdock_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=false relax_protein={relax_protein} export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=5 method_top_n_to_select=5 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[flowdock] ensemble_ranking_method={ensemble_ranking_method} input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_flowdock_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=true relax_protein={relax_protein} export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=5 method_top_n_to_select=5 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
        ],
        "analyze_casp15": [
            "python3 posebench/analysis/inference_analysis_casp.py method=flowdock dataset=casp15 relax_protein={relax_protein} repeat_index={repeat_index}",
        ],
    },
    "rfaa": {
        "prepare_input": [
            "python3 posebench/data/rfaa_input_preparation.py dataset={dataset} pocket_only_baseline={pocket_only_baseline}",
        ],
        "run_inference": [
            "conda activate forks/RoseTTAFold-All-Atom/RFAA/",
            "python3 posebench/models/rfaa_inference.py dataset={dataset} cuda_device_index={cuda_device_index} pocket_only_baseline={pocket_only_baseline} run_inference_directly=true",
            "conda deactivate",
        ],
        "extract_outputs": [
            "python3 posebench/data/rfaa_output_extraction.py dataset={dataset} pocket_only_baseline={pocket_only_baseline} repeat_index={repeat_index}",
        ],
        "relax": [
            "python3 posebench/models/inference_relaxation.py method=rfaa dataset={dataset} cuda_device_index={cuda_device_index} pocket_only_baseline={pocket_only_baseline} relax_protein={relax_protein} remove_initial_protein_hydrogens=true",
        ],
        "align_complexes": [
            "conda activate PyMOL-PoseBench",
            "python3 posebench/analysis/complex_alignment.py method=rfaa dataset={dataset} pocket_only_baseline={pocket_only_baseline}",
            "conda deactivate",
        ],
        "analyze_results": [
            "python3 posebench/analysis/inference_analysis.py method=rfaa dataset={dataset} pocket_only_baseline={pocket_only_baseline} relax_protein={relax_protein}",
        ],
        "assemble_casp15": [
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[rfaa] ensemble_ranking_method={ensemble_ranking_method} input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_rfaa_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=false relax_protein={relax_protein} export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=5 method_top_n_to_select=5 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[rfaa] ensemble_ranking_method={ensemble_ranking_method} input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_rfaa_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=true relax_protein={relax_protein} export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=5 method_top_n_to_select=5 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
        ],
        "analyze_casp15": [
            "python3 posebench/analysis/inference_analysis_casp.py method=rfaa dataset=casp15 repeat_index={repeat_index} relax_protein={relax_protein}",
        ],
    },
    "chai-lab": {
        "prepare_input": [
            "python3 posebench/data/chai_input_preparation.py dataset={dataset} pocket_only_baseline={pocket_only_baseline}",
        ],
        "run_inference": [
            "conda activate forks/chai-lab/chai-lab/",
            "python3 posebench/models/chai_inference.py dataset={dataset} cuda_device_index={cuda_device_index} pocket_only_baseline={pocket_only_baseline} repeat_index={repeat_index}",
            "conda deactivate",
        ],
        "extract_outputs": [
            "python3 posebench/data/chai_output_extraction.py dataset={dataset} pocket_only_baseline={pocket_only_baseline} repeat_index={repeat_index}",
        ],
        "relax": [
            "python3 posebench/models/inference_relaxation.py method=chai-lab dataset={dataset} cuda_device_index={cuda_device_index} pocket_only_baseline={pocket_only_baseline} relax_protein={relax_protein} remove_initial_protein_hydrogens=true repeat_index={repeat_index}",
        ],
        "align_complexes": [
            "conda activate PyMOL-PoseBench",
            "python3 posebench/analysis/complex_alignment.py method=chai-lab dataset={dataset} pocket_only_baseline={pocket_only_baseline} repeat_index={repeat_index}",
            "conda deactivate",
        ],
        "analyze_results": [
            "python3 posebench/analysis/inference_analysis.py method=chai-lab dataset={dataset} pocket_only_baseline={pocket_only_baseline} relax_protein={relax_protein} repeat_index={repeat_index}",
        ],
        "assemble_casp15": [
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[chai-lab] ensemble_ranking_method={ensemble_ranking_method} input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_chai-lab_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=false relax_protein={relax_protein} export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=5 method_top_n_to_select=5 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[chai-lab] ensemble_ranking_method={ensemble_ranking_method} input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_chai-lab_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=true relax_protein={relax_protein} export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=5 method_top_n_to_select=5 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
        ],
        "analyze_casp15": [
            "python3 posebench/analysis/inference_analysis_casp.py method=chai-lab dataset=casp15 repeat_index={repeat_index} relax_protein={relax_protein}",
        ],
    },
    "boltz": {
        "prepare_input": [
            "python3 posebench/data/boltz_input_preparation.py dataset={dataset} pocket_only_baseline={pocket_only_baseline}",
        ],
        "run_inference": [
            "conda activate forks/boltz/boltz/",
            "python3 posebench/models/boltz_inference.py dataset={dataset} cuda_device_index={cuda_device_index} pocket_only_baseline={pocket_only_baseline} repeat_index={repeat_index}",
            "conda deactivate",
        ],
        "extract_outputs": [
            "python3 posebench/data/boltz_output_extraction.py dataset={dataset} pocket_only_baseline={pocket_only_baseline} repeat_index={repeat_index}",
        ],
        "relax": [
            "python3 posebench/models/inference_relaxation.py method=boltz dataset={dataset} cuda_device_index={cuda_device_index} pocket_only_baseline={pocket_only_baseline} relax_protein={relax_protein} remove_initial_protein_hydrogens=true repeat_index={repeat_index}",
        ],
        "align_complexes": [
            "conda activate PyMOL-PoseBench",
            "python3 posebench/analysis/complex_alignment.py method=boltz dataset={dataset} pocket_only_baseline={pocket_only_baseline} repeat_index={repeat_index}",
            "conda deactivate",
        ],
        "analyze_results": [
            "python3 posebench/analysis/inference_analysis.py method=boltz dataset={dataset} pocket_only_baseline={pocket_only_baseline} relax_protein={relax_protein} repeat_index={repeat_index}",
        ],
        "assemble_casp15": [
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[boltz] ensemble_ranking_method={ensemble_ranking_method} input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_boltz_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=false relax_protein={relax_protein} export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=5 method_top_n_to_select=5 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[boltz] ensemble_ranking_method={ensemble_ranking_method} input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_boltz_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=true relax_protein={relax_protein} export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=5 method_top_n_to_select=5 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
        ],
        "analyze_casp15": [
            "python3 posebench/analysis/inference_analysis_casp.py method=boltz dataset=casp15 repeat_index={repeat_index} relax_protein={relax_protein}",
        ],
    },
    "alphafold3": {
        "extract_outputs": [
            "python3 posebench/data/af3_output_extraction.py dataset={dataset} pocket_only_baseline={pocket_only_baseline} repeat_index={repeat_index}",
        ],
        "relax": [
            "python3 posebench/models/inference_relaxation.py method=alphafold3 dataset={dataset} cuda_device_index={cuda_device_index} pocket_only_baseline={pocket_only_baseline} relax_protein={relax_protein} remove_initial_protein_hydrogens=true repeat_index={repeat_index}",
        ],
        "align_complexes": [
            "conda activate PyMOL-PoseBench",
            "python3 posebench/analysis/complex_alignment.py method=alphafold3 dataset={dataset} pocket_only_baseline={pocket_only_baseline} repeat_index={repeat_index}",
            "conda deactivate",
        ],
        "analyze_results": [
            "python3 posebench/analysis/inference_analysis.py method=alphafold3 dataset={dataset} pocket_only_baseline={pocket_only_baseline} relax_protein={relax_protein} repeat_index={repeat_index}",
        ],
        "assemble_casp15": [
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[alphafold3] ensemble_ranking_method={ensemble_ranking_method} input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_alphafold3_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=false relax_protein={relax_protein} export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=5 method_top_n_to_select=5 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[alphafold3] ensemble_ranking_method={ensemble_ranking_method} input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_alphafold3_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=true relax_protein={relax_protein} export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=5 method_top_n_to_select=5 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
        ],
        "analyze_casp15": [
            "python3 posebench/analysis/inference_analysis_casp.py method=alphafold3 dataset=casp15 repeat_index={repeat_index} relax_protein={relax_protein}",
        ],
    },
    "vina": {
        "prepare_input": [
            "cp forks/DiffDock/inference/diffdock_{dataset}_inputs.csv forks/Vina/inference/vina_{dataset}_inputs.csv",
        ],
        "run_inference": [
            "python3 posebench/models/vina_inference.py dataset={dataset} method={vina_binding_site_method} pocket_only_baseline={pocket_only_baseline} repeat_index={repeat_index}",
        ],
        "copy_predictions": [
            "mkdir -p forks/Vina/inference/vina_{vina_binding_site_method}_{dataset}_outputs_{repeat_index} && cp -r data/test_cases/{dataset}/vina_{vina_binding_site_method}_{dataset}_outputs_{repeat_index}/* forks/Vina/inference/vina_{vina_binding_site_method}_{dataset}_outputs_{repeat_index}",
        ],
        "relax": [
            "python3 posebench/models/inference_relaxation.py method=vina vina_binding_site_method={vina_binding_site_method} dataset={dataset} cuda_device_index={cuda_device_index} pocket_only_baseline={pocket_only_baseline} relax_protein={relax_protein} remove_initial_protein_hydrogens=true assign_partial_charges_manually=true repeat_index={repeat_index}",
        ],
        "analyze_results": [
            "python3 posebench/analysis/inference_analysis.py method=vina vina_binding_site_method={vina_binding_site_method} dataset={dataset} pocket_only_baseline={pocket_only_baseline} relax_protein={relax_protein} repeat_index={repeat_index}",
        ],
        "assemble_casp15": [
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[vina] ensemble_ranking_method={ensemble_ranking_method} vina_binding_site_methods=[{vina_binding_site_method}] input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_vina_{vina_binding_site_method}_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=false relax_protein={relax_protein} export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=5 method_top_n_to_select=5 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset={dataset} cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[vina] ensemble_ranking_method={ensemble_ranking_method} vina_binding_site_methods=[{vina_binding_site_method}] input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_vina_{vina_binding_site_method}_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=true relax_protein={relax_protein} export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=5 method_top_n_to_select=5 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset={dataset} cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
        ],
        "analyze_casp15": [
            "python3 posebench/analysis/inference_analysis_casp.py method=vina vina_binding_site_method={vina_binding_site_method} dataset=casp15 relax_protein={relax_protein} repeat_index={repeat_index}",
        ],
    },
    "tulip": {
        "prepare_input": [
            "python3 posebench/data/tulip_output_extraction.py dataset={dataset}",
        ],
        "relax": [
            "python3 posebench/models/inference_relaxation.py method=tulip dataset={dataset} cuda_device_index={cuda_device_index} relax_protein={relax_protein} remove_initial_protein_hydrogens=true assign_partial_charges_manually=true repeat_index={repeat_index}",
        ],
        "analyze_results": [
            "python3 posebench/analysis/inference_analysis.py method=tulip dataset={dataset} relax_protein={relax_protein} repeat_index={repeat_index}",
        ],
        "assemble_casp15": [
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[tulip] ensemble_ranking_method={ensemble_ranking_method} input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_tulip_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=false relax_protein={relax_protein} export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=5 method_top_n_to_select=5 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset={dataset} cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
            "python3 posebench/models/ensemble_generation.py ensemble_methods=[tulip] ensemble_ranking_method={ensemble_ranking_method} input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_tulip_ensemble_predictions_{repeat_index} skip_existing=true relax_method_ligands_post_ranking=true relax_protein={relax_protein} export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=5 method_top_n_to_select=5 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset={dataset} cuda_device_index={cuda_device_index} ensemble_benchmarking_repeat_index={repeat_index}",
        ],
        "analyze_casp15": [
            "python3 posebench/analysis/inference_analysis_casp.py method=tulip dataset=casp15 repeat_index={repeat_index} relax_protein={relax_protein}",
        ],
    },
    "ensemble": {
        "run_inference": [
            "python3 posebench/models/ensemble_generation.py pocket_only_baseline={pocket_only_baseline} ensemble_ranking_method={ensemble_ranking_method} input_csv_filepath=data/test_cases/{dataset}/ensemble_inputs.csv output_dir=data/test_cases/{dataset}/top_consensus_ensemble_predictions_{repeat_index} max_method_predictions=5 method_top_n_to_select=3 export_top_n={export_top_n} export_file_format={dataset} skip_existing=true relax_method_ligands_post_ranking=false relax_protein={relax_protein} resume=true cuda_device_index={cuda_device_index} ensemble_methods='[diffdock, dynamicbind, neuralplexer, rfaa]' vina_binding_site_methods=[{vina_binding_site_method}] ensemble_benchmarking=true ensemble_benchmarking_dataset={dataset} ensemble_ranking_method=consensus ensemble_benchmarking_repeat_index={repeat_index}",
            "python3 posebench/models/ensemble_generation.py pocket_only_baseline={pocket_only_baseline} ensemble_ranking_method={ensemble_ranking_method} input_csv_filepath=data/test_cases/{dataset}/ensemble_inputs.csv output_dir=data/test_cases/{dataset}/top_consensus_ensemble_predictions_{repeat_index} max_method_predictions=5 method_top_n_to_select=3 export_top_n={export_top_n} export_file_format={dataset} skip_existing=true relax_method_ligands_post_ranking=true relax_protein={relax_protein} resume=true cuda_device_index={cuda_device_index} ensemble_methods='[diffdock, dynamicbind, neuralplexer, rfaa]' vina_binding_site_methods=[{vina_binding_site_method}] ensemble_benchmarking=true ensemble_benchmarking_dataset={dataset} ensemble_ranking_method=consensus ensemble_benchmarking_repeat_index={repeat_index}",
        ],
        "analyze_results": [
            "python3 posebench/analysis/inference_analysis.py method=ensemble dataset={dataset} pocket_only_baseline={pocket_only_baseline} relax_protein={relax_protein} repeat_index={repeat_index}",
        ],
        "analyze_casp15": [
            "python3 posebench/analysis/inference_analysis_casp.py method=ensemble dataset=casp15 relax_protein={relax_protein} repeat_index={repeat_index}",
        ],
    },
}

INFERENCE_METHODS = Literal[
    "diffdock",
    "fabind",
    "dynamicbind",
    "neuralplexer",
    "flowdock",
    "rfaa",
    "chai-lab",
    "boltz",
    "alphafold3",
    "vina",
    "tulip",
    "ensemble",
]
VINA_BINDING_SITE_INFERENCE_METHODS = Literal["diffdock", "p2rank"]
INFERENCE_ENSEMBLE_RANKING_METHODS = Literal["consensus", "ff"]
INFERENCE_DATASETS = Literal["posebusters_benchmark", "astex_diverse", "dockgen", "casp15"]

NON_GENERATIVE_INFERENCE_METHODS = {"fabind", "rfaa", "tulip"}
POCKET_ONLY_COMPATIBLE_METHODS = {
    "diffdock",
    "fabind",
    "dynamicbind",
    "neuralplexer",
    "flowdock",
    "rfaa",
    "chai-lab",
    "boltz",
    "alphafold3",
    "vina",
    "ensemble",
}
INVALID_METHOD_DATASET_COMBINATIONS = {
    ("fabind", "casp15"),
}


@beartype
def build_inference_script(
    method: INFERENCE_METHODS,
    vina_binding_site_method: VINA_BINDING_SITE_INFERENCE_METHODS,
    ensemble_ranking_method: INFERENCE_ENSEMBLE_RANKING_METHODS,
    dataset: INFERENCE_DATASETS,
    repeat_index: int,
    cuda_device_index: int,
    output_script_dir: str,
    pocket_only_baseline: bool = False,
    v1_baseline: bool = False,
    no_ilcl: bool = False,
    relax_protein: bool = False,
    export_hpc_headers: bool = False,
    verbose: bool = False,
    gpu_partition: str = "chengji-lab-gpu",
    gpu_account: str = "chengji-lab",
    gpu_type: Literal["A100", "H100", ""] = "",
    cpu_memory_in_gb: int = 59,
    time_limit: str = "2-00:00:00",
):
    """Build an inference script according to user arguments.

    :param method: Inference method to use.
    :param vina_binding_site_method: Vina binding site method to use.
    :param ensemble_ranking_method: Ensemble ranking method to use.
    :param dataset: Dataset to use.
    :param repeat_index: Index of the repeat.
    :param cuda_device_index: Index of the CUDA device to use.
    :param output_script_dir: Output script directory.
    :param pocket_only_baseline: Whether to perform a pocket-only
        baseline for the PoseBusters Benchmark set.
    :param v1_baseline: Whether to perform a V1 baseline for DiffDock.
    :param no_ilcl: Whether to use model weights trained with an inter-
        ligand clash loss (ILCL) for the CASP15 set.
    :param relax_protein: Whether to relax the protein structure before
        scoring.
    :param export_hpc_headers: Whether to export HPC headers.
    :param verbose: Whether to print verbose (i.e., invalid
        configuration) output.
    :param gpu_partition: GPU partition to use.
    :param gpu_account: GPU account to use.
    :param gpu_type: GPU type to use.
    :param cpu_memory_in_gb: CPU memory in GB.
    :param time_limit: Time limit.
    """
    commands = COMMANDS.get(method)

    # Inform user of invalid function calls
    if not commands:
        raise ValueError(f"Unsupported method: {method}")

    if (method, dataset) in INVALID_METHOD_DATASET_COMBINATIONS:
        if verbose:
            logging.info(f"Method {method} does not support dataset {dataset}. Skipping.")
        return

    if method in NON_GENERATIVE_INFERENCE_METHODS and repeat_index > 1:
        if verbose:
            logging.info(
                f"Method {method} does not support multiple repeats. Skipping repeat_index {repeat_index}."
            )
        return

    if pocket_only_baseline and not (
        method in POCKET_ONLY_COMPATIBLE_METHODS and dataset == "posebusters_benchmark"
    ):
        if verbose:
            logging.info(
                f"Method-dataset combination {method}-{dataset} does not support argument `pocket_only_baseline`. Skipping."
            )
        return

    if v1_baseline and not (method == "diffdock"):
        if verbose:
            logging.info(
                f"Method-dataset combination {method}-{dataset} does not support argument `v1_baseline`. Skipping."
            )
        return

    if no_ilcl and not (method == "neuralplexer" and dataset == "casp15"):
        if verbose:
            logging.info(
                f"Method-dataset combination {method}-{dataset} does not support argument `no_ilcl`. Skipping."
            )
        return

    os.makedirs(output_script_dir, exist_ok=True)
    vina_binding_site_method_suffix = f"_{vina_binding_site_method}" if method == "vina" else ""
    ensemble_ranking_method_suffix = f"_{ensemble_ranking_method}" if method == "ensemble" else ""
    v1_baseline_suffix = "v1" if v1_baseline else ""
    pocket_only_suffix = "_pocket_only" if pocket_only_baseline else ""
    no_ilcl_suffix = "_no_ilcl" if no_ilcl else ""
    relax_protein_suffix = "_relax_protein" if relax_protein else ""
    hpc_suffix = "_hpc" if export_hpc_headers else ""
    output_script = os.path.join(
        output_script_dir,
        f"{method}{vina_binding_site_method_suffix}{ensemble_ranking_method_suffix}{v1_baseline_suffix}{pocket_only_suffix}{no_ilcl_suffix}{relax_protein_suffix}_{dataset}{hpc_suffix}_inference_{repeat_index}.sh",
    )

    # Build script in sections
    with open(output_script, "w") as f:
        f.write("#!/bin/bash -l\n\n")
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
            # NOTE: The following HPC environment activation command assumes the
            # `PoseBench` Conda environment was created using `--prefix PoseBench/`
            # to reduce storage usage in one's HPC home directory
            f.write("\nconda activate PoseBench/\n\n")

            # NOTE: Model weights may take up too much space in one's HPC
            # home directory, so we recommend using a command like the following
            # to store the model weights in a larger storage location (e.g., `/scratch`):
            f.write(
                "# Store model weights in a larger storage location\n"
                + 'export TORCH_HOME="/cluster/pixstor/chengji-lab/$USER/torch_cache"\n'
                + 'export HF_HOME="/cluster/pixstor/chengji-lab/$USER/hf_cache"\n\n'
                + 'mkdir -p "$TORCH_HOME"\n'
                + 'mkdir -p "$HF_HOME"\n\n'
            )
        else:
            f.write(
                "# shellcheck source=/dev/null\n"
                + "source /home/$USER/mambaforge/etc/profile.d/conda.sh\n\n"
                + "# Activate PoseBench environment\n"
                + "conda activate PoseBench\n\n"
            )

        # Prepare input files
        if "prepare_input" in commands:
            diffdock_casp15_input_suffix = (
                " input_data_dir=data/casp15_set/targets input_protein_structure_dir=data/casp15_set/casp15_holo_aligned_predicted_structures"
                if method == "diffdock" and dataset == "casp15"
                else ""
            )
            dynamicbind_casp15_input_suffix = (
                " input_data_dir=data/casp15_set/targets"
                if method == "dynamicbind" and dataset == "casp15"
                else ""
            )
            neuralplexer_and_flowdock_casp15_input_suffix = (
                " input_data_dir=data/casp15_set/targets input_receptor_structure_dir=data/casp15_set/casp15_holo_aligned_predicted_structures"
                if method in ["neuralplexer", "flowdock"] and dataset == "casp15"
                else ""
            )
            rfaa_casp15_input_suffix = (
                " input_data_dir=data/casp15_set/targets"
                if method == "rfaa" and dataset == "casp15"
                else ""
            )
            chai_casp15_input_suffix = (
                " input_data_dir=data/casp15_set/targets"
                if method == "chai-lab" and dataset == "casp15"
                else ""
            )
            boltz_casp15_input_suffix = (
                " input_data_dir=data/casp15_set/targets"
                if method == "boltz" and dataset == "casp15"
                else ""
            )
            alphafold3_casp15_input_suffix = (
                " input_data_dir=data/casp15_set/targets"
                if method == "alphafold3" and dataset == "casp15"
                else ""
            )
            f.write("# Prepare input files\n")
            for cmd in commands.get("prepare_input", []):
                prepare_input_string = (
                    cmd.format(dataset=dataset, pocket_only_baseline=pocket_only_baseline)
                    + diffdock_casp15_input_suffix
                    + dynamicbind_casp15_input_suffix
                    + neuralplexer_and_flowdock_casp15_input_suffix
                    + rfaa_casp15_input_suffix
                    + chai_casp15_input_suffix
                    + boltz_casp15_input_suffix
                    + alphafold3_casp15_input_suffix
                    + "\n"
                )
                if method == "vina" and pocket_only_baseline:
                    prepare_input_string = prepare_input_string.replace(
                        f"diffdock_{dataset}", f"diffdock_pocket_only_{dataset}"
                    )
                    prepare_input_string = prepare_input_string.replace(
                        f"vina_{dataset}", f"vina_pocket_only_{dataset}"
                    )
                f.write(prepare_input_string)
            f.write("\n")

        # Run inference
        if "run_inference" in commands:
            export_top_n = 5 if dataset == "casp15" else 1
            diffdock_casp15_inference_suffix = (
                " batch_size=1" if method == "diffdock" and dataset == "casp15" else ""
            )
            dynamicbind_casp15_inference_suffix = (
                " batch_size=1 input_data_dir=data/casp15_set/casp15_holo_aligned_predicted_structures"
                if method == "dynamicbind" and dataset == "casp15"
                else ""
            )
            neuralplexer_and_flowdock_casp15_inference_suffix = (
                " chunk_size=5"
                if method in ["neuralplexer", "flowdock"] and dataset == "casp15"
                else ""
            )
            ensemble_casp15_inference_suffix = (
                " combine_casp_output_files=true"
                if method == "ensemble" and dataset == "casp15"
                else ""
            )
            f.write("# Run inference\n")
            for cmd in commands.get("run_inference", []):
                f.write(
                    cmd.format(
                        dataset=dataset,
                        repeat_index=repeat_index,
                        cuda_device_index=cuda_device_index,
                        vina_binding_site_method=vina_binding_site_method,
                        ensemble_ranking_method=ensemble_ranking_method,
                        export_top_n=export_top_n,
                        pocket_only_baseline=pocket_only_baseline,
                        v1_baseline=v1_baseline,
                        no_ilcl=no_ilcl,
                        relax_protein=relax_protein,
                    )
                    + diffdock_casp15_inference_suffix
                    + dynamicbind_casp15_inference_suffix
                    + neuralplexer_and_flowdock_casp15_inference_suffix
                    + ensemble_casp15_inference_suffix
                    + "\n"
                )
            if diffdock_casp15_inference_suffix:
                f.write(
                    "# NOTE: Due to DiffDock-L's occasional numerical instabilities "
                    + "on the CASP15 dataset, you may have to re-run this inference script "
                    + "several times (with the default 'skip_existing=true') to have it "
                    + "successfully predict for all CASP targets.\n"
                )
                f.write(
                    "# Consider running the following commands to clean up DiffDock-L's "
                    + "inference run directory (e.g., `_1`) before re-running this script:\n"
                )
                f.write(
                    "# rm -r forks/DiffDock/inference/diffdock_casp15_output_1/*_*/\n"
                    + "# find forks/DiffDock/inference/diffdock_casp15_output_1/* -type d ! -exec test -e {}/rank1.sdf \\; -exec sh -c 'rm -rf {}/' \\;\n"
                )
            if dynamicbind_casp15_inference_suffix:
                f.write(
                    "# NOTE: Due to DynamicBind's occasional numerical instabilities "
                    + "on the CASP15 dataset, you may have to re-run this inference script "
                    + "several times (with the default 'skip_existing=true') to have it "
                    + "successfully predict for all CASP targets.\n"
                )
                f.write(
                    "# Consider running the following commands to clean up DiffDock-L's "
                    + "inference run directory (e.g., `_1`) before re-running this script:\n"
                )
                f.write(
                    "# find forks/DynamicBind/inference/outputs/results/casp15__1/index0_idx_0 -type d ! -exec test -e {}/cleaned_input_proteinFile.pdb \\; -exec sh -c 'rm -rf $(dirname {})/' \\;\n"
                    + "# find forks/DynamicBind/inference/outputs/results/casp15_*_1/ -type d -empty -delete\n"
                )
            f.write("\n")

        # Extract outputs (if applicable)
        if "extract_outputs" in commands:
            f.write("# Extract outputs\n")
            for cmd in commands.get("extract_outputs", []):
                f.write(
                    cmd.format(
                        dataset=dataset,
                        pocket_only_baseline=pocket_only_baseline,
                        repeat_index=repeat_index,
                    )
                    + "\n"
                )
            f.write("\n")

        # Copy predictions (if applicable)
        if "copy_predictions" in commands:
            f.write("# Copy predictions\n")
            for cmd in commands.get("copy_predictions", []):
                copy_predictions_string = (
                    cmd.format(
                        dataset=dataset,
                        repeat_index=repeat_index,
                        vina_binding_site_method=vina_binding_site_method,
                    )
                    + "\n"
                )
                if method == "vina" and pocket_only_baseline:
                    copy_predictions_string = copy_predictions_string.replace(
                        f"vina_{vina_binding_site_method}",
                        f"vina_pocket_only_{vina_binding_site_method}",
                    )
                f.write(copy_predictions_string)
            f.write("\n")

        # Relax generated ligand structures (if applicable)
        if dataset != "casp15" and "relax" in commands:
            # NOTE: CASP15 predictions are instead relaxed using the `ensemble_generation.py` script
            f.write("# Relax generated ligand structures\n")
            for cmd in commands.get("relax", []):
                f.write(
                    cmd.format(
                        dataset=dataset,
                        repeat_index=repeat_index,
                        cuda_device_index=cuda_device_index,
                        vina_binding_site_method=vina_binding_site_method,
                        pocket_only_baseline=pocket_only_baseline,
                        v1_baseline=v1_baseline,
                        relax_protein=relax_protein,
                    )
                    + "\n"
                )
            f.write("\n")

        # Align complexes (if applicable)
        if dataset != "casp15" and "align_complexes" in commands:
            # NOTE: CASP15 predictions are instead aligned using the `ensemble_generation.py` script
            f.write("# Align complexes\n")
            for cmd in commands.get("align_complexes", []):
                f.write(
                    cmd.format(
                        dataset=dataset,
                        repeat_index=repeat_index,
                        pocket_only_baseline=pocket_only_baseline,
                    )
                    + "\n"
                )
            f.write("\n")

        # Analyze inference results (if applicable)
        if dataset != "casp15" and "analyze_results" in commands:
            # NOTE: CASP15 predictions are instead analyzed using the `inference_analysis_casp.py` script
            f.write("# Analyze inference results\n")
            for cmd in commands.get("analyze_results", []):
                f.write(
                    cmd.format(
                        dataset=dataset,
                        repeat_index=repeat_index,
                        vina_binding_site_method=vina_binding_site_method,
                        pocket_only_baseline=pocket_only_baseline,
                        v1_baseline=v1_baseline,
                        relax_protein=relax_protein,
                    )
                    + "\n"
                )
            f.write("\n")

        # Assemble CASP15 predictions (if applicable)
        if dataset == "casp15" and "assemble_casp15" in commands:
            f.write("# Assemble CASP15 results\n")
            for cmd in commands.get("assemble_casp15", []):
                f.write(
                    cmd.format(
                        dataset=dataset,
                        repeat_index=repeat_index,
                        cuda_device_index=cuda_device_index,
                        vina_binding_site_method=vina_binding_site_method,
                        ensemble_ranking_method=ensemble_ranking_method,
                        v1_baseline=v1_baseline,
                        no_ilcl=no_ilcl,
                        relax_protein=relax_protein,
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
                        v1_baseline=v1_baseline,
                        no_ilcl=no_ilcl,
                        relax_protein=relax_protein,
                    )
                    + "\n"
                )
            f.write("\n")

        # Inform user of run completion
        f.write("# Inform user of run completion\n" + f"echo 'Run {repeat_index} completed.'\n")

    logging.info(f"Script {output_script} created successfully.")


@beartype
def build_inference_scripts(
    methods_to_sweep: List[INFERENCE_METHODS],
    vina_binding_site_methods_to_sweep: List[VINA_BINDING_SITE_INFERENCE_METHODS],
    ensemble_ranking_methods_to_sweep: List[INFERENCE_ENSEMBLE_RANKING_METHODS],
    datasets_to_sweep: List[INFERENCE_DATASETS],
    num_sweep_repeats: int,
    cuda_device_index: int,
    output_script_dir: str,
    pocket_only_baseline: Optional[bool] = None,
    v1_baseline: Optional[bool] = None,
    no_ilcl: Optional[bool] = None,
    relax_protein: Optional[bool] = None,
    export_hpc_headers: bool = False,
    verbose: bool = False,
):
    """Build inference scripts according to user sweep arguments.

    :param methods_to_sweep: Inference methods to sweep.
    :param vina_binding_site_methods_to_sweep: Vina binding site methods
        to sweep.
    :param ensemble_ranking_methods_to_sweep: Ensemble ranking methods
        to sweep.
    :param datasets_to_sweep: Datasets to sweep.
    :param num_sweep_repeats: Number of repeats in the sweep.
    :param cuda_device_index: Index of the CUDA device to use.
    :param output_script: Output script file.
    :param pocket_only_baseline: Whether to perform a pocket-only
        baseline for the PoseBusters Benchmark set.
    :param v1_baseline: Whether to perform a V1 baseline for DiffDock.
    :param no_ilcl: Whether to use model weights trained with an inter-
        ligand clash loss (ILCL) for the CASP15 set.
    :param relax_protein: Whether to relax the protein structure before
        scoring.
    :param export_hpc_headers: Whether to export HPC headers.
    :param verbose: Whether to print verbose (i.e., invalid
        configuration) output.
    """
    for method in methods_to_sweep:
        for vina_binding_site_method in vina_binding_site_methods_to_sweep:
            for ensemble_ranking_method in ensemble_ranking_methods_to_sweep:
                for dataset in datasets_to_sweep:
                    for pocket_only in [True, False]:
                        pocket_only_mode = (
                            pocket_only_baseline
                            if pocket_only_baseline is not None
                            else pocket_only
                        )
                        for v1 in [True, False]:
                            v1_mode = v1_baseline if v1_baseline is not None else v1
                            for n_ilcl in [True, False]:
                                no_ilcl_mode = no_ilcl if no_ilcl is not None else n_ilcl
                                for relax_prot in [True, False]:
                                    relax_protein_mode = (
                                        relax_protein if relax_protein is not None else relax_prot
                                    )
                                    for repeat_index in range(1, num_sweep_repeats + 1):
                                        build_inference_script(
                                            method=method,
                                            vina_binding_site_method=vina_binding_site_method,
                                            ensemble_ranking_method=ensemble_ranking_method,
                                            dataset=dataset,
                                            repeat_index=repeat_index,
                                            cuda_device_index=cuda_device_index,
                                            output_script_dir=output_script_dir,
                                            pocket_only_baseline=pocket_only_mode,
                                            v1_baseline=v1_mode,
                                            no_ilcl=no_ilcl_mode,
                                            relax_protein=relax_protein_mode,
                                            export_hpc_headers=export_hpc_headers,
                                            verbose=verbose,
                                        )


@hydra.main(
    version_base="1.3",
    config_path="../configs/scripts",
    config_name="build_inference_script.yaml",
)
def main(cfg: DictConfig):
    """Build an inference script or sweep according to user arguments."""
    if cfg.sweep:
        build_inference_scripts(
            methods_to_sweep=list(cfg.methods_to_sweep),
            vina_binding_site_methods_to_sweep=list(cfg.vina_binding_site_methods_to_sweep),
            ensemble_ranking_methods_to_sweep=list(cfg.ensemble_ranking_methods_to_sweep),
            datasets_to_sweep=list(cfg.datasets_to_sweep),
            num_sweep_repeats=cfg.num_sweep_repeats,
            cuda_device_index=cfg.cuda_device_index,
            output_script_dir=cfg.output_script_dir,
            pocket_only_baseline=cfg.pocket_only_baseline,
            v1_baseline=cfg.v1_baseline,
            no_ilcl=cfg.no_ilcl,
            relax_protein=cfg.relax_protein,
            export_hpc_headers=cfg.export_hpc_headers,
            verbose=cfg.verbose,
        )
    else:
        build_inference_script(
            method=cfg.method,
            vina_binding_site_method=cfg.vina_binding_site_method,
            ensemble_ranking_method=cfg.ensemble_ranking_method,
            dataset=cfg.dataset,
            repeat_index=cfg.repeat_index,
            cuda_device_index=cfg.cuda_device_index,
            output_script_dir=cfg.output_script_dir,
            pocket_only_baseline=cfg.pocket_only_baseline,
            v1_baseline=cfg.v1_baseline,
            no_ilcl=cfg.no_ilcl,
            relax_protein=cfg.relax_protein,
            export_hpc_headers=cfg.export_hpc_headers,
            verbose=cfg.verbose,
        )


if __name__ == "__main__":
    main()
