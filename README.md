<div align="center">

# PoseBench

[![Paper](http://img.shields.io/badge/arXiv-2405.14108-B31B1B.svg)](https://arxiv.org/abs/2405.14108)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11477766.svg)](https://doi.org/10.5281/zenodo.11477766)
[![PyPI version](https://badge.fury.io/py/posebench.svg)](https://badge.fury.io/py/posebench)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Docs](https://assets.readthedocs.org/static/projects/badges/passing-flat.svg)](https://bioinfomachinelearning.github.io/PoseBench/)
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/config-hydra-89b8cd"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<img src="./img/PoseBench.png" width="600">

</div>

## Description

Comprehensive benchmarking of protein-ligand structure generation methods

[Documentation](https://bioinfomachinelearning.github.io/PoseBench/)

## Contents

- [Installation](#installation)
- [Tutorials](#tutorials)
- [How to prepare PoseBench data](#how-to-prepare-posebench-data)
- [Available inference methods](#available-inference-methods)
- [How to run inference with individual methods](#how-to-run-inference-with-individual-methods)
- [How to run inference with a method ensemble](#how-to-run-inference-with-a-method-ensemble)
- [How to create comparative plots of inference results](#how-to-create-comparative-plots-of-inference-results)
- [For developers](#for-developers)
- [Acknowledgements](#acknowledgements)
- [Citing this work](#citing-this-work)
- [Bonus](#bonus)

## Installation

<details>

### Portable installation

To reuse modules and utilities within `PoseBench` in other projects, one can simply use `pip`

```bash
pip install posebench
```

### Full installation

To reproduce, customize, or extend the `PoseBench` benchmark, we recommend fully installing `PoseBench` using `mamba` as follows:

First, install `mamba` for dependency management (as a fast alternative to Anaconda)

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh  # accept all terms and install to the default location
rm Mambaforge-$(uname)-$(uname -m).sh  # (optionally) remove installer after using it
source ~/.bashrc  # alternatively, one can restart their shell session to achieve the same result
```

Install dependencies for each method's environment (as desired)

```bash
# clone project
sudo apt-get install git-lfs  # NOTE: run this if you have not already installed `git-lfs`
git lfs install
git clone https://github.com/BioinfoMachineLearning/PoseBench --recursive
cd PoseBench

# create conda environments (~80 GB total)
# - PoseBench environment # (~15 GB)
mamba env create -f environments/posebench_environment.yaml
conda activate PoseBench  # NOTE: one still needs to use `conda` to (de)activate environments
pip3 install -e .
# - casp15_ligand_scoring environment (~3 GB)
mamba env create -f environments/casp15_ligand_scoring_environment.yaml
conda activate casp15_ligand_scoring  # NOTE: one still needs to use `conda` to (de)activate environments
pip3 install -e .
# - DiffDock environment (~13 GB)
mamba env create -f environments/diffdock_environment.yaml --prefix forks/DiffDock/DiffDock/
conda activate forks/DiffDock/DiffDock/  # NOTE: one still needs to use `conda` to (de)activate environments
# - FABind environment (~6 GB)
mamba env create -f environments/fabind_environment.yaml --prefix forks/FABind/FABind/
conda activate forks/FABind/FABind/  # NOTE: one still needs to use `conda` to (de)activate environments
# - DynamicBind environment (~13 GB)
mamba env create -f environments/dynamicbind_environment.yaml --prefix forks/DynamicBind/DynamicBind/
conda activate forks/DynamicBind/DynamicBind/  # NOTE: one still needs to use `conda` to (de)activate environments
# - NeuralPLexer environment (~14 GB)
mamba env create -f environments/neuralplexer_environment.yaml --prefix forks/NeuralPLexer/NeuralPLexer/
conda activate forks/NeuralPLexer/NeuralPLexer/  # NOTE: one still needs to use `conda` to (de)activate environments
cd forks/NeuralPLexer/ && pip3 install -e . && cd ../../
# - RoseTTAFold-All-Atom environment (~14 GB) - NOTE: after running these commands, follow the installation instructions in `forks/RoseTTAFold-All-Atom/README.md` starting at Step 4 (with `forks/RoseTTAFold-All-Atom/` as the current working directory)
mamba env create -f environments/rfaa_environment.yaml --prefix forks/RoseTTAFold-All-Atom/RFAA/
conda activate forks/RoseTTAFold-All-Atom/RFAA/  # NOTE: one still needs to use `conda` to (de)activate environments
cd forks/RoseTTAFold-All-Atom/rf2aa/SE3Transformer/ && pip3 install --no-cache-dir -r requirements.txt && python3 setup.py install && cd ../../../../
# - AutoDock Vina Tools environment (~1 GB)
mamba env create -f environments/adfr_environment.yaml --prefix forks/Vina/ADFR/
conda activate forks/Vina/ADFR/  # NOTE: one still needs to use `conda` to (de)activate environments
# - P2Rank (~0.5 GB)
wget -P forks/P2Rank/ https://github.com/rdk/p2rank/releases/download/2.4.2/p2rank_2.4.2.tar.gz
tar -xzf forks/P2Rank/p2rank_2.4.2.tar.gz -C forks/P2Rank/
rm forks/P2Rank/p2rank_2.4.2.tar.gz
```

Download checkpoints (~8.25 GB total)

```bash
# DynamicBind checkpoint (~0.25 GB)
cd forks/DynamicBind/
wget https://zenodo.org/records/10137507/files/workdir.zip
unzip workdir.zip
rm workdir.zip
cd ../../

# NeuralPLexer checkpoint (~6.5 GB)
cd forks/NeuralPLexer/
wget https://zenodo.org/records/10373581/files/neuralplexermodels_downstream_datasets_predictions.zip
unzip neuralplexermodels_downstream_datasets_predictions.zip
rm neuralplexermodels_downstream_datasets_predictions.zip
cd ../../

# RoseTTAFold-All-Atom checkpoint (~1.5 GB)
cd forks/RoseTTAFold-All-Atom/
wget http://files.ipd.uw.edu/pub/RF-All-Atom/weights/RFAA_paper_weights.pt
cd ../../
```

</details>

## Tutorials

<details>

We provide a two-part tutorial series of Jupyter notebooks to provide users with examples
of how to extend `PoseBench`, as outlined below.

1. [Adding a new dataset](https://github.com/BioinfoMachineLearning/PoseBench/blob/main/notebooks/adding_new_dataset_tutorial.ipynb)
2. [Adding a new method](https://github.com/BioinfoMachineLearning/PoseBench/blob/main/notebooks/adding_new_method_tutorial.ipynb)

</details>

## How to prepare `PoseBench` data

<details>

### Downloading Astex, PoseBusters, DockGen, and CASP15 data

```bash
# fetch, extract, and clean-up preprocessed Astex Diverse, PoseBusters Benchmark, DockGen, and CASP15 data (~3 GB) #
wget https://zenodo.org/records/11477766/files/astex_diverse_set.tar.gz
wget https://zenodo.org/records/11477766/files/posebusters_benchmark_set.tar.gz
wget https://zenodo.org/records/11477766/files/dockgen_set.tar.gz
wget https://zenodo.org/records/11477766/files/casp15_set.tar.gz
tar -xzf astex_diverse_set.tar.gz
tar -xzf posebusters_benchmark_set.tar.gz
tar -xzf dockgen_set.tar.gz
tar -xzf casp15_set.tar.gz
rm astex_diverse_set.tar.gz
rm posebusters_benchmark_set.tar.gz
rm dockgen_set.tar.gz
rm casp15_set.tar.gz
```

### Downloading benchmark method predictions

```bash
# fetch, extract, and clean-up benchmark method predictions to reproduce paper results (~19 GB) #
# DiffDock predictions and results
wget https://zenodo.org/records/11477766/files/diffdock_benchmark_method_predictions.tar.gz
tar -xzf diffdock_benchmark_method_predictions.tar.gz
rm diffdock_benchmark_method_predictions.tar.gz
# FABind predictions and results
wget https://zenodo.org/records/11477766/files/fabind_benchmark_method_predictions.tar.gz
tar -xzf fabind_benchmark_method_predictions.tar.gz
rm fabind_benchmark_method_predictions.tar.gz
# DynamicBind predictions and results
wget https://zenodo.org/records/11477766/files/dynamicbind_benchmark_method_predictions.tar.gz
tar -xzf dynamicbind_benchmark_method_predictions.tar.gz
rm dynamicbind_benchmark_method_predictions.tar.gz
# NeuralPLexer predictions and results
wget https://zenodo.org/records/11477766/files/neuralplexer_benchmark_method_predictions.tar.gz
tar -xzf neuralplexer_benchmark_method_predictions.tar.gz
rm neuralplexer_benchmark_method_predictions.tar.gz
# RoseTTAFold-All-Atom predictions and results
wget https://zenodo.org/records/11477766/files/rfaa_benchmark_method_predictions.tar.gz
tar -xzf rfaa_benchmark_method_predictions.tar.gz
rm rfaa_benchmark_method_predictions.tar.gz
# TULIP predictions and results
wget https://zenodo.org/records/11477766/files/tulip_benchmark_method_predictions.tar.gz
tar -xzf tulip_benchmark_method_predictions.tar.gz
rm tulip_benchmark_method_predictions.tar.gz
# AutoDock Vina predictions and results
wget https://zenodo.org/records/11477766/files/vina_benchmark_method_predictions.tar.gz
tar -xzf vina_benchmark_method_predictions.tar.gz
rm vina_benchmark_method_predictions.tar.gz
# Astex Diverse, PoseBusters Benchmark (w/ pocket-only results), DockGen, and CASP15 consensus ensemble predictions and results
wget https://zenodo.org/records/11477766/files/astex_diverse_ensemble_benchmark_method_predictions.tar.gz
wget https://zenodo.org/records/11477766/files/posebusters_benchmark_ensemble_benchmark_method_predictions.tar.gz
wget https://zenodo.org/records/11477766/files/dockgen_ensemble_benchmark_method_predictions.tar.gz
wget https://zenodo.org/records/11477766/files/casp15_ensemble_benchmark_method_predictions.tar.gz
tar -xzf astex_diverse_ensemble_benchmark_method_predictions.tar.gz
tar -xzf posebusters_benchmark_ensemble_benchmark_method_predictions.tar.gz
tar -xzf dockgen_ensemble_benchmark_method_predictions.tar.gz
tar -xzf casp15_ensemble_benchmark_method_predictions.tar.gz
rm astex_diverse_ensemble_benchmark_method_predictions.tar.gz
rm posebusters_benchmark_ensemble_benchmark_method_predictions.tar.gz
rm dockgen_ensemble_benchmark_method_predictions.tar.gz
rm casp15_ensemble_benchmark_method_predictions.tar.gz
```

**NOTE:** One can reproduce the *pocket-only* experiments with the PoseBusters Benchmark set by adding the argument `pocket_only_baseline=true` to each command below used to run PoseBusters Benchmark dataset inference with all the baseline methods, since the pocket-only versions of the dataset's holo-aligned predicted protein structures have also been included in the downloadable Zenodo archive `posebusters_benchmark_set.tar.gz` referenced above. However, be aware that one then needs to *rename* any existing directories containing PoseBusters Benchmark dataset inference results for each baseline method, to prevent these existing inference directories from being merged with new pocket-only results. Please see the config files within `configs/data/`, `configs/model/`, and `configs/analysis/` for more details.

### Downloading sequence databases (required only for RoseTTAFold-All-Atom inference)

```bash
# acquire multiple sequence alignment databases for RoseTTAFold-All-Atom (~2.5 TB)
cd forks/RoseTTAFold-All-Atom/

# uniref30 [46G]
wget http://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz
mkdir -p UniRef30_2020_06
tar xfz UniRef30_2020_06_hhsuite.tar.gz -C ./UniRef30_2020_06

# BFD [272G]
wget https://bfd.mmseqs.com/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz
mkdir -p bfd
tar xfz bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz -C ./bfd

# structure templates (including *_a3m.ffdata, *_a3m.ffindex)
wget https://files.ipd.uw.edu/pub/RoseTTAFold/pdb100_2021Mar03.tar.gz
tar xfz pdb100_2021Mar03.tar.gz

cd ../../
```

### Predicting apo protein structures using ESMFold

First create all the corresponding FASTA files for each protein sequence

```bash
python3 posebench/data/components/esmfold_fasta_preparation.py dataset=posebusters_benchmark
python3 posebench/data/components/esmfold_fasta_preparation.py dataset=astex_diverse
```

To generate the apo version of each protein structure,
create ESMFold-ready versions of the combined FASTA files
prepared above by the script `esmfold_fasta_preparation.py`
for the PoseBusters Benchmark and Astex Diverse sets, respectively

```bash
python3 posebench/data/components/esmfold_sequence_preparation.py dataset=posebusters_benchmark
python3 posebench/data/components/esmfold_sequence_preparation.py dataset=astex_diverse
```

Then, predict each apo protein structure using ESMFold's batch
inference script

```bash
python3 posebench/data/components/esmfold_batch_structure_prediction.py -i data/posebusters_benchmark_set/posebusters_benchmark_esmfold_sequences.fasta -o data/posebusters_benchmark_set/posebusters_benchmark_esmfold_structures --skip-existing
python3 posebench/data/components/esmfold_batch_structure_prediction.py -i data/astex_diverse_set/astex_diverse_esmfold_sequences.fasta -o data/astex_diverse_set/astex_diverse_esmfold_structures --skip-existing
```

**NOTE:** Having a CUDA-enabled device available when running ESMFold is highly recommended

**NOTE:** ESMFold may not be able to predict apo protein structures for a handful of exceedingly-long (e.g., >2000 token) input sequences

Lastly, align each apo protein structure to its corresponding
holo protein structure counterpart in the PoseBusters Benchmark
or Astex Diverse set, taking ligand conformations into account
during each alignment

```bash
python3 posebench/data/components/esmfold_apo_to_holo_alignment.py dataset=posebusters_benchmark num_workers=1
python3 posebench/data/components/esmfold_apo_to_holo_alignment.py dataset=astex_diverse num_workers=1
```

**NOTE:** The preprocessed Astex Diverse, PoseBusters Benchmark, DockGen, and CASP15 data available via [Zenodo](https://doi.org/10.5281/zenodo.11477766) provide pre-holo-aligned predicted protein structures for these respective datasets.

</details>

## Available inference methods

<details>

### Methods available individually

#### Fixed Protein Methods

| Name            | Source                                                                | Astex Benchmarked | PoseBusters Benchmarked | DockGen Benchmarked | CASP Benchmarked |
| --------------- | --------------------------------------------------------------------- | ----------------- | ----------------------- | ------------------- | ---------------- |
| `DiffDock`      | [Corso et al.](https://openreview.net/forum?id=UfBIxpTK10)            | ✓                 | ✓                       | ✓                   | ✓                |
| `FABind`        | [Pei et al.](https://openreview.net/forum?id=PnWakgg1RL)              | ✓                 | ✓                       | ✓                   | ✗                |
| `AutoDock Vina` | [Eberhardt et al.](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00203) | ✓                 | ✓                       | ✓                   | ✓                |
| `TULIP`         |                                                                       | ✓                 | ✓                       | ✗                   | ✓                |

#### Flexible Protein Methods

| Name                   | Source                                                                | Astex Benchmarked | PoseBusters Benchmarked | DockGen Benchmarked | CASP Benchmarked |
| ---------------------- | --------------------------------------------------------------------- | ----------------- | ----------------------- | ------------------- | ---------------- |
| `DynamicBind`          | [Lu et al.](https://www.nature.com/articles/s41467-024-45461-2)       | ✓                 | ✓                       | ✓                   | ✓                |
| `NeuralPLexer`         | [Qiao et al.](https://www.nature.com/articles/s42256-024-00792-z)     | ✓                 | ✓                       | ✓                   | ✓                |
| `RoseTTAFold-All-Atom` | [Krishna et al.](https://www.science.org/doi/10.1126/science.adl2528) | ✓                 | ✓                       | ✓                   | ✓                |

### Methods available for ensembling

#### Fixed Protein Methods

| Name            | Source                                                                | Astex Benchmarked | PoseBusters Benchmarked | DockGen Benchmarked | CASP Benchmarked |
| --------------- | --------------------------------------------------------------------- | ----------------- | ----------------------- | ------------------- | ---------------- |
| `DiffDock`      | [Corso et al.](https://openreview.net/forum?id=UfBIxpTK10)            | ✓                 | ✓                       | ✓                   | ✓                |
| `AutoDock Vina` | [Eberhardt et al.](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00203) | ✓                 | ✓                       | ✓                   | ✓                |
| `TULIP`         |                                                                       | ✓                 | ✓                       | ✗                   | ✓                |

#### Flexible Protein Methods

| Name                   | Source                                                                | Astex Benchmarked | PoseBusters Benchmarked | DockGen Benchmarked | CASP Benchmarked |
| ---------------------- | --------------------------------------------------------------------- | ----------------- | ----------------------- | ------------------- | ---------------- |
| `DynamicBind`          | [Lu et al.](https://www.nature.com/articles/s41467-024-45461-2)       | ✓                 | ✓                       | ✓                   | ✓                |
| `NeuralPLexer`         | [Qiao et al.](https://www.nature.com/articles/s42256-024-00792-z)     | ✓                 | ✓                       | ✓                   | ✓                |
| `RoseTTAFold-All-Atom` | [Krishna et al.](https://www.science.org/doi/10.1126/science.adl2528) | ✓                 | ✓                       | ✓                   | ✓                |

**NOTE**: Have a new method to add? Please let us know by creating a pull request. We would be happy to work with you to integrate new methodology into this benchmark!

</details>

## How to run inference with individual methods

<details>

### How to run inference with `DiffDock`

Prepare CSV input files

```bash
python3 posebench/data/diffdock_input_preparation.py dataset=posebusters_benchmark
python3 posebench/data/diffdock_input_preparation.py dataset=astex_diverse
python3 posebench/data/diffdock_input_preparation.py dataset=dockgen
python3 posebench/data/diffdock_input_preparation.py dataset=casp15 input_data_dir="$PWD"/data/casp15_set/targets input_protein_structure_dir="$PWD"/data/casp15_set/predicted_structures
```

Run inference on each dataset

```bash
python3 posebench/models/diffdock_inference.py dataset=posebusters_benchmark repeat_index=1
...
python3 posebench/models/diffdock_inference.py dataset=astex_diverse repeat_index=1
...
python3 posebench/models/diffdock_inference.py dataset=dockgen repeat_index=1
...
python3 posebench/models/diffdock_inference.py dataset=casp15 batch_size=1 repeat_index=1
...
```

Relax the generated ligand structures inside of their respective protein pockets

```bash
python3 posebench/models/inference_relaxation.py method=diffdock dataset=posebusters_benchmark remove_initial_protein_hydrogens=true assign_partial_charges_manually=true num_processes=1 repeat_index=1
...
python3 posebench/models/inference_relaxation.py method=diffdock dataset=astex_diverse remove_initial_protein_hydrogens=true assign_partial_charges_manually=true num_processes=1 repeat_index=1
...
python3 posebench/models/inference_relaxation.py method=diffdock dataset=dockgen remove_initial_protein_hydrogens=true assign_partial_charges_manually=true num_processes=1 repeat_index=1
...
```

**NOTE**: Increase `num_processes` according to your available CPU/GPU resources to improve throughput

Analyze inference results for each dataset

```bash
python3 posebench/analysis/inference_analysis.py method=diffdock dataset=posebusters_benchmark repeat_index=1
...
python3 posebench/analysis/inference_analysis.py method=diffdock dataset=astex_diverse repeat_index=1
...
python3 posebench/analysis/inference_analysis.py method=diffdock dataset=dockgen repeat_index=1
...
```

Analyze inference results for the CASP15 dataset

```bash
# first assemble (unrelaxed and post ranking-relaxed) CASP15-compliant prediction submission files for scoring
python3 posebench/models/ensemble_generation.py ensemble_methods=\[diffdock\] input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_diffdock_ensemble_predictions_1 skip_existing=true relax_method_ligands_post_ranking=false export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=40 method_top_n_to_select=40 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index=0 ensemble_benchmarking_repeat_index=1
python3 posebench/models/ensemble_generation.py ensemble_methods=\[diffdock\] input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_diffdock_ensemble_predictions_1 skip_existing=true relax_method_ligands_post_ranking=true export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=40 method_top_n_to_select=40 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index=0 ensemble_benchmarking_repeat_index=1
# NOTE: the suffixes for both `output_dir` and `ensemble_benchmarking_repeat_index` should be modified to e.g., 2, 3, ...
...
# now score the CASP15-compliant submissions using the official CASP scoring pipeline
python3 posebench/analysis/inference_analysis_casp.py method=diffdock dataset=casp15 repeat_index=1
...
```

### How to run inference with `FABind`

Prepare CSV input files

```bash
python3 posebench/data/fabind_input_preparation.py dataset=posebusters_benchmark
python3 posebench/data/fabind_input_preparation.py dataset=astex_diverse
python3 posebench/data/fabind_input_preparation.py dataset=dockgen
```

Run inference on each dataset

```bash
python3 posebench/models/fabind_inference.py dataset=posebusters_benchmark repeat_index=1
...
python3 posebench/models/fabind_inference.py dataset=astex_diverse repeat_index=1
...
python3 posebench/models/fabind_inference.py dataset=dockgen repeat_index=1
...
```

Relax the generated ligand structures inside of their respective protein pockets

```bash
python3 posebench/models/inference_relaxation.py method=fabind dataset=posebusters_benchmark remove_initial_protein_hydrogens=true assign_partial_charges_manually=true num_processes=1 repeat_index=1
...
python3 posebench/models/inference_relaxation.py method=fabind dataset=astex_diverse remove_initial_protein_hydrogens=true assign_partial_charges_manually=true num_processes=1 repeat_index=1
...
python3 posebench/models/inference_relaxation.py method=fabind dataset=dockgen remove_initial_protein_hydrogens=true assign_partial_charges_manually=true num_processes=1 repeat_index=1
...
```

**NOTE**: Increase `num_processes` according to your available CPU/GPU resources to improve throughput

Analyze inference results for each dataset

```bash
python3 posebench/analysis/inference_analysis.py method=fabind dataset=posebusters_benchmark repeat_index=1
...
python3 posebench/analysis/inference_analysis.py method=fabind dataset=astex_diverse repeat_index=1
...
python3 posebench/analysis/inference_analysis.py method=fabind dataset=dockgen repeat_index=1
...
```

### How to run inference with `DynamicBind`

Prepare CSV input files

```bash
python3 posebench/data/dynamicbind_input_preparation.py dataset=posebusters_benchmark
python3 posebench/data/dynamicbind_input_preparation.py dataset=astex_diverse
python3 posebench/data/dynamicbind_input_preparation.py dataset=dockgen
python3 posebench/data/dynamicbind_input_preparation.py dataset=casp15 input_data_dir="$PWD"/data/casp15_set/targets
```

Run inference on each dataset

```bash
python3 posebench/models/dynamicbind_inference.py dataset=posebusters_benchmark repeat_index=1
...
python3 posebench/models/dynamicbind_inference.py dataset=astex_diverse repeat_index=1
...
python3 posebench/models/dynamicbind_inference.py dataset=dockgen repeat_index=1
...
python3 posebench/models/dynamicbind_inference.py dataset=casp15 batch_size=1 input_data_dir="$PWD"/data/casp15_set/predicted_structures repeat_index=1
...
```

Relax the generated ligand structures inside of their respective protein pockets

```bash
python3 posebench/models/inference_relaxation.py method=dynamicbind dataset=posebusters_benchmark remove_initial_protein_hydrogens=true assign_partial_charges_manually=true num_processes=1 repeat_index=1
...
python3 posebench/models/inference_relaxation.py method=dynamicbind dataset=astex_diverse remove_initial_protein_hydrogens=true assign_partial_charges_manually=true num_processes=1 repeat_index=1
...
python3 posebench/models/inference_relaxation.py method=dynamicbind dataset=dockgen remove_initial_protein_hydrogens=true assign_partial_charges_manually=true num_processes=1 repeat_index=1
...
```

**NOTE**: Increase `num_processes` according to your available CPU/GPU resources to improve throughput

Analyze inference results for each dataset

```bash
python3 posebench/analysis/inference_analysis.py method=dynamicbind dataset=posebusters_benchmark repeat_index=1
...
python3 posebench/analysis/inference_analysis.py method=dynamicbind dataset=astex_diverse repeat_index=1
...
python3 posebench/analysis/inference_analysis.py method=dynamicbind dataset=dockgen repeat_index=1
...
```

Analyze inference results for the CASP15 dataset

```bash
# first assemble (unrelaxed and post ranking-relaxed) CASP15-compliant prediction submission files for scoring
python3 posebench/models/ensemble_generation.py ensemble_methods=\[dynamicbind\] input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_dynamicbind_ensemble_predictions_1 skip_existing=true relax_method_ligands_post_ranking=false export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=40 method_top_n_to_select=40 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index=0 ensemble_benchmarking_repeat_index=1
python3 posebench/models/ensemble_generation.py ensemble_methods=\[dynamicbind\] input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_dynamicbind_ensemble_predictions_1 skip_existing=true relax_method_ligands_post_ranking=true export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=40 method_top_n_to_select=40 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index=0 ensemble_benchmarking_repeat_index=1
# NOTE: the suffixes for both `output_dir` and `ensemble_benchmarking_repeat_index` should be modified to e.g., 2, 3, ...
...
# now score the CASP15-compliant submissions using the official CASP scoring pipeline
python3 posebench/analysis/inference_analysis_casp.py method=dynamicbind dataset=casp15 repeat_index=1
...
```

### How to run inference with `NeuralPLexer`

Prepare CSV input files

```bash
python3 posebench/data/neuralplexer_input_preparation.py dataset=posebusters_benchmark
python3 posebench/data/neuralplexer_input_preparation.py dataset=astex_diverse
python3 posebench/data/neuralplexer_input_preparation.py dataset=dockgen
python3 posebench/data/neuralplexer_input_preparation.py dataset=casp15 input_data_dir="$PWD"/data/casp15_set/targets input_receptor_structure_dir="$PWD"/data/casp15_set/predicted_structures
```

Run inference on each dataset

```bash
python3 posebench/models/neuralplexer_inference.py dataset=posebusters_benchmark repeat_index=1
...
python3 posebench/models/neuralplexer_inference.py dataset=astex_diverse repeat_index=1
...
python3 posebench/models/neuralplexer_inference.py dataset=dockgen repeat_index=1
...
python3 posebench/models/neuralplexer_inference.py dataset=casp15 repeat_index=1
...
```

Relax the generated ligand structures inside of their respective protein pockets

```bash
python3 posebench/models/inference_relaxation.py method=neuralplexer dataset=posebusters_benchmark num_processes=1 remove_initial_protein_hydrogens=true assign_partial_charges_manually=true cache_files=false repeat_index=1
...
python3 posebench/models/inference_relaxation.py method=neuralplexer dataset=astex_diverse num_processes=1 remove_initial_protein_hydrogens=true assign_partial_charges_manually=true cache_files=false repeat_index=1
...
python3 posebench/models/inference_relaxation.py method=neuralplexer dataset=dockgen num_processes=1 remove_initial_protein_hydrogens=true assign_partial_charges_manually=true cache_files=false repeat_index=1
...
```

**NOTE**: Increase `num_processes` according to your available CPU/GPU resources to improve throughput

Align predicted protein-ligand structures to ground-truth complex structures

```bash
python3 posebench/analysis/complex_alignment.py method=neuralplexer dataset=posebusters_benchmark repeat_index=1
...
python3 posebench/analysis/complex_alignment.py method=neuralplexer dataset=astex_diverse repeat_index=1
...
python3 posebench/analysis/complex_alignment.py method=neuralplexer dataset=dockgen repeat_index=1
...
```

Analyze inference results for each dataset

```bash
python3 posebench/analysis/inference_analysis.py method=neuralplexer dataset=posebusters_benchmark repeat_index=1
...
python3 posebench/analysis/inference_analysis.py method=neuralplexer dataset=astex_diverse repeat_index=1
...
python3 posebench/analysis/inference_analysis.py method=neuralplexer dataset=dockgen repeat_index=1
...
```

Analyze inference results for the CASP15 dataset

```bash
# first assemble (unrelaxed and post ranking-relaxed) CASP15-compliant prediction submission files for scoring
python3 posebench/models/ensemble_generation.py ensemble_methods=\[neuralplexer\] input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_neuralplexer_ensemble_predictions_1 skip_existing=true relax_method_ligands_post_ranking=false export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=40 method_top_n_to_select=40 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index=0 ensemble_benchmarking_repeat_index=1
python3 posebench/models/ensemble_generation.py ensemble_methods=\[neuralplexer\] input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_neuralplexer_ensemble_predictions_1 skip_existing=true relax_method_ligands_post_ranking=true export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=40 method_top_n_to_select=40 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index=0 ensemble_benchmarking_repeat_index=1
# NOTE: the suffixes for both `output_dir` and `ensemble_benchmarking_repeat_index` should be modified to e.g., 2, 3, ...
...
# now score the CASP15-compliant submissions using the official CASP scoring pipeline
python3 posebench/analysis/inference_analysis_casp.py method=neuralplexer dataset=casp15 repeat_index=1
...
```

### How to run inference with `RoseTTAFold-All-Atom`

Prepare CSV input files

```bash
python3 posebench/data/rfaa_input_preparation.py dataset=posebusters_benchmark
python3 posebench/data/rfaa_input_preparation.py dataset=astex_diverse
python3 posebench/data/rfaa_input_preparation.py dataset=dockgen
python3 posebench/data/rfaa_input_preparation.py dataset=casp15 input_data_dir="$PWD"/data/casp15_set/targets
```

Run inference on each dataset

```bash
conda activate forks/RoseTTAFold-All-Atom/RFAA/
python3 posebench/models/rfaa_inference.py dataset=posebusters_benchmark run_inference_directly=true
python3 posebench/models/rfaa_inference.py dataset=astex_diverse run_inference_directly=true
python3 posebench/models/rfaa_inference.py dataset=dockgen run_inference_directly=true
python3 posebench/models/rfaa_inference.py dataset=casp15 run_inference_directly=true
conda deactivate
```

Extract predictions into separate files for proteins and ligands

```bash
python3 posebench/data/rfaa_output_extraction.py dataset=posebusters_benchmark
python3 posebench/data/rfaa_output_extraction.py dataset=astex_diverse
python3 posebench/data/rfaa_output_extraction.py dataset=dockgen
python3 posebench/data/rfaa_output_extraction.py dataset=casp15
```

Relax the generated ligand structures inside of their respective protein pockets

```bash
python3 posebench/models/inference_relaxation.py method=rfaa dataset=posebusters_benchmark num_processes=1 remove_initial_protein_hydrogens=true
python3 posebench/models/inference_relaxation.py method=rfaa dataset=astex_diverse num_processes=1 remove_initial_protein_hydrogens=true
python3 posebench/models/inference_relaxation.py method=rfaa dataset=dockgen num_processes=1 remove_initial_protein_hydrogens=true
```

**NOTE**: Increase `num_processes` according to your available CPU/GPU resources to improve throughput

Align predicted protein-ligand structures to ground-truth complex structures

```bash
python3 posebench/analysis/complex_alignment.py method=rfaa dataset=posebusters_benchmark
python3 posebench/analysis/complex_alignment.py method=rfaa dataset=astex_diverse
python3 posebench/analysis/complex_alignment.py method=rfaa dataset=dockgen
```

Analyze inference results for each dataset

```bash
python3 posebench/analysis/inference_analysis.py method=rfaa dataset=posebusters_benchmark
python3 posebench/analysis/inference_analysis.py method=rfaa dataset=astex_diverse
python3 posebench/analysis/inference_analysis.py method=rfaa dataset=dockgen
```

Analyze inference results for the CASP15 dataset

```bash
# first assemble (unrelaxed and post ranking-relaxed) CASP15-compliant prediction submission files for scoring
python3 posebench/models/ensemble_generation.py ensemble_methods=\[rfaa\] input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_rfaa_ensemble_predictions_1 skip_existing=true relax_method_ligands_post_ranking=false export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=40 method_top_n_to_select=40 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index=0 ensemble_benchmarking_repeat_index=1
python3 posebench/models/ensemble_generation.py ensemble_methods=\[rfaa\] input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_rfaa_ensemble_predictions_1 skip_existing=true relax_method_ligands_post_ranking=true export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=40 method_top_n_to_select=40 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index=0 ensemble_benchmarking_repeat_index=1
# NOTE: the suffixes for both `output_dir` and `ensemble_benchmarking_repeat_index` should be modified to e.g., 2, 3, ...
...
# now score the CASP15-compliant submissions using the official CASP scoring pipeline
python3 posebench/analysis/inference_analysis_casp.py method=rfaa dataset=casp15 targets="[T1124, T1127v2, T1146, T1152, T1158v1, T1158v2, T1158v3, T1158v4, T1186, T1187, T1188]" repeat_index=1
...
```

### How to run inference with `AutoDock Vina`

Prepare CSV input files

```bash
cp forks/DiffDock/inference/diffdock_posebusters_benchmark_inputs.csv forks/Vina/inference/vina_posebusters_benchmark_inputs.csv
cp forks/DiffDock/inference/diffdock_astex_diverse_inputs.csv forks/Vina/inference/vina_astex_diverse_inputs.csv
cp forks/DiffDock/inference/diffdock_dockgen_inputs.csv forks/Vina/inference/vina_dockgen_inputs.csv
cp forks/DiffDock/inference/diffdock_casp15_inputs.csv forks/Vina/inference/vina_casp15_inputs.csv
```

Run inference on each dataset

```bash
python3 posebench/models/vina_inference.py dataset=posebusters_benchmark method=diffdock repeat_index=1 # NOTE: DiffDock-L's binding pockets are recommended as the default Vina input
...
python3 posebench/models/vina_inference.py dataset=astex_diverse method=diffdock repeat_index=1
...
python3 posebench/models/vina_inference.py dataset=dockgen method=diffdock repeat_index=1
...
python3 posebench/models/vina_inference.py dataset=casp15 method=diffdock repeat_index=1
...
```

Copy Vina's predictions to the corresponding inference directory for each repeat

```bash
mkdir -p forks/Vina/inference/vina_diffdock_posebusters_benchmark_outputs_1 && cp -r data/test_cases/posebusters_benchmark/vina_diffdock_posebusters_benchmark_outputs_1/* forks/Vina/inference/vina_diffdock_posebusters_benchmark_outputs_1
...
mkdir -p forks/Vina/inference/vina_diffdock_astex_diverse_outputs_1 && cp -r data/test_cases/astex_diverse/vina_diffdock_astex_diverse_outputs_1/* forks/Vina/inference/vina_diffdock_astex_diverse_outputs_1
...
mkdir -p forks/Vina/inference/vina_diffdock_dockgen_outputs_1 && cp -r data/test_cases/dockgen/vina_diffdock_dockgen_outputs_1/* forks/Vina/inference/vina_diffdock_dockgen_outputs_1
...
mkdir -p forks/Vina/inference/vina_diffdock_casp15_outputs_1 && cp -r data/test_cases/casp15/vina_diffdock_casp15_outputs_1/* forks/Vina/inference/vina_diffdock_casp15_outputs_1
...
```

Relax the generated ligand structures inside of their respective protein pockets

```bash
python3 posebench/models/inference_relaxation.py method=vina vina_binding_site_method=diffdock dataset=posebusters_benchmark remove_initial_protein_hydrogens=true assign_partial_charges_manually=true num_processes=1 repeat_index=1
...
python3 posebench/models/inference_relaxation.py method=vina vina_binding_site_method=diffdock dataset=astex_diverse remove_initial_protein_hydrogens=true assign_partial_charges_manually=true num_processes=1 repeat_index=1
...
python3 posebench/models/inference_relaxation.py method=vina vina_binding_site_method=diffdock dataset=dockgen remove_initial_protein_hydrogens=true assign_partial_charges_manually=true num_processes=1 repeat_index=1
...
```

**NOTE**: Increase `num_processes` according to your available CPU/GPU resources to improve throughput

Analyze inference results for each dataset

```bash
python3 posebench/analysis/inference_analysis.py method=vina vina_binding_site_method=diffdock dataset=posebusters_benchmark repeat_index=1
...
python3 posebench/analysis/inference_analysis.py method=vina vina_binding_site_method=diffdock dataset=astex_diverse repeat_index=1
...
python3 posebench/analysis/inference_analysis.py method=vina vina_binding_site_method=diffdock dataset=dockgen repeat_index=1
...
```

Analyze inference results for the CASP15 dataset

```bash
# assemble (unrelaxed and post ranking-relaxed) CASP15-compliant prediction submission files for scoring
python3 posebench/models/ensemble_generation.py ensemble_methods=\[vina\] vina_binding_site_methods=\[diffdock\] input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_vina_diffdock_ensemble_predictions_1 skip_existing=true relax_method_ligands_post_ranking=false export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=40 method_top_n_to_select=40 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index=0 ensemble_benchmarking_repeat_index=1
python3 posebench/models/ensemble_generation.py ensemble_methods=\[vina\] vina_binding_site_methods=\[diffdock\] input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_vina_diffdock_ensemble_predictions_1 skip_existing=true relax_method_ligands_post_ranking=true export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=40 method_top_n_to_select=40 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index=0 ensemble_benchmarking_repeat_index=1
# NOTE: the suffixes for both `output_dir` and `ensemble_benchmarking_repeat_index` should be modified to e.g., 2, 3, ...
...
# now score the CASP15-compliant submissions using the official CASP scoring pipeline
python3 posebench/analysis/inference_analysis_casp.py method=vina vina_binding_site_method=diffdock dataset=casp15 repeat_index=1
...
```

### How to run inference with `TULIP`

Gather all template ligands generated by `TULIP` via its dedicated [GitHub repository](https://github.com/BioinfoMachineLearning/tulip) and collate the resulting ligand fragment SDF files

```bash
python3 posebench/data/tulip_output_extraction.py dataset=posebusters_benchmark
python3 posebench/data/tulip_output_extraction.py dataset=astex_diverse
python3 posebench/data/tulip_output_extraction.py dataset=casp15
```

Relax the generated ligand structures inside of their respective protein pockets

```bash
python3 posebench/models/inference_relaxation.py method=tulip dataset=posebusters_benchmark remove_initial_protein_hydrogens=true assign_partial_charges_manually=true num_processes=1
...
python3 posebench/models/inference_relaxation.py method=tulip dataset=astex_diverse remove_initial_protein_hydrogens=true assign_partial_charges_manually=true num_processes=1
...
```

**NOTE**: Increase `num_processes` according to your available CPU/GPU resources to improve throughput

Analyze inference results for each dataset

```bash
python3 posebench/analysis/inference_analysis.py method=tulip dataset=posebusters_benchmark
...
python3 posebench/analysis/inference_analysis.py method=tulip dataset=astex_diverse
...
```

Analyze inference results for the CASP15 dataset

```bash
# then assemble (unrelaxed and post ranking-relaxed) CASP15-compliant prediction submission files for scoring
python3 posebench/models/ensemble_generation.py ensemble_methods=\[tulip\] input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_tulip_ensemble_predictions_1 skip_existing=true relax_method_ligands_post_ranking=false export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=40 method_top_n_to_select=40 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index=0 ensemble_benchmarking_repeat_index=1
python3 posebench/models/ensemble_generation.py ensemble_methods=\[tulip\] input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_tulip_ensemble_predictions_1 skip_existing=true relax_method_ligands_post_ranking=true export_file_format=casp15 export_top_n=5 combine_casp_output_files=true max_method_predictions=40 method_top_n_to_select=40 resume=true ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 cuda_device_index=0 ensemble_benchmarking_repeat_index=1
# NOTE: the suffixes for both `output_dir` and `ensemble_benchmarking_repeat_index` should be modified to e.g., 2, 3, ...
...
# now score the CASP15-compliant submissions using the official CASP scoring pipeline
python3 posebench/analysis/inference_analysis_casp.py method=tulip dataset=casp15 targets='[H1135, H1171v1, H1171v2, H1172v1, H1172v2, H1172v3, H1172v4, T1124, T1127v2, T1152, T1158v1, T1158v2, T1158v3, T1158v4, T1186, T1187]'
...
```

</details>

## How to run inference with a method ensemble

<details>

Using an `ensemble` of methods, generate predictions for a new protein target using each method and (e.g., consensus-)rank the pool of predictions

```bash
# generate each method's prediction script for a target
# NOTE: to predict input ESMFold protein structures when they are not already locally available in `data/ensemble_proteins/`, e.g., on a SLURM cluster first run e.g., `srun --partition=gpu --gres=gpu:A100:1 --mem=59G --time=01:00:00 --pty bash` to ensure a GPU is available for inference
python3 posebench/models/ensemble_generation.py input_csv_filepath=data/test_cases/5S8I_2LY/ensemble_inputs.csv output_dir=data/test_cases/5S8I_2LY/top_consensus_ensemble_predictions_1 max_method_predictions=40 ensemble_ranking_method=consensus resume=false ensemble_methods='[diffdock, dynamicbind, neuralplexer, rfaa]'
# ...
# now, manually run each desired method's generated prediction script, with the exception of AutoDock Vina which uses other methods' predictions
# ...
python3 posebench/models/ensemble_generation.py input_csv_filepath=data/test_cases/5S8I_2LY/ensemble_inputs.csv output_dir=data/test_cases/5S8I_2LY/top_consensus_ensemble_predictions_1 max_method_predictions=40 ensemble_ranking_method=consensus resume=true generate_vina_scripts=true vina_binding_site_methods=[diffdock]
# now, manually run AutoDock Vina's generated prediction script for each binding site prediction method
#...
# lastly, organize each method's predictions together
python3 posebench/models/ensemble_generation.py input_csv_filepath=data/test_cases/5S8I_2LY/ensemble_inputs.csv output_dir=data/test_cases/5S8I_2LY/top_consensus_ensemble_predictions_1 max_method_predictions=40 ensemble_ranking_method=consensus resume=true generate_vina_scripts=false vina_binding_site_methods=[diffdock]
```

Benchmark (ensemble-)ranked predictions across each test dataset

```bash
# benchmark using the PoseBusters Benchmark dataset e.g., after generating 40 complexes per target with each method
python3 posebench/models/ensemble_generation.py input_csv_filepath=data/test_cases/posebusters_benchmark/ensemble_inputs.csv output_dir=data/test_cases/posebusters_benchmark/top_consensus_ensemble_predictions_1 max_method_predictions=40 export_top_n=1 export_file_format=null skip_existing=true relax_method_ligands_post_ranking=false resume=true cuda_device_index=0 ensemble_methods='[diffdock, dynamicbind, neuralplexer, rfaa, tulip, vina]' ensemble_benchmarking=true ensemble_benchmarking_dataset=posebusters_benchmark ensemble_ranking_method=consensus ensemble_benchmarking_repeat_index=1
python3 posebench/models/ensemble_generation.py input_csv_filepath=data/test_cases/posebusters_benchmark/ensemble_inputs.csv output_dir=data/test_cases/posebusters_benchmark/top_consensus_ensemble_predictions_1 max_method_predictions=40 export_top_n=1 export_file_format=null skip_existing=true relax_method_ligands_post_ranking=true resume=true cuda_device_index=0 ensemble_methods='[diffdock, dynamicbind, neuralplexer, rfaa, tulip, vina]' ensemble_benchmarking=true ensemble_benchmarking_dataset=posebusters_benchmark ensemble_ranking_method=consensus ensemble_benchmarking_repeat_index=1
...
# benchmark using the Astex Diverse dataset e.g., after generating 40 complexes per target with each method
python3 posebench/models/ensemble_generation.py input_csv_filepath=data/test_cases/astex_diverse/ensemble_inputs.csv output_dir=data/test_cases/astex_diverse/top_consensus_ensemble_predictions_1 max_method_predictions=40 export_top_n=1 export_file_format=null skip_existing=true relax_method_ligands_post_ranking=false resume=true cuda_device_index=0 ensemble_methods='[diffdock, dynamicbind, neuralplexer, rfaa, tulip, vina]' ensemble_benchmarking=true ensemble_benchmarking_dataset=astex_diverse ensemble_ranking_method=consensus ensemble_benchmarking_repeat_index=1
python3 posebench/models/ensemble_generation.py input_csv_filepath=data/test_cases/astex_diverse/ensemble_inputs.csv output_dir=data/test_cases/astex_diverse/top_consensus_ensemble_predictions_1 max_method_predictions=40 export_top_n=1 export_file_format=null skip_existing=true relax_method_ligands_post_ranking=true resume=true cuda_device_index=0 ensemble_methods='[diffdock, dynamicbind, neuralplexer, rfaa, tulip, vina]' ensemble_benchmarking=true ensemble_benchmarking_dataset=astex_diverse ensemble_ranking_method=consensus ensemble_benchmarking_repeat_index=1
...
# benchmark using the DockGen dataset e.g., after generating 40 complexes per target with each method
python3 posebench/models/ensemble_generation.py input_csv_filepath=data/test_cases/dockgen/ensemble_inputs.csv output_dir=data/test_cases/dockgen/top_consensus_ensemble_predictions_1 max_method_predictions=40 export_top_n=1 export_file_format=null skip_existing=true relax_method_ligands_post_ranking=false resume=true cuda_device_index=0 ensemble_methods='[diffdock, dynamicbind, neuralplexer, rfaa, vina]' ensemble_benchmarking=true ensemble_benchmarking_dataset=dockgen ensemble_ranking_method=consensus ensemble_benchmarking_repeat_index=1
python3 posebench/models/ensemble_generation.py input_csv_filepath=data/test_cases/dockgen/ensemble_inputs.csv output_dir=data/test_cases/dockgen/top_consensus_ensemble_predictions_1 max_method_predictions=40 export_top_n=1 export_file_format=null skip_existing=true relax_method_ligands_post_ranking=true resume=true cuda_device_index=0 ensemble_methods='[diffdock, dynamicbind, neuralplexer, rfaa, vina]' ensemble_benchmarking=true ensemble_benchmarking_dataset=dockgen ensemble_ranking_method=consensus ensemble_benchmarking_repeat_index=1
...
# benchmark using the CASP15 dataset e.g., after generating 40 complexes per target with each method
python3 posebench/models/ensemble_generation.py input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_consensus_ensemble_predictions_1 combine_casp_output_files=true max_method_predictions=40 export_top_n=5 export_file_format=casp15 skip_existing=true relax_method_ligands_post_ranking=false resume=true cuda_device_index=0 ensemble_methods='[diffdock, dynamicbind, neuralplexer, rfaa, tulip, vina]' ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 ensemble_ranking_method=consensus ensemble_benchmarking_repeat_index=1
python3 posebench/models/ensemble_generation.py input_csv_filepath=data/test_cases/casp15/ensemble_inputs.csv output_dir=data/test_cases/casp15/top_consensus_ensemble_predictions_1 combine_casp_output_files=true max_method_predictions=40 export_top_n=5 export_file_format=casp15 skip_existing=true relax_method_ligands_post_ranking=true resume=true cuda_device_index=0 ensemble_methods='[diffdock, dynamicbind, neuralplexer, rfaa, tulip, vina]' ensemble_benchmarking=true ensemble_benchmarking_dataset=casp15 ensemble_ranking_method=consensus ensemble_benchmarking_repeat_index=1
...
# analyze benchmarking results for the PoseBusters Benchmark dataset
python3 posebench/analysis/inference_analysis.py method=ensemble dataset=posebusters_benchmark repeat_index=1
...
# analyze benchmarking results for the Astex Diverse dataset
python3 posebench/analysis/inference_analysis.py method=ensemble dataset=astex_diverse repeat_index=1
...
# analyze benchmarking results for the DockGen dataset
python3 posebench/analysis/inference_analysis.py method=ensemble dataset=dockgen repeat_index=1
...
# analyze benchmarking results for the CASP15 dataset
python3 posebench/analysis/inference_analysis_casp.py method=ensemble dataset=casp15 ensemble_ranking_method=consensus repeat_index=1
...
```

To benchmark ensemble ranking using the above commands, you must have already run the corresponding `*_inference.py` script for each method described in the section [How to run inference with individual methods](#how-to-run-inference-with-individual-methods) (with the exception of `FABind`, which will not referenced during CASP15 benchmarking)

**NOTE**: In addition to having `consensus` as an available value for `ensemble_ranking_method`, one can also set `ensemble_ranking_method=ff` to have the method ensemble's top-ranked predictions selected using the criterion of "minimum (molecular dynamics) force field energy" (albeit while incurring a very large runtime complexity)

</details>

## How to create comparative plots of inference results

<details>

Execute (and customize as desired) notebooks to prepare paper-ready result plots

```bash
jupyter notebook notebooks/posebusters_astex_inference_results_plotting.ipynb
jupyter notebook notebooks/posebusters_pocket_only_inference_results_plotting.ipynb
jupyter notebook notebooks/dockgen_inference_results_plotting.ipynb
jupyter notebook notebooks/casp15_inference_results_plotting.ipynb
```

</details>

## For developers

<details>

### Dependency management

We use `mamba` to manage the project's underlying dependencies. Notably, to update the dependencies listed in a particular `environments/*_environment.yml` file:

```bash
mamba env export > env.yaml # e.g., run this after installing new dependencies locally within a given `conda` environment
diff environments/posebench_environment.yaml env.yaml # note the differences and copy accepted changes back into e.g., `environments/posebench_environment.yaml`
rm env.yaml # clean up temporary environment file
```

### Code formatting

We use `pre-commit` to automatically format the project's code. To set up `pre-commit` (one time only) for automatic code linting and formatting upon each execution of `git commit`:

```bash
pre-commit install
```

To manually reformat all files in the project as desired:

```bash
pre-commit run -a
```

### Documentation

We `sphinx` to maintain the project's code documentation. To build a local version of the project's `sphinx` documentation web pages:

```bash
# assuming you are located in the `PoseBench` top-level directory
pip install -r docs/.docs.requirements # one-time only
rm -rf docs/build/ && sphinx-build docs/source/ docs/build/ # NOTE: errors can safely be ignored
```

</details>

## Acknowledgements

`PoseBench` builds upon the source code and data from the following projects:

- [AutoDock-Vina](https://github.com/ccsb-scripps/AutoDock-Vina)
- [casp15_ligand](https://git.scicore.unibas.ch/schwede/casp15_ligand)
- [DiffDock](https://github.com/gcorso/DiffDock)
- [FABind](https://github.com/QizhiPei/FABind)
- [DynamicBind](https://github.com/luwei0917/DynamicBind)
- [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
- [NeuralPLexer](https://github.com/zrqiao/NeuralPLexer)
- [posebusters](https://github.com/maabuu/posebusters)
- [posebusters_em](https://github.com/maabuu/posebusters_em)
- [ProteinWorkshop](https://github.com/a-r-j/ProteinWorkshop)
- [RoseTTAFold-All-Atom](https://github.com/baker-laboratory/RoseTTAFold-All-Atom)
- [tulip](https://github.com/BioinfoMachineLearning/tulip)

We thank all their contributors and maintainers!

## Citing this work

If you use the code or benchmark method predictions associated with this repository or otherwise find this work useful, please cite:

```bibtex
@inproceedings{morehead2024posebench,
  title={Deep Learning for Protein-Ligand Docking: Are We There Yet?},
  author={Morehead, Alex and Giri, Nabin and Liu, Jian and Cheng, Jianlin},
  booktitle={ICML AI4Science Workshop},
  year={2024},
  note={selected as a spotlight presentation},
}
```

## Bonus

<details>

Lastly, thanks to Stable Diffusion for generating this quaint representation of what my brain looked like after assembling this codebase. 💣

<div align="center">

<img src="./img/WorkBench.jpeg" width="600">

</div>

</details>
