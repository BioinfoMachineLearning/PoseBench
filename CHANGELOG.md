### 0.6.0 - TBD

- Added new baseline methods (AlphaFold 3, NeuralPLexer3, Chai-1 with multiple sequence alignments (MSAs))
- Added new binding site-focused implementation of `complex_alignment.py` based on PyMOL's `align` command, which in many cases yields 3x better docking evaluation scores for baseline methods
- Added new script for analyzing baseline methods' protein conformational changes w.r.t. input (e.g., AlphaFold) protein structures and the corresponding reference (crystal) protein structures
- Added the new centroid RMSD and PLIF-EMD/WM metrics
- Added a failure mode analysis notebook
- Introducing DockGen-E, a new version of the DockGen benchmark dataset featuring enhanced biomolecular context for docking and co-folding predictions - namely, now all DockGen complexes represent the first (biologically relevant) bioassembly of the corresponding PDB structure
- For the single-ligand datasets (i.e., Astex Diverse, PoseBusters Benchmark, and DockGen), now providing each baseline method with primary *and cofactor* ligand SMILES strings for prediction, to enhance the biomolecular context of these methods' predicted structures - as a result, for these single-ligand datasets, now the predicted ligand *most similar* to the primary ligand (in terms of both Tanimoto and structural similarity) is selected for scoring
- Updated Chai-1's inference code to commit `44375d5d4ea44c0b5b7204519e63f40b063e4a7c`, and ran it also with NeuralPLexer3's (paired) MSAs
- Replaced all AlphaFold 3 server predictions of each dataset's protein structures with predictions from AlphaFold 3's local inference code
- Pocket-only benchmarking has been deprecated
- With all the above changed in place, simplified, re-ran, and re-analyzed all baseline methods for each benchmark dataset, and updated the baseline predictions and datasets (now containing standardized MSAs) hosted on Zenodo

### 0.5.0 - 09/30/2024

- Added results with AlphaFold 3 predicted structures (now the default)
- Added results for the new Chai-1 model from Chai Discovery
- Added a new inference sweep pipeline for HPC clusters to allow users to quickly run an exhaustive sweep of all baseline methods, datasets, and tasks e.g., using generated batch scripts and a SLURM scheduler
- Updated Zenodo links to point to the latest version of the project's Zenodo record, which now includes the above-mentioned AlphaFold 3 predicted structures and baseline method results using them
- Updated documentation project-wide according to the additions listed above
- Fixed some CI testing issues

### 0.4.0 - 08/12/2024

- Renamed `src` root directory to `posebench` to support `pip` packaging
- Added and documented `pip` installation option
- Added mmCIF to PDB file conversion script
- Added apo-to-holo predicted protein structure accuracy assessment and plotting script
- Added support to `notebooks/dockgen_inference_results_plotting.ipynb` for analyzing the protein-ligand interactions within the PDBBind 2020 dataset's experimental structures
- Updated dataset documentation in `README.md`

### 0.3.0 - 07/07/2024

- Added a notebook for plotting expanded DockGen benchmark results
- Added support for scoring relaxed-protein predictions
- Fixed runtime error for relaxed-protein energy minimization
- Fixed runtime error for compute benchmarking RoseTTAFold-All-Atom predictions

### 0.2.0 - 07/04/2024

- Added P2Rank as a new binding site prediction method available to use with AutoDock-Vina
- Added OpenJDK to the `PoseBench` Conda environment to enable P2Rank inference
- Added a script to benchmark the required compute resources for each baseline method
- Updated citation
- Corrected directory navigation instructions (i.e., `cd` references) in `README.md` to reflect the directory structure of each Zenodo archive file
- Corrected Biopython, NumPy, and ProDy versions in the DiffDock Conda environment to avoid GCC compilation errors

### 0.1.0 - 06/08/2024

- First public release
