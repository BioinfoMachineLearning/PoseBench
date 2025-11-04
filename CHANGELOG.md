### 1.0.0 - 11/04/2025

**Changes**:

- Added dot plots on top of bar plots for improved clarity in CASP15 results.

### 0.7.1 - 08/11/2025

**Changes**:

- Updated PyPI package and documentation.

### 0.7.0 - 08/11/2025

**Additions**:

- Added new baseline methods (Boltz-1/2 w/ and w/o with multiple sequence alignments (MSAs)).
- Added a new failure mode analysis notebook incorporating PLINDER for robust method generalization analysis(n.b., see new arXiv preprint for more details).
- Added a script for performing custom method generalization analyses (e.g., when new methods are released which were trained on more recent splits of the PDB compared to AlphaFold 3's cutoff date of September 30, 2021).

**Changes**:

- Updated the `posebusters` package version to (at the time of writing) the latest version `0.4.5` to integrate new (stricter) pose checks (e.g., regarding flat aliphatic rings). Overall, each method's PoseBusters validity scores have decreased slightly on average (as expected).

**Bug Fixes**:

- Fixed a bug related to the number of DockGen-E protein chains provided to Chai-1 for inference on this dataset. Chai-1's DockGen-E results accordingly have been updated.

**Results**:

- With all the above changes in place, re-analyzed all baseline methods for each benchmark dataset, and updated the baseline predictions hosted on Zenodo.
- **NOTE**: The updated arXiv preprint should be publicly available by 08/15/2025.

### 0.6.0 - 02/09/2025

**Additions**:

- Added new baseline methods (AlphaFold 3, Chai-1 with multiple sequence alignments (MSAs))
- Added new binding site-focused implementation of `complex_alignment.py` based on PyMOL's `align` command, which in many cases yields 3x better docking evaluation scores for baseline methods
- Added new script for analyzing baseline methods' protein conformational changes w.r.t. input (e.g., AlphaFold) protein structures and the corresponding reference (crystal) protein structures
- Added the new centroid RMSD and **PLIF-EMD/WM** metrics (n.b., see new arXiv preprint for more details)
- Added a failure mode analysis notebook (n.b., see new arXiv preprint for more details)

**Changes**:

- Introducing **DockGen-E**, a new version of the DockGen benchmark dataset featuring enhanced biomolecular context for docking and co-folding predictions - namely, now all DockGen complexes represent the first (biologically relevant) bioassembly of the corresponding PDB structure
- For the single-ligand datasets (i.e., Astex Diverse, PoseBusters Benchmark, and DockGen), now providing each baseline method with primary _and cofactor_ ligand SMILES strings for prediction, to enhance the biomolecular context of these methods' predicted structures - as a result, for these single-ligand datasets, now the predicted ligand _most similar_ to the primary ligand (in terms of both Tanimoto and structural similarity) is selected for scoring (which adds an additional layer of challenges for baseline methods)
- Updated Chai-1's inference code to commit `44375d5d4ea44c0b5b7204519e63f40b063e4a7c`, and ran it also with standardized (paired) MSAs
- Replaced all AlphaFold 3 server predictions of each dataset's protein structures with predictions from AlphaFold 3's local inference code

**Deprecations**:

- Pocket-only benchmarking has been deprecated

**Results**:

- With all the above changes in place, simplified, re-ran, and re-analyzed all baseline methods for each benchmark dataset, and updated the baseline predictions and datasets (now containing standardized MSAs) hosted on Zenodo
- **NOTE**: The updated arXiv preprint should be publicly available by 02/12/2025

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
