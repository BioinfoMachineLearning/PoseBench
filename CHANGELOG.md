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
