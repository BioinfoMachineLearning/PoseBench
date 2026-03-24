Analysis
==============

This section describes the configurations for various analysis-related scripts.

Complex alignment
--------------------------

This config file is used to determine how a predicted protein-ligand complex structure is optimally aligned to a corresponding ground-truth protein-ligand complex.

.. literalinclude:: ../../../configs/analysis/complex_alignment.yaml
    :language: yaml
    :caption: :file:`analysis/complex_alignment.yaml`


Inference analysis (PoseBusters, Astex, and DockGen)
--------------------------

This config file is used to determine how to score a predicted protein-ligand complex from the PoseBusters Benchmark, Astex Diverse, or DockGen datasets.

.. literalinclude:: ../../../configs/analysis/inference_analysis.yaml
    :language: yaml
    :caption: :file:`analysis/inference_analysis.yaml`


Inference analysis (CASP)
--------------------------

This config file is used to determine how to score a predicted protein-ligand complex from the CASP15 dataset.

.. literalinclude:: ../../../configs/analysis/inference_analysis_casp.yaml
    :language: yaml
    :caption: :file:`analysis/inference_analysis_casp.yaml`
