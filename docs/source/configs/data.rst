Data
==============

This section describes the configurations for various data-related scripts.


Input data components
------------------------

These data component configurations are used to modify how the input (apo) protein structures are predicted or aligned.

Protein apo-to-holo alignment
^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../configs/data/components/protein_apo_to_holo_alignment.yaml
    :language: yaml
    :caption: :file:`data/components/protein_apo_to_holo_alignment.yaml`

FASTA preparation
^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../configs/data/components/fasta_preparation.yaml
    :language: yaml
    :caption: :file:`data/components/fasta_preparation.yaml`

ESMFold sequence preparation
^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../configs/data/components/esmfold_sequence_preparation.yaml
    :language: yaml
    :caption: :file:`data/components/esmfold_sequence_preparation.yaml`


Method data parsers
------------------------

These data parser configurations are used to modify how the input (output) protein-ligand complex structures of each method are prepared (extracted).

Binding site crop preparation
^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../configs/data/binding_site_crop_preparation.yaml
    :language: yaml
    :caption: :file:`data/binding_site_crop_preparation.yaml`

DiffDock input preparation
^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../configs/data/diffdock_input_preparation.yaml
    :language: yaml
    :caption: :file:`data/diffdock_input_preparation.yaml`

FABind input preparation
^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../configs/data/fabind_input_preparation.yaml
    :language: yaml
    :caption: :file:`data/fabind_input_preparation.yaml`

DynamicBind input preparation
^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../configs/data/dynamicbind_input_preparation.yaml
    :language: yaml
    :caption: :file:`data/dynamicbind_input_preparation.yaml`

NeuralPLexer input preparation
^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../configs/data/neuralplexer_input_preparation.yaml
    :language: yaml
    :caption: :file:`data/neuralplexer_input_preparation.yaml`

FlowDock input preparation
^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../configs/data/flowdock_input_preparation.yaml
    :language: yaml
    :caption: :file:`data/flowdock_input_preparation.yaml`

RoseTTAFold-All-Atom input preparation
^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../configs/data/rfaa_input_preparation.yaml
    :language: yaml
    :caption: :file:`data/rfaa_input_preparation.yaml`

RoseTTAFold-All-Atom output extraction
^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../configs/data/rfaa_output_extraction.yaml
    :language: yaml
    :caption: :file:`data/rfaa_output_extraction.yaml`

Chai-1 input preparation
^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../configs/data/chai_input_preparation.yaml
    :language: yaml
    :caption: :file:`data/chai_input_preparation.yaml`

Chai-1 output extraction
^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../configs/data/chai_output_extraction.yaml
    :language: yaml
    :caption: :file:`data/chai_output_extraction.yaml`

AlphaFold 3 output extraction
^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../configs/data/af3_output_extraction.yaml
    :language: yaml
    :caption: :file:`data/af3_output_extraction.yaml`

TULIP output extraction
^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../configs/data/tulip_output_extraction.yaml
    :language: yaml
    :caption: :file:`data/tulip_output_extraction.yaml`
