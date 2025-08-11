Model
==============

This section describes the configurations for various method-related scripts.


Method inference
------------------------

These configurations are used to specify how inference is performed with each method.

DiffDock inference
^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../configs/model/diffdock_inference.yaml
    :language: yaml
    :caption: :file:`model/diffdock_inference.yaml`

FABind inference
^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../configs/model/fabind_inference.yaml
    :language: yaml
    :caption: :file:`model/fabind_inference.yaml`

DynamicBind inference
^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../configs/model/dynamicbind_inference.yaml
    :language: yaml
    :caption: :file:`model/dynamicbind_inference.yaml`

NeuralPLexer inference
^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../configs/model/neuralplexer_inference.yaml
    :language: yaml
    :caption: :file:`model/neuralplexer_inference.yaml`

FlowDock inference
^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../configs/model/flowdock_inference.yaml
    :language: yaml
    :caption: :file:`model/flowdock_inference.yaml`

RoseTTAFold-All-Atom inference
^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../configs/model/rfaa_inference.yaml
    :language: yaml
    :caption: :file:`model/rfaa_inference.yaml`

Chai-1 inference
^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../configs/model/chai_inference.yaml
    :language: yaml
    :caption: :file:`model/chai_inference.yaml`

Boltz-1 inference
^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../configs/model/boltz_inference.yaml
    :language: yaml
    :caption: :file:`model/boltz_inference.yaml`

Vina inference
^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../configs/model/vina_inference.yaml
    :language: yaml
    :caption: :file:`model/vina_inference.yaml`


Ensemble inference
------------------------

This configuration is used to specify how inference is performed with a method ensemble (e.g., via `consensus` ranking).

.. note::
    This script not only enables inference with a method ensemble, but it also provides a unified wrapper with which one
    can relax and structure a method's predictions in a CASP-compliant file format for scoring.

Ensemble generation
^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../configs/model/ensemble_generation.yaml
    :language: yaml
    :caption: :file:`model/ensemble_generation.yaml`


Structure relaxation
------------------------

These configurations are used to specify how relaxation is (optionally) applied to a predicted protein-ligand complex structure using molecular dynamics (i.e., `OpenMM <https://openmm.org>`_).

.. note::
    The `inference_relaxation` configuration describes the behavior of the script that serves as an entry point for the relaxation process. The `minimize_energy` configuration is a multi-ligand generalization of the main energy minimization script originally implemented for the `PoseBusters <https://github.com/maabuu/posebusters_em>`_ software suite.

Inference relaxation (entry point)
^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../configs/model/inference_relaxation.yaml
    :language: yaml
    :caption: :file:`model/inference_relaxation.yaml`

Minimize energy (relaxation engine)
^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../configs/model/minimize_energy.yaml
    :language: yaml
    :caption: :file:`model/minimize_energy.yaml`
