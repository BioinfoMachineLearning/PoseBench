.. PoseBench documentation master file, created by
   sphinx-quickstart on Sun May 12 14:49:07 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PoseBench's documentation!
============================================


.. mdinclude:: ../../README.md
    :start-line: 4
    :end-line: 14

.. image:: ./_static/PoseBench.png
  :alt: Overview of PoseBench
  :align: center
  :width: 600

.. mdinclude:: ../../README.md
    :start-line: 20
    :end-line: 22


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   tutorials
   data_preparation
   available_methods
   sweep_inference
   method_inference
   ensemble_inference
   comparative_plots
   for_developers
   acknowledgements
   citing_this_work
   bonus

.. toctree::
   :glob:
   :maxdepth: 2
   :hidden:
   :caption: Default Configs

   configs/analysis
   configs/data
   configs/model

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: API Reference

   modules/posebench.binding_site_crop_preparation
   modules/posebench.complex_alignment
   modules/posebench.inference_relaxation
   modules/posebench.minimize_energy
   modules/posebench.ensemble_generation
   modules/posebench.data_utils
   modules/posebench.model_utils
   modules/posebench.utils
   modules/posebench.resolvers

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
