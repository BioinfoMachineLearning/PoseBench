How to run inference with a method ensemble
================

.. mdinclude:: ../../README.md
    :start-line: 1178
    :end-line: 1229

.. note::
    In addition to having `consensus` as an available value for `ensemble_ranking_method`, one can also set `ensemble_ranking_method=ff` to have the method ensemble's top-ranked predictions selected using the criterion of "minimum (molecular dynamics) force field energy" (albeit while incurring a very large runtime complexity).
