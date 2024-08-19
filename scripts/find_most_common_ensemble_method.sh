#!/bin/bash

# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

# Finding for each dataset the most frequently selected baseline method by the (structural) consensus ensembling baseline #

for dataset in astex_diverse posebusters_benchmark dockgen casp15; do
    if [ "$dataset" = "posebusters_benchmark" ]; then
        echo "Top-3 baseline methods most frequently selected by the (structural) consensus ensembling baseline for $dataset (pocket-only):"

        # Step 1: Find all files in the ensemble baseline method's subdirectories for a given dataset
        find data/test_cases/"$dataset"/top_consensus_pocket_only_ensemble_predictions_*/ -type f |

            # Step 2: Extract the method names using grep with a regex
            grep -oP '(?<=/)[^/]+(?=_rank)' |

            # Step 3: Count the occurrences of each method using awk
            awk '{count[$1]++} END {for (method in count) print count[method], method}' |

            # Step 4: Sort the results and find the most frequent methods at the top of the command's output
            sort -nr | head -n 3
    fi

    echo "Top-3 baseline methods most frequently selected by the (structural) consensus ensembling baseline for $dataset:"

    # Step 1: Find all files in the ensemble baseline method's subdirectories for a given dataset
    find data/test_cases/"$dataset"/top_consensus_ensemble_predictions_*/ -type f |

        # Step 2: Extract the method names using grep with a regex
        grep -oP '(?<=/)[^/]+(?=_rank)' |

        # Step 3: Count the occurrences of each method using awk
        awk '{count[$1]++} END {for (method in count) print count[method], method}' |

        # Step 4: Sort the results and find the most frequent methods at the top of the command's output
        sort -nr | head -n 3
done
