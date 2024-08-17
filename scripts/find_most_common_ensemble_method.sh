#!/bin/bash

for dataset in astex_diverse posebusters_benchmark dockgen casp15; do
    if [ "$dataset" = "posebusters_benchmark" ]; then
        echo "Baseline method most frequently selected by the (structural) consensus ensembling baseline for $dataset (pocket-only):"

        # Step 1: Find all files in the ensemble baseline method's subdirectories for a given dataset
        find data/test_cases/"$dataset"/top_consensus_pocket_only_ensemble_predictions_*/ -type f |

        # Step 2: Extract the method names using grep with a regex
        grep -oP '(?<=/)[^/]+(?=_rank)' |

        # Step 3: Count the occurrences of each method using awk
        awk '{count[$1]++} END {for (method in count) print count[method], method}' |

        # Step 4: Sort the results and find the most frequent method at the top of the command's output
        sort -nr | head -n 1
    fi

    echo "Baseline method most frequently selected by the (structural) consensus ensembling baseline for $dataset:"

    # Step 1: Find all files in the ensemble baseline method's subdirectories for a given dataset
    find data/test_cases/"$dataset"/top_consensus_ensemble_predictions_*/ -type f |

    # Step 2: Extract the method names using grep with a regex
    grep -oP '(?<=/)[^/]+(?=_rank)' |

    # Step 3: Count the occurrences of each method using awk
    awk '{count[$1]++} END {for (method in count) print count[method], method}' |

    # Step 4: Sort the results and find the most frequent method at the top of the command's output
    sort -nr | head -n 1
done
