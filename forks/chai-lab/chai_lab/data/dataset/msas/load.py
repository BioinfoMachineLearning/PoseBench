# Copyright (c) 2024 Chai Discovery, Inc.
# This source code is licensed under the Chai Discovery Community License
# Agreement (LICENSE.md) found in the root directory of this source tree.

import logging
from pathlib import Path

import torch

from chai_lab.data.dataset.msas.msa_context import MSAContext
from chai_lab.data.dataset.msas.preprocess import (
    drop_duplicates,
    merge_main_msas_by_chain,
    pair_and_merge_msas,
)
from chai_lab.data.dataset.structure.chain import Chain
from chai_lab.data.parsing.msas.a3m import tokenize_sequences_to_arrays
from chai_lab.data.parsing.msas.aligned_pqt import (
    expected_basename,
    parse_aligned_pqt_to_msa_context,
)
from chai_lab.data.parsing.msas.data_source import MSADataSource

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_msa_contexts(
    chains: list[Chain],
    msa_directory: Path,
    pdb_id: str | None = None,
) -> tuple[MSAContext, MSAContext]:
    """
    Looks inside msa_directory to find .aligned.pqt files to load alignments from.

    Returns two contexts

    - First context to tokenize and give to model
    - Second context for computing summary statistics
    """

    pdb_ids = set(chain.entity_data.pdb_id for chain in chains)
    assert len(pdb_ids) == 1, f"Found >1 pdb ids in chains: {pdb_ids=}"

    pdb_id = pdb_id if pdb_id else pdb_ids.pop()

    # MSAs are constructed based on sequence, so use the unique sequences present
    # in input chains to determine the MSAs that need to be loaded

    def get_msa_contexts_for_seq(seq, chain_index) -> MSAContext:
        path = msa_directory / expected_basename(seq)
        if not path.is_file():
            # Try parsing custom chain MSA file
            path = msa_directory / f"{pdb_id}_chain_{chain_index}.aligned.pqt"
        if not path.is_file():
            logger.warning(f"No MSA found for {pdb_id} sequence: {seq}")
            [tokenized_seq] = tokenize_sequences_to_arrays([seq])[0]
            return MSAContext.create_single_seq(
                MSADataSource.QUERY, tokens=torch.from_numpy(tokenized_seq)
            )
        msa = parse_aligned_pqt_to_msa_context(path)
        logger.info(f"MSA found for sequence: {seq}, {msa.depth=}")
        return msa

    # For each chain, either fetch the corresponding MSA or create an empty MSA if it is missing
    # + reindex to handle residues that are tokenized per-atom (this also crops if necessary)
    msa_contexts = [
        get_msa_contexts_for_seq(chain.entity_data.sequence, chain_index)[
            :, chain.structure_context.token_residue_index
        ]
        for chain_index, chain in enumerate(chains)
    ]

    # used later only for profile statistics
    profile_msa = merge_main_msas_by_chain(
        [drop_duplicates(msa) for msa in msa_contexts]
    )

    joined_msa = pair_and_merge_msas(msa_contexts)
    joined_msa = drop_duplicates(joined_msa)  # rare dups after pairings

    logger.info(f"Prepared MSA context with {joined_msa.depth=}")
    return joined_msa, profile_msa
