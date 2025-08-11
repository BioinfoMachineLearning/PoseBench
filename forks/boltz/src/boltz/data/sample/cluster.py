from typing import Dict, Iterator, List

import numpy as np
from numpy.random import RandomState

from boltz.data import const
from boltz.data.types import ChainInfo, InterfaceInfo, Record
from boltz.data.sample.sampler import Sample, Sampler


def get_chain_cluster(chain: ChainInfo, record: Record) -> str:  # noqa: ARG001
    """Get the cluster id for a chain.

    Parameters
    ----------
    chain : ChainInfo
        The chain id to get the cluster id for.
    record : Record
        The record the interface is part of.

    Returns
    -------
    str
        The cluster id of the chain.

    """
    return chain.cluster_id


def get_interface_cluster(interface: InterfaceInfo, record: Record) -> str:
    """Get the cluster id for an interface.

    Parameters
    ----------
    interface : InterfaceInfo
        The interface to get the cluster id for.
    record : Record
        The record the interface is part of.

    Returns
    -------
    str
        The cluster id of the interface.

    """
    chain1 = record.chains[interface.chain_1]
    chain2 = record.chains[interface.chain_2]

    cluster_1 = str(chain1.cluster_id)
    cluster_2 = str(chain2.cluster_id)

    cluster_id = (cluster_1, cluster_2)
    cluster_id = tuple(sorted(cluster_id))

    return cluster_id


def get_chain_weight(
    chain: ChainInfo,
    record: Record,  # noqa: ARG001
    clusters: Dict[str, int],
    beta_chain: float,
    alpha_prot: float,
    alpha_nucl: float,
    alpha_ligand: float,
) -> float:
    """Get the weight of a chain.

    Parameters
    ----------
    chain : ChainInfo
        The chain to get the weight for.
    record : Record
        The record the chain is part of.
    clusters : Dict[str, int]
        The cluster sizes.
    beta_chain : float
        The beta value for chains.
    alpha_prot : float
        The alpha value for proteins.
    alpha_nucl : float
        The alpha value for nucleic acids.
    alpha_ligand : float
        The alpha value for ligands.

    Returns
    -------
    float
        The weight of the chain.

    """
    prot_id = const.chain_type_ids["PROTEIN"]
    rna_id = const.chain_type_ids["RNA"]
    dna_id = const.chain_type_ids["DNA"]
    ligand_id = const.chain_type_ids["NONPOLYMER"]

    weight = beta_chain / clusters[chain.cluster_id]
    if chain.mol_type == prot_id:
        weight *= alpha_prot
    elif chain.mol_type in [rna_id, dna_id]:
        weight *= alpha_nucl
    elif chain.mol_type == ligand_id:
        weight *= alpha_ligand

    return weight


def get_interface_weight(
    interface: InterfaceInfo,
    record: Record,
    clusters: Dict[str, int],
    beta_interface: float,
    alpha_prot: float,
    alpha_nucl: float,
    alpha_ligand: float,
) -> float:
    """Get the weight of an interface.

    Parameters
    ----------
    interface : InterfaceInfo
        The interface to get the weight for.
    record : Record
        The record the interface is part of.
    clusters : Dict[str, int]
        The cluster sizes.
    beta_interface : float
        The beta value for interfaces.
    alpha_prot : float
        The alpha value for proteins.
    alpha_nucl : float
        The alpha value for nucleic acids.
    alpha_ligand : float
        The alpha value for ligands.

    Returns
    -------
    float
        The weight of the interface.

    """
    prot_id = const.chain_type_ids["PROTEIN"]
    rna_id = const.chain_type_ids["RNA"]
    dna_id = const.chain_type_ids["DNA"]
    ligand_id = const.chain_type_ids["NONPOLYMER"]

    chain1 = record.chains[interface.chain_1]
    chain2 = record.chains[interface.chain_2]

    n_prot = (chain1.mol_type) == prot_id
    n_nuc = chain1.mol_type in [rna_id, dna_id]
    n_ligand = chain1.mol_type == ligand_id

    n_prot += chain2.mol_type == prot_id
    n_nuc += chain2.mol_type in [rna_id, dna_id]
    n_ligand += chain2.mol_type == ligand_id

    weight = beta_interface / clusters[get_interface_cluster(interface, record)]
    weight *= alpha_prot * n_prot + alpha_nucl * n_nuc + alpha_ligand * n_ligand
    return weight


class ClusterSampler(Sampler):
    """The weighted sampling approach, as described in AF3.

    Each chain / interface is given a weight according
    to the following formula, and sampled accordingly:

    w = b / n_clust *(a_prot * n_prot + a_nuc * n_nuc
        + a_ligand * n_ligand)

    """

    def __init__(
        self,
        alpha_prot: float = 3.0,
        alpha_nucl: float = 3.0,
        alpha_ligand: float = 1.0,
        beta_chain: float = 0.5,
        beta_interface: float = 1.0,
    ) -> None:
        """Initialize the sampler.

        Parameters
        ----------
        alpha_prot : float, optional
            The alpha value for proteins.
        alpha_nucl : float, optional
            The alpha value for nucleic acids.
        alpha_ligand : float, optional
            The alpha value for ligands.
        beta_chain : float, optional
            The beta value for chains.
        beta_interface : float, optional
            The beta value for interfaces.

        """
        self.alpha_prot = alpha_prot
        self.alpha_nucl = alpha_nucl
        self.alpha_ligand = alpha_ligand
        self.beta_chain = beta_chain
        self.beta_interface = beta_interface

    def sample(self, records: List[Record], random: RandomState) -> Iterator[Sample]:  # noqa: C901, PLR0912
        """Sample a structure from the dataset infinitely.

        Parameters
        ----------
        records : List[Record]
            The records to sample from.
        random : RandomState
            The random state for reproducibility.

        Yields
        ------
        Sample
            A data sample.

        """
        # Compute chain cluster sizes
        chain_clusters: Dict[str, int] = {}
        for record in records:
            for chain in record.chains:
                if not chain.valid:
                    continue
                cluster_id = get_chain_cluster(chain, record)
                if cluster_id not in chain_clusters:
                    chain_clusters[cluster_id] = 0
                chain_clusters[cluster_id] += 1

        # Compute interface clusters sizes
        interface_clusters: Dict[str, int] = {}
        for record in records:
            for interface in record.interfaces:
                if not interface.valid:
                    continue
                cluster_id = get_interface_cluster(interface, record)
                if cluster_id not in interface_clusters:
                    interface_clusters[cluster_id] = 0
                interface_clusters[cluster_id] += 1

        # Compute weights
        items, weights = [], []
        for record in records:
            for chain_id, chain in enumerate(record.chains):
                if not chain.valid:
                    continue
                weight = get_chain_weight(
                    chain,
                    record,
                    chain_clusters,
                    self.beta_chain,
                    self.alpha_prot,
                    self.alpha_nucl,
                    self.alpha_ligand,
                )
                items.append((record, 0, chain_id))
                weights.append(weight)

            for int_id, interface in enumerate(record.interfaces):
                if not interface.valid:
                    continue
                weight = get_interface_weight(
                    interface,
                    record,
                    interface_clusters,
                    self.beta_interface,
                    self.alpha_prot,
                    self.alpha_nucl,
                    self.alpha_ligand,
                )
                items.append((record, 1, int_id))
                weights.append(weight)

        # Sample infinitely
        weights = np.array(weights) / np.sum(weights)
        while True:
            item_idx = random.choice(len(items), p=weights)
            record, kind, index = items[item_idx]
            if kind == 0:
                yield Sample(record=record, chain_id=index)
            else:
                yield Sample(record=record, interface_id=index)
