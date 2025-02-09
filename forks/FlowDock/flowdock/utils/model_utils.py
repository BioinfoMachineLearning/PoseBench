# Adapted from: https://github.com/zrqiao/NeuralPLexer

import rootutils
import torch
import torch.nn.functional as F
from beartype.typing import Any, Dict, List, Optional, Tuple, Union
from torch_scatter import scatter_max, scatter_min

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from flowdock.utils import RankedLogger

MODEL_BATCH = Dict[str, Any]
STATE_DICT = Dict[str, Any]

log = RankedLogger(__name__, rank_zero_only=True)


class GELUMLP(torch.nn.Module):
    """Simple MLP with post-LayerNorm."""

    def __init__(
        self,
        n_in_feats: int,
        n_out_feats: int,
        n_hidden_feats: Optional[int] = None,
        dropout: float = 0.0,
        zero_init: bool = False,
    ):
        """Initialize the GELUMLP."""
        super().__init__()
        self.dropout = dropout
        if n_hidden_feats is None:
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(n_in_feats, n_in_feats),
                torch.nn.GELU(),
                torch.nn.LayerNorm(n_in_feats),
                torch.nn.Linear(n_in_feats, n_out_feats),
            )
        else:
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(n_in_feats, n_hidden_feats),
                torch.nn.GELU(),
                torch.nn.Dropout(p=self.dropout),
                torch.nn.Linear(n_hidden_feats, n_hidden_feats),
                torch.nn.GELU(),
                torch.nn.LayerNorm(n_hidden_feats),
                torch.nn.Linear(n_hidden_feats, n_out_feats),
            )
        torch.nn.init.xavier_uniform_(self.layers[0].weight, gain=1)
        # zero init for residual branches
        if zero_init:
            self.layers[-1].weight.data.fill_(0.0)
        else:
            torch.nn.init.xavier_uniform_(self.layers[-1].weight, gain=1)

    def _zero_init(self, module):
        """Zero-initialize weights and biases."""
        if isinstance(module, torch.nn.Linear):
            module.weight.data.zero_()
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor):
        """Forward pass through the GELUMLP."""
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers(x)


class SumPooling(torch.nn.Module):
    """Sum pooling layer."""

    def __init__(self, learnable: bool, hidden_dim: int = 1):
        """Initialize the SumPooling layer."""
        super().__init__()
        self.pooled_transform = (
            torch.nn.Linear(hidden_dim, hidden_dim) if learnable else torch.nn.Identity()
        )

    def forward(self, x, dst_idx, dst_size):
        """Forward pass through the SumPooling layer."""
        return self.pooled_transform(segment_sum(x, dst_idx, dst_size))


class AveragePooling(torch.nn.Module):
    """Average pooling layer."""

    def __init__(self, learnable: bool, hidden_dim: int = 1):
        """Initialize the AveragePooling layer."""
        super().__init__()
        self.pooled_transform = (
            torch.nn.Linear(hidden_dim, hidden_dim) if learnable else torch.nn.Identity()
        )

    def forward(self, x, dst_idx, dst_size):
        """Forward pass through the AveragePooling layer."""
        out = torch.zeros(
            dst_size,
            *x.shape[1:],
            dtype=x.dtype,
            device=x.device,
        ).index_add_(0, dst_idx, x)
        nmr = torch.zeros(
            dst_size,
            *x.shape[1:],
            dtype=x.dtype,
            device=x.device,
        ).index_add_(0, dst_idx, torch.ones_like(x))
        return self.pooled_transform(out / (nmr + 1e-8))


def init_weights(m):
    """Initialize weights with Kaiming uniform."""
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)


def segment_sum(src, dst_idx, dst_size):
    """Computes the sum of each segment in a tensor."""
    out = torch.zeros(
        dst_size,
        *src.shape[1:],
        dtype=src.dtype,
        device=src.device,
    ).index_add_(0, dst_idx, src)
    return out


def segment_mean(src, dst_idx, dst_size):
    """Computes the mean value of each segment in a tensor."""
    out = torch.zeros(
        dst_size,
        *src.shape[1:],
        dtype=src.dtype,
        device=src.device,
    ).index_add_(0, dst_idx, src)
    denom = (
        torch.zeros(
            dst_size,
            *src.shape[1:],
            dtype=src.dtype,
            device=src.device,
        ).index_add_(0, dst_idx, torch.ones_like(src))
        + 1e-8
    )
    return out / denom


def segment_argmin(scores, dst_idx, dst_size, randomize: bool = False) -> torch.Tensor:
    """Samples the index of the minimum value in each segment."""
    if randomize:
        noise = torch.rand_like(scores)
        scores = scores - torch.log(-torch.log(noise))
    _, sampled_idx = scatter_min(scores, dst_idx, dim=0, dim_size=dst_size)
    return sampled_idx


def segment_logsumexp(src, dst_idx, dst_size, extra_dims=None):
    """Computes the logsumexp of each segment in a tensor."""
    src_max, _ = scatter_max(src, dst_idx, dim=0, dim_size=dst_size)
    if extra_dims is not None:
        src_max = torch.amax(src_max, dim=extra_dims, keepdim=True)
    src = src - src_max[dst_idx]
    out = torch.zeros(
        dst_size,
        *src.shape[1:],
        dtype=src.dtype,
        device=src.device,
    ).index_add_(0, dst_idx, torch.exp(src))
    if extra_dims is not None:
        out = torch.sum(out, dim=extra_dims)
    return torch.log(out + 1e-8) + src_max.view(*out.shape)


def segment_softmax(src, dst_idx, dst_size, extra_dims=None, floor_value=None):
    """Computes the softmax of each segment in a tensor."""
    src_max, _ = scatter_max(src, dst_idx, dim=0, dim_size=dst_size)
    if extra_dims is not None:
        src_max = torch.amax(src_max, dim=extra_dims, keepdim=True)
    src = src - src_max[dst_idx]
    exp1 = torch.exp(src)
    exp0 = torch.zeros(
        dst_size,
        *src.shape[1:],
        dtype=src.dtype,
        device=src.device,
    ).index_add_(0, dst_idx, exp1)
    if extra_dims is not None:
        exp0 = torch.sum(exp0, dim=extra_dims, keepdim=True)
    exp0 = torch.index_select(input=exp0, dim=0, index=dst_idx)
    exp = exp1.div(exp0 + 1e-8)
    if floor_value is not None:
        exp = exp.clamp(min=floor_value)
        exp0 = torch.zeros(
            dst_size,
            *src.shape[1:],
            dtype=src.dtype,
            device=src.device,
        ).index_add_(0, dst_idx, exp)
        if extra_dims is not None:
            exp0 = torch.sum(exp0, dim=extra_dims, keepdim=True)
        exp0 = torch.index_select(input=exp0, dim=0, index=dst_idx)
        exp = exp.div(exp0 + 1e-8)
    return exp


def batched_sample_onehot(logits, dim=0, max_only=False):
    """Implements the Gumbel-Max trick to sample from a one-hot distribution."""
    if max_only:
        sampled_idx = torch.argmax(logits, dim=dim, keepdim=True)
    else:
        noise = torch.rand_like(logits)
        sampled_idx = torch.argmax(logits - torch.log(-torch.log(noise)), dim=dim, keepdim=True)
    out_onehot = torch.zeros_like(logits, dtype=torch.bool)
    out_onehot.scatter_(dim=dim, index=sampled_idx, value=1)
    return out_onehot


def topk_edge_mask_from_logits(scores, k, randomize=False):
    """Samples the top-k edges from a set of logits."""
    assert len(scores.shape) == 3, "Scores should have shape [B, N, N]"
    if randomize:
        noise = torch.rand_like(scores)
        scores = scores - torch.log(-torch.log(noise))
    node_degree = min(k, scores.shape[2])
    _, topk_idx = torch.topk(scores, node_degree, dim=-1, largest=True)
    edge_mask = scores.new_zeros(scores.shape, dtype=torch.bool)
    edge_mask = edge_mask.scatter_(dim=2, index=topk_idx, value=1).bool()
    return edge_mask


def sample_inplace_to_torch(sample):
    """Convert NumPy sample to PyTorch tensors."""
    if sample is None:
        return None
    sample["features"] = {k: torch.FloatTensor(v) for k, v in sample["features"].items()}
    sample["indexer"] = {k: torch.LongTensor(v) for k, v in sample["indexer"].items()}
    if "labels" in sample.keys():
        sample["labels"] = {k: torch.FloatTensor(v) for k, v in sample["labels"].items()}
    return sample


def inplace_to_device(sample, device):
    """Move sample to device."""
    sample["features"] = {k: v.to(device) for k, v in sample["features"].items()}
    sample["indexer"] = {k: v.to(device) for k, v in sample["indexer"].items()}
    if "labels" in sample.keys():
        sample["labels"] = sample["labels"].to(device)
    return sample


def inplace_to_torch(sample):
    """Convert NumPy sample to PyTorch tensors."""
    if sample is None:
        return None
    sample["features"] = {k: torch.FloatTensor(v) for k, v in sample["features"].items()}
    sample["indexer"] = {k: torch.LongTensor(v) for k, v in sample["indexer"].items()}
    if "labels" in sample.keys():
        sample["labels"] = {k: torch.FloatTensor(v) for k, v in sample["labels"].items()}
    return sample


def distance_to_gaussian_contact_logits(
    x: torch.Tensor, contact_scale: float, cutoff: Optional[float] = None
) -> torch.Tensor:
    """Convert distance to Gaussian contact logits.

    :param x: Distance tensor.
    :param contact_scale: The contact scale.
    :param cutoff: The distance cutoff.
    :return: Gaussian contact logits.
    """
    if cutoff is None:
        cutoff = contact_scale * 2
    return torch.log(torch.clamp(1 - (x / cutoff), min=1e-9))


def distogram_to_gaussian_contact_logits(
    dgram: torch.Tensor, dist_bins: torch.Tensor, contact_scale: float
) -> torch.Tensor:
    """Convert a distance histogram (distogram) matrix to a Gaussian contact map.

    :param dgram: A distogram matrix.
    :return: A Gaussian contact map.
    """
    return torch.logsumexp(
        dgram + distance_to_gaussian_contact_logits(dist_bins, contact_scale),
        dim=-1,
    )


def eval_true_contact_maps(
    batch: MODEL_BATCH, contact_scale: float, **kwargs: Dict[str, Any]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate true contact maps.

    :param batch: A batch dictionary.
    :param contact_scale: The contact scale.
    :param kwargs: Additional keyword arguments.
    :return: True contact maps.
    """
    indexer = batch["indexer"]
    batch_size = batch["metadata"]["num_structid"]
    with torch.no_grad():
        # Residue centroids
        res_cent_coords = (
            batch["features"]["res_atom_positions"]
            .mul(batch["features"]["res_atom_mask"].bool()[:, :, None])
            .sum(dim=1)
            .div(batch["features"]["res_atom_mask"].bool().sum(dim=1)[:, None] + 1e-9)
        )
        res_lig_dist = (
            res_cent_coords.view(batch_size, -1, 3)[:, :, None]
            - batch["features"]["sdf_coordinates"][indexer["gather_idx_U_u"]].view(
                batch_size, -1, 3
            )[:, None, :]
        ).norm(dim=-1)
        res_lig_contact_logit = distance_to_gaussian_contact_logits(
            res_lig_dist, contact_scale, **kwargs
        )
    return res_lig_dist, res_lig_contact_logit.flatten()


def sample_reslig_contact_matrix(
    batch: MODEL_BATCH, res_lig_logits: torch.Tensor, last: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Sample residue-ligand contact matrix.

    :param batch: A batch dictionary.
    :param res_lig_logits: Residue-ligand contact logits.
    :param last: The last contact matrix.
    :return: Sampled residue-ligand contact matrix.
    """
    metadata = batch["metadata"]
    batch_size = metadata["num_structid"]
    max(metadata["num_molid_per_sample"])
    n_a_per_sample = max(metadata["num_a_per_sample"])
    n_I_per_sample = max(metadata["num_I_per_sample"])
    res_lig_logits = res_lig_logits.view(batch_size, n_a_per_sample, n_I_per_sample)
    # Sampling from unoccupied lattice sites
    if last is None:
        last = torch.zeros_like(res_lig_logits, dtype=torch.bool)
    # Column-graph-wise masking for already sampled ligands
    # sampled_ligand_mask = torch.amax(last, dim=1, keepdim=True)
    sampled_frame_mask = torch.sum(last, dim=1, keepdim=True).contiguous()
    masked_logits = res_lig_logits - sampled_frame_mask * 1e9
    sampled_block_onehot = batched_sample_onehot(masked_logits.flatten(1, 2), dim=1).view(
        batch_size, n_a_per_sample, n_I_per_sample
    )
    new_block_contact_mat = last + sampled_block_onehot
    # Remove non-contact patches
    valid_logit_mask = res_lig_logits > -16.0
    new_block_contact_mat = (new_block_contact_mat * valid_logit_mask).bool()
    return new_block_contact_mat


def merge_res_lig_logits_to_graph(
    batch: MODEL_BATCH,
    res_lig_logits: torch.Tensor,
    single_protein_batch: bool,
) -> torch.Tensor:
    """Patch merging [B, N_res, N_atm] -> [B, N_res, N_graph].

    :param batch: A batch dictionary.
    :param res_lig_logits: Residue-ligand contact logits.
    :param single_protein_batch: Whether to use single protein batch.
    :return: Merged residue-ligand logits.
    """
    assert single_protein_batch, "Only single protein batch is supported."
    metadata = batch["metadata"]
    indexer = batch["indexer"]
    batch_size = metadata["num_structid"]
    max(metadata["num_molid_per_sample"])
    n_mol_per_sample = max(metadata["num_molid_per_sample"])
    n_a_per_sample = max(metadata["num_a_per_sample"])
    n_I_per_sample = max(metadata["num_I_per_sample"])
    res_lig_logits = res_lig_logits.view(batch_size, n_a_per_sample, n_I_per_sample)
    graph_wise_logits = segment_logsumexp(
        res_lig_logits.permute(2, 0, 1),
        indexer["gather_idx_I_molid"][:n_I_per_sample],
        n_mol_per_sample,
    ).permute(1, 2, 0)
    return graph_wise_logits


def sample_res_rowmask_from_contacts(
    batch: MODEL_BATCH,
    res_ligatm_logits: torch.Tensor,
    single_protein_batch: bool,
) -> torch.Tensor:
    """Sample residue row mask from contacts.

    :param batch: A batch dictionary.
    :param res_ligatm_logits: Residue-ligand atom contact logits.
    :return: Sampled residue row mask.
    """
    metadata = batch["metadata"]
    max(metadata["num_molid_per_sample"])
    lig_wise_logits = (
        merge_res_lig_logits_to_graph(batch, res_ligatm_logits, single_protein_batch)
        .permute(0, 2, 1)
        .contiguous()
    )
    sampled_res_onehot_mask = batched_sample_onehot(lig_wise_logits.flatten(0, 1), dim=1)
    return sampled_res_onehot_mask


def extract_esm_embeddings(
    esm_model: torch.nn.Module,
    esm_alphabet: torch.nn.Module,
    esm_batch_converter: torch.nn.Module,
    sequences: List[str],
    device: Union[str, torch.device],
    esm_repr_layer: int = 33,
) -> List[torch.Tensor]:
    """Extract embeddings from ESM model.

    :param esm_model: ESM model.
    :param esm_alphabet: ESM alphabet.
    :param esm_batch_converter: ESM batch converter.
    :param sequences: A list of sequences.
    :param device: Device to use.
    :param esm_repr_layer: ESM representation layer index from which to extract embeddings.
    :return: A corresponding list of embeddings.
    """
    # Disable dropout for deterministic results
    esm_model.eval()

    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    data = [(str(i), seq) for i, seq in enumerate(sequences)]
    _, _, batch_tokens = esm_batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    batch_lens = (batch_tokens != esm_alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[esm_repr_layer], return_contacts=True)
    token_representations = results["representations"][esm_repr_layer]

    # Generate per-residue representations
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1])

    return sequence_representations
