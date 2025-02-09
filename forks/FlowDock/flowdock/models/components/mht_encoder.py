# Adapted from: https://github.com/zrqiao/NeuralPLexer

import rootutils
import torch
from beartype.typing import Any, Dict, Optional, Tuple
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from flowdock.models.components.hetero_graph import make_multi_relation_graph_batcher
from flowdock.models.components.modules import TransformerLayer
from flowdock.utils import RankedLogger
from flowdock.utils.model_utils import GELUMLP, segment_softmax, segment_sum

MODEL_BATCH = Dict[str, Any]
STATE_DICT = Dict[str, Any]

log = RankedLogger(__name__, rank_zero_only=True)


class PathConvStack(torch.nn.Module):
    """Path integral convolution stack for ligand encoding."""

    def __init__(
        self,
        pair_channels: int,
        n_heads: int = 8,
        max_pi_length: int = 8,
        dropout: float = 0.0,
    ):
        """Initialize PathConvStack model."""
        super().__init__()
        self.pair_channels = pair_channels
        self.max_pi_length = max_pi_length
        self.n_heads = n_heads

        self.prop_value_layer = torch.nn.Linear(pair_channels, n_heads, bias=False)
        self.triangle_pair_kernel_layer = torch.nn.Linear(pair_channels, n_heads, bias=False)
        self.prop_update_mlp = GELUMLP(
            n_heads * (max_pi_length + 1), pair_channels, dropout=dropout
        )

    def forward(
        self,
        prop_attr: torch.Tensor,
        stereo_attr: torch.Tensor,
        indexer: Dict[str, torch.LongTensor],
        metadata: Dict[str, Any],
    ) -> torch.Tensor:
        """Forward pass for PathConvStack model.

        :param prop_attr: Atom-frame pair attributes.
        :param stereo_attr: Stereochemistry attributes.
        :param indexer: A dictionary of indices.
        :param metadata: A dictionary of metadata.
        :return: Updated atom-frame pair attributes.
        """
        triangle_pair_kernel = self.triangle_pair_kernel_layer(stereo_attr)
        # Segment-wise softmax, normalized by outgoing triangles
        triangle_pair_alpha = segment_softmax(
            triangle_pair_kernel, indexer["gather_idx_ijkl_jkl"], metadata["num_ijk"]
        )  # .div(self.max_pi_length)
        # Uijk,ijkl->ujkl pair representation update
        kernel = triangle_pair_alpha[indexer["gather_idx_Uijkl_ijkl"]]
        out_prop_attr = [self.prop_value_layer(prop_attr)]
        for _ in range(self.max_pi_length):
            gathered_prop_attr = out_prop_attr[-1][indexer["gather_idx_Uijkl_Uijk"]]
            out_prop_attr.append(
                segment_sum(
                    kernel.mul(gathered_prop_attr),
                    indexer["gather_idx_Uijkl_ujkl"],
                    metadata["num_Uijk"],
                )
            )
        new_prop_attr = torch.cat(out_prop_attr, dim=-1)
        new_prop_attr = self.prop_update_mlp(new_prop_attr) + prop_attr
        return new_prop_attr


class PIFormer(torch.nn.Module):
    """PIFormer model for ligand encoding."""

    def __init__(
        self,
        node_channels: int,
        pair_channels: int,
        n_atom_encodings: int,
        n_bond_encodings: int,
        n_atom_pos_encodings: int,
        n_stereo_encodings: int,
        heads: int,
        head_dim: int,
        max_path_length: int = 4,
        n_transformer_stacks=4,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """Initialize PIFormer model."""
        super().__init__()
        self.node_channels = node_channels
        self.pair_channels = pair_channels
        self.max_pi_length = max_path_length
        self.n_transformer_stacks = n_transformer_stacks
        self.n_atom_encodings = n_atom_encodings
        self.n_bond_encodings = n_bond_encodings
        self.n_atom_pair_encodings = n_bond_encodings + 4
        self.n_atom_pos_encodings = n_atom_pos_encodings

        self.input_atom_layer = torch.nn.Linear(n_atom_encodings, node_channels)
        self.input_pair_layer = torch.nn.Linear(self.n_atom_pair_encodings, pair_channels)
        self.input_stereo_layer = torch.nn.Linear(n_stereo_encodings, pair_channels)
        self.input_prop_layer = GELUMLP(
            self.n_atom_pair_encodings * 3,
            pair_channels,
        )
        self.path_integral_stacks = torch.nn.ModuleList(
            [
                PathConvStack(
                    pair_channels,
                    max_pi_length=max_path_length,
                    dropout=dropout,
                )
                for _ in range(n_transformer_stacks)
            ]
        )
        self.graph_transformer_stacks = torch.nn.ModuleList(
            [
                TransformerLayer(
                    node_channels,
                    heads,
                    head_dim=head_dim,
                    edge_channels=pair_channels,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    edge_update=True,
                )
                for _ in range(n_transformer_stacks)
            ]
        )

    def forward(self, batch: MODEL_BATCH, masking_rate: float = 0.0) -> MODEL_BATCH:
        """Forward pass for PIFormer model.

        :param batch: A batch dictionary.
        :param masking_rate: Masking rate.
        :return: A batch dictionary.
        """
        features = batch["features"]
        indexer = batch["indexer"]
        metadata = batch["metadata"]
        features["atom_encodings"] = features["atom_encodings"]
        atom_attr = features["atom_encodings"]
        atom_pair_attr = features["atom_pair_encodings"]
        af_pair_attr = features["atom_frame_pair_encodings"]
        stereo_enc = features["stereo_chemistry_encodings"]
        batch["features"]["lig_atom_token"] = atom_attr.detach().clone()
        batch["features"]["lig_pair_token"] = atom_pair_attr.detach().clone()

        atom_mask = torch.rand(atom_attr.shape[0], device=atom_attr.device) > masking_rate
        stereo_mask = torch.rand(stereo_enc.shape[0], device=stereo_enc.device) > masking_rate
        atom_pair_mask = (
            torch.rand(atom_pair_attr.shape[0], device=atom_pair_attr.device) > masking_rate
        )
        af_pair_mask = (
            torch.rand(af_pair_attr.shape[0], device=atom_pair_attr.device) > masking_rate
        )
        atom_attr = atom_attr * atom_mask[:, None]
        stereo_enc = stereo_enc * stereo_mask[:, None]
        atom_pair_attr = atom_pair_attr * atom_pair_mask[:, None]
        af_pair_attr = af_pair_attr * af_pair_mask[:, None]

        # Embedding blocks
        metadata["num_atom"] = metadata["num_u"]
        metadata["num_frame"] = metadata["num_ijk"]
        atom_attr = self.input_atom_layer(atom_attr)
        atom_pair_attr = self.input_pair_layer(atom_pair_attr)
        triangle_attr = atom_attr.new_zeros(metadata["num_frame"], self.node_channels)
        # Initialize atom-frame pair attributes. Reusing uv indices
        prop_attr = self.input_prop_layer(af_pair_attr)
        stereo_attr = self.input_stereo_layer(stereo_enc)

        graph_relations = [
            ("atom_to_atom", "gather_idx_uv_u", "gather_idx_uv_v", "atom", "atom"),
            (
                "atom_to_frame",
                "gather_idx_Uijk_u",
                "gather_idx_Uijk_ijk",
                "atom",
                "frame",
            ),
            (
                "frame_to_atom",
                "gather_idx_Uijk_ijk",
                "gather_idx_Uijk_u",
                "frame",
                "atom",
            ),
            (
                "frame_to_frame",
                "gather_idx_ijkl_ijk",
                "gather_idx_ijkl_jkl",
                "frame",
                "frame",
            ),
        ]

        graph_batcher = make_multi_relation_graph_batcher(graph_relations, indexer, metadata)
        merged_edge_idx = graph_batcher.collate_idx_list(indexer)
        node_reps = {"atom": atom_attr, "frame": triangle_attr}
        edge_reps = {
            "atom_to_atom": atom_pair_attr,
            "atom_to_frame": prop_attr,
            "frame_to_atom": prop_attr,
            "frame_to_frame": stereo_attr,
        }

        # Graph path integral recursion
        for block_id in range(self.n_transformer_stacks):
            merged_node_attr = graph_batcher.collate_node_attr(node_reps)
            merged_edge_attr = graph_batcher.collate_edge_attr(edge_reps)
            _, merged_node_attr, merged_edge_attr = self.graph_transformer_stacks[block_id](
                merged_node_attr,
                merged_node_attr,
                merged_edge_idx,
                merged_edge_attr,
            )
            node_reps = graph_batcher.offload_node_attr(merged_node_attr)
            edge_reps = graph_batcher.offload_edge_attr(merged_edge_attr)
            prop_attr = edge_reps["atom_to_frame"]
            stereo_attr = edge_reps["frame_to_frame"]
            prop_attr = prop_attr + self.path_integral_stacks[block_id](
                prop_attr,
                stereo_attr,
                indexer,
                metadata,
            )
            edge_reps["atom_to_frame"] = prop_attr

        node_reps["sampled_frame"] = node_reps["frame"][indexer["gather_idx_I_ijk"]]

        batch["metadata"]["num_lig_atm"] = metadata["num_u"]
        batch["metadata"]["num_lig_trp"] = metadata["num_I"]

        batch["features"]["lig_atom_attr"] = node_reps["atom"]
        # Downsampled ligand frames
        batch["features"]["lig_trp_attr"] = node_reps["sampled_frame"]
        batch["features"]["lig_atom_pair_attr"] = edge_reps["atom_to_atom"]
        batch["features"]["lig_prop_attr"] = edge_reps["atom_to_frame"]
        edge_reps["sampled_atom_to_sampled_frame"] = edge_reps["atom_to_frame"][
            indexer["gather_idx_UI_Uijk"]
        ]
        batch["features"]["lig_af_pair_attr"] = edge_reps["sampled_atom_to_sampled_frame"]
        return batch


def resolve_ligand_encoder(
    ligand_model_cfg: DictConfig,
    task_cfg: DictConfig,
    state_dict: Optional[STATE_DICT] = None,
) -> torch.nn.Module:
    """Instantiates a PIFormer model for ligand encoding.

    :param ligand_model_cfg: Ligand model configuration.
    :param task_cfg: Task configuration.
    :param state_dict: Optional (potentially-pretrained) state dictionary.
    :return: Ligand encoder model.
    """
    model = PIFormer(
        ligand_model_cfg.node_channels,
        ligand_model_cfg.pair_channels,
        ligand_model_cfg.n_atom_encodings,
        ligand_model_cfg.n_bond_encodings,
        ligand_model_cfg.n_atom_pos_encodings,
        ligand_model_cfg.n_stereo_encodings,
        ligand_model_cfg.n_attention_heads,
        ligand_model_cfg.attention_head_dim,
        hidden_dim=ligand_model_cfg.hidden_dim,
        max_path_length=ligand_model_cfg.max_path_integral_length,
        n_transformer_stacks=ligand_model_cfg.n_transformer_stacks,
        dropout=task_cfg.dropout,
    )
    if ligand_model_cfg.from_pretrained and state_dict is not None:
        try:
            model.load_state_dict(
                {
                    ".".join(k.split(".")[1:]): v
                    for k, v in state_dict.items()
                    if k.startswith("ligand_encoder")
                }
            )
            log.info(
                "Successfully loaded pretrained ligand Molecular Heat Transformer (MHT) weights."
            )
        except Exception as e:
            log.warning(
                f"Skipping loading of pretrained ligand Molecular Heat Transformer (MHT) weights due to: {e}."
            )
    return model


def resolve_relational_reasoning_module(
    protein_model_cfg: DictConfig,
    ligand_model_cfg: DictConfig,
    relational_reasoning_cfg: DictConfig,
    state_dict: Optional[STATE_DICT] = None,
) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    """Instantiates relational reasoning module for ligand encoding.

    :param protein_model_cfg: Protein model configuration.
    :param ligand_model_cfg: Ligand model configuration.
    :param relational_reasoning_cfg: Relational reasoning configuration.
    :param state_dict: Optional (potentially-pretrained) state dictionary.
    :return: Relational reasoning modules for ligand encoding.
    """
    molgraph_single_projector = torch.nn.Linear(
        ligand_model_cfg.node_channels, protein_model_cfg.residue_dim, bias=False
    )
    molgraph_pair_projector = torch.nn.Linear(
        ligand_model_cfg.pair_channels, protein_model_cfg.pair_dim, bias=False
    )
    covalent_embed = torch.nn.Embedding(2, protein_model_cfg.pair_dim)
    if relational_reasoning_cfg.from_pretrained and state_dict is not None:
        try:
            molgraph_single_projector.load_state_dict(
                {
                    ".".join(k.split(".")[1:]): v
                    for k, v in state_dict.items()
                    if k.startswith("molgraph_single_projector")
                }
            )
            log.info("Successfully loaded pretrained ligand graph single projector weights.")
        except Exception as e:
            log.warning(
                f"Skipping loading of pretrained ligand graph single projector weights due to: {e}."
            )

        try:
            molgraph_pair_projector.load_state_dict(
                {
                    ".".join(k.split(".")[1:]): v
                    for k, v in state_dict.items()
                    if k.startswith("molgraph_pair_projector")
                }
            )
            log.info("Successfully loaded pretrained ligand graph pair projector weights.")
        except Exception as e:
            log.warning(
                f"Skipping loading of pretrained ligand graph pair projector weights due to: {e}."
            )

        try:
            covalent_embed.load_state_dict(
                {
                    ".".join(k.split(".")[1:]): v
                    for k, v in state_dict.items()
                    if k.startswith("covalent_embed")
                }
            )
            log.info("Successfully loaded pretrained ligand covalent embedding weights.")
        except Exception as e:
            log.warning(
                f"Skipping loading of pretrained ligand covalent embedding weights due to: {e}."
            )
    return molgraph_single_projector, molgraph_pair_projector, covalent_embed
