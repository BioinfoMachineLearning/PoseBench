import math

import rootutils
import torch
import torch.nn.functional as F
from beartype.typing import Optional, Tuple, Union
from openfold.model.primitives import Attention
from openfold.utils.tensor_utils import permute_final_dims

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from flowdock.utils.model_utils import GELUMLP, segment_softmax


class MultiHeadAttentionConv(torch.nn.Module):
    """Native Pytorch implementation."""

    def __init__(
        self,
        dim: Union[int, Tuple[int, int]],
        head_dim: int,
        edge_dim: int = None,
        n_heads: int = 1,
        dropout: float = 0.0,
        edge_lin: bool = True,
        **kwargs,
    ):
        """Multi-Head Attention Convolution layer."""
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.edge_lin = edge_lin
        self._alpha = None

        if isinstance(dim, int):
            dim = (dim, dim)

        self.lin_key = torch.nn.Linear(dim[0], n_heads * head_dim, bias=False)
        self.lin_query = torch.nn.Linear(dim[1], n_heads * head_dim, bias=False)
        self.lin_value = torch.nn.Linear(dim[0], n_heads * head_dim, bias=False)
        if edge_lin is True:
            self.lin_edge = torch.nn.Linear(edge_dim, n_heads, bias=False)
        else:
            self.lin_edge = self.register_parameter("lin_edge", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters of the layer."""
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_lin:
            self.lin_edge.reset_parameters()

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
        return_attention_weights=None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass of the Multi-Head Attention Convolution layer.

        :param x (torch.Tensor or Tuple[torch.Tensor, torch.Tensor]): The input features.
        :param edge_index (torch.Tensor): The edge index tensor.
        :param edge_attr (torch.Tensor, optional): The edge attribute tensor.
        :param return_attention_weights (bool, optional): If set to `True`,
            will additionally return the tuple
            `(edge_index, attention_weights)`, holding the computed
            attention weights for each edge. Default is `None`.
        :return: The output features or the tuple
            `(output_features, (edge_index, attention_weights))`.
        """

        H, C = self.n_heads, self.head_dim

        if isinstance(x, torch.Tensor):
            x = (x, x)

        query = self.lin_query(x[1]).view(*x[1].shape[:-1], H, C)
        key = self.lin_key(x[0]).view(*x[0].shape[:-1], H, C)
        value = self.lin_value(x[0]).view(*x[0].shape[:-1], H, C)

        attended_values = self.message(key, query, value, edge_attr, edge_index)
        out = self.aggregate(attended_values, edge_index[1], query.shape[0])

        alpha = self._alpha
        self._alpha = None

        out = out.contiguous().view(*out.shape[:-2], H * C)

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            return out, (edge_index, alpha)
        else:
            return out

    def message(
        self,
        key: torch.Tensor,
        query: torch.Tensor,
        value: torch.Tensor,
        edge_attr: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        """Add the relative positional encodings to attention scores.

        :param key (torch.Tensor): The key tensor.
        :param query (torch.Tensor): The query tensor.
        :param value (torch.Tensor): The value tensor.
        :param edge_attr (torch.Tensor): The edge attribute tensor.
        :param index (torch.Tensor): The edge index tensor.
        :return: The output tensor.
        """
        edge_bias = 0
        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_bias = self.lin_edge(edge_attr)

        _alpha_z = (query[index[1]] * key[index[0]]).sum(dim=-1) / math.sqrt(
            self.head_dim
        ) + edge_bias
        self._alpha = _alpha_z
        alpha = segment_softmax(_alpha_z, index[1], query.shape[0])
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value[index[0]]
        out *= alpha.unsqueeze(-1)
        return out

    def aggregate(
        self, src: torch.Tensor, dst_idx: torch.Tensor, dst_size: torch.Size
    ) -> torch.Tensor:
        """Aggregate the source tensor to the destination tensor.

        :param src (torch.Tensor): The source tensor.
        :param dst_idx (torch.Tensor): The destination index tensor.
        :param dst_size (torch.Size): The destination size tensor.
        :return: The output tensor.
        """
        out = torch.zeros(
            dst_size,
            *src.shape[1:],
            dtype=src.dtype,
            device=src.device,
        ).index_add_(0, dst_idx, src)
        return out


class TransformerLayer(torch.nn.Module):
    """A single layer of a transformer model."""

    def __init__(
        self,
        node_dim: int,
        n_heads: int,
        head_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        bidirectional: bool = False,
        edge_channels: Optional[int] = None,
        dropout: float = 0.0,
        edge_update: bool = False,
    ):
        """Initialize the transformer layer."""
        super().__init__()
        edge_lin = edge_channels is not None
        self.edge_update = edge_update
        if head_dim is None:
            head_dim = node_dim // n_heads
        self.conv = MultiHeadAttentionConv(
            node_dim,
            head_dim,
            edge_dim=edge_channels,
            n_heads=n_heads,
            edge_lin=edge_lin,
            dropout=dropout,
        )
        self.bidirectional = bidirectional
        self.projector = torch.nn.Linear(head_dim * n_heads, node_dim, bias=False)
        self.norm = torch.nn.LayerNorm(node_dim)
        self.mlp = GELUMLP(
            node_dim,
            node_dim,
            n_hidden_feats=hidden_dim,
            dropout=dropout,
            zero_init=True,
        )
        if edge_update:
            self.mlpe = GELUMLP(
                n_heads + edge_channels, edge_channels, dropout=dropout, zero_init=True
            )

    def forward(
        self,
        x_s: torch.Tensor,
        x_a: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass through the transformer layer.

        :param x_s (torch.Tensor): The source node features.
        :param x_a (torch.Tensor): The target node features.
        :param edge_index (torch.Tensor): The edge index tensor. :param edge_attr (torch.Tensor,
            optional): The edge attribute tensor.
        :return: The output source and target node features.
        """
        out_a, (edge_index, alpha) = self.conv(
            (x_s, x_a),
            edge_index,
            edge_attr,
            return_attention_weights=True,
        )
        x_a = x_a + self.projector(out_a)
        x_a = self.mlp(self.norm(x_a)) + x_a
        if self.bidirectional:
            out_s = self.conv((x_a, x_s), (edge_index[1], edge_index[0]), edge_attr)
            x_s = x_s + self.projector(out_s)
            x_s = self.mlp(self.norm(x_s)) + x_s
        if self.edge_update:
            edge_attr = edge_attr + self.mlpe(torch.cat([alpha, edge_attr], dim=-1))
            return x_s, x_a, edge_attr
        else:
            return x_s, x_a


class PointSetAttention(torch.nn.Module):
    """PointSetAttention module."""

    def __init__(
        self,
        fiber_dim: int,
        heads: int = 8,
        point_dim: int = 4,
        edge_dim: Optional[int] = None,
        edge_update: bool = False,
        dropout: float = 0.0,
    ):
        """Initialize the PointSetAttention module."""
        super().__init__()
        self.fiber_dim = fiber_dim
        self.edge_dim = edge_dim
        self.heads = heads
        self.point_dim = point_dim
        self.dropout = dropout
        self.edge_update = edge_update
        self.distance_scaling = 10  # 1 nm

        # num attention contributions
        num_attn_logits = 2

        self.lin_query = torch.nn.Linear(fiber_dim, point_dim * heads, bias=False)
        self.lin_key = torch.nn.Linear(fiber_dim, point_dim * heads, bias=False)
        self.lin_value = torch.nn.Linear(fiber_dim, point_dim * heads, bias=False)
        if edge_dim is not None:
            self.lin_edge = torch.nn.Linear(edge_dim, heads, bias=False)
            if edge_update:
                self.edge_update_mlp = GELUMLP(heads + edge_dim, edge_dim)

        # qkv projection for scalar attention (normal)
        self.scalar_attn_logits_scale = (num_attn_logits * point_dim) ** -0.5

        # qkv projection for point attention (coordinate and orientation aware)
        point_weight_init_value = torch.log(torch.exp(torch.full((heads,), 1.0)) - 1.0)
        self.point_weights = torch.nn.Parameter(point_weight_init_value)

        self.point_attn_logits_scale = ((num_attn_logits * point_dim) * (9 / 2)) ** -0.5
        point_weight_init_value = torch.log(torch.exp(torch.full((heads,), 1.0)) - 1.0)
        self.point_weights = torch.nn.Parameter(point_weight_init_value)

        # combine out - point dim * 4
        self.to_out = torch.nn.Linear(heads * point_dim, fiber_dim, bias=False)

    def forward(
        self,
        x_k: torch.Tensor,
        x_q: torch.Tensor,
        edge_index: torch.LongTensor,
        point_centers_k: torch.Tensor,
        point_centers_q: torch.Tensor,
        x_edge: torch.Tensor = None,
    ):
        """Forward pass of the PointSetAttention module."""
        H, P = self.heads, self.point_dim

        q = self.lin_query(x_q)
        k = self.lin_key(x_k)
        v = self.lin_value(x_k)

        scalar_q = q[..., 0, :].view(-1, H, P)
        scalar_k = k[..., 0, :].view(-1, H, P)
        scalar_v = v[..., 0, :].view(-1, H, P)

        point_q_local = q[..., 1:, :].view(-1, 3, H, P)
        point_k_local = k[..., 1:, :].view(-1, 3, H, P)
        point_v_local = v[..., 1:, :].view(-1, 3, H, P)

        point_q = point_q_local + point_centers_q[..., None, None] / self.distance_scaling
        point_k = point_k_local + point_centers_k[..., None, None] / self.distance_scaling
        point_v = point_v_local + point_centers_k[..., None, None] / self.distance_scaling

        if self.edge_dim is not None:
            edge_bias = self.lin_edge(x_edge)
        else:
            edge_bias = 0

        attn_logits, attentions = self.compute_attention(
            scalar_k, scalar_q, point_k, point_q, edge_bias, edge_index
        )
        res_scalar = self.aggregate(
            attentions[:, :, None] * scalar_v[edge_index[0]],
            edge_index[1],
            scalar_q.shape[0],
        )
        res_points = self.aggregate(
            attentions[:, None, :, None] * point_v[edge_index[0]],
            edge_index[1],
            point_q.shape[0],
        )
        res_points_local = res_points - point_centers_q[..., None, None] / self.distance_scaling

        # [N, H, P], [N, 3, H, P] -> [N, 4, C]
        res = torch.cat([res_scalar.unsqueeze(-3), res_points_local], dim=-3).flatten(-2, -1)
        out = self.to_out(res)  # [N, 4, C]
        if self.edge_update:
            edge_out = self.edge_update_mlp(torch.cat([attn_logits, x_edge], dim=-1))
            return out, edge_out
        return out

    def compute_attention(self, scalar_k, scalar_q, point_k, point_q, edge_bias, index):
        """Compute the attention scores."""
        scalar_q = scalar_q[index[1]]
        scalar_k = scalar_k[index[0]]
        point_q = point_q[index[1]]
        point_k = point_k[index[0]]

        scalar_logits = (scalar_q * scalar_k).sum(dim=-1) * self.scalar_attn_logits_scale
        point_weights = F.softplus(self.point_weights).unsqueeze(0)
        point_logits = (
            torch.square(point_q - point_k).sum(dim=(-3, -1)) * self.point_attn_logits_scale
        )

        logits = scalar_logits - 1 / 2 * point_logits * point_weights + edge_bias
        alpha = segment_softmax(logits, index[1], scalar_q.shape[0])
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return logits, alpha

    @staticmethod
    def aggregate(src, dst_idx, dst_size):
        """Aggregate the source tensor to the destination tensor."""
        out = torch.zeros(
            dst_size,
            *src.shape[1:],
            dtype=src.dtype,
            device=src.device,
        ).index_add_(0, dst_idx, src)
        return out


class BiDirectionalTriangleAttention(torch.nn.Module):
    """
    Adapted from https://github.com/aqlaboratory/openfold
    supports rectangular pair representation tensors
    """

    def __init__(self, c_in: int, c_hidden: int, no_heads: int, inf: float = 1e9):
        """Initialize the Bi-Directional Triangle Attention layer."""
        super().__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf

        self.linear = torch.nn.Linear(c_in, self.no_heads, bias=False)

        self.mha_1 = Attention(self.c_in, self.c_in, self.c_in, self.c_hidden, self.no_heads)
        self.mha_2 = Attention(self.c_in, self.c_in, self.c_in, self.c_hidden, self.no_heads)
        self.layer_norm = torch.nn.LayerNorm(self.c_in)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x_pair: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_lma: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the Bi-Directional Triangle Attention layer."""
        if mask is None:
            # [*, I, J, K]
            mask = x_pair.new_ones(
                x_pair.shape[:-1],
            )

        # [*, I, J, C_in]
        x1 = self.layer_norm(x1)
        # [*, I, K, C_in]
        x2 = self.layer_norm(x2)

        # [*, I, 1, J, K]
        mask_bias = (self.inf * (mask - 1))[..., :, None, :, :]

        # [*, I, H, J, K]
        triangle_bias = permute_final_dims(self.linear(x_pair), [0, 3, 1, 2])

        biases_J2I = [mask_bias, triangle_bias]

        x1_out = self.mha_1(q_x=x1, kv_x=x2, biases=biases_J2I, use_lma=use_lma)
        x1 = x1 + x1_out

        # transpose the triangle bias for I->J attention.
        mask_bias_T_ = mask_bias.transpose(-2, -1).contiguous()
        triangle_bias_T_ = triangle_bias.transpose(-2, -1).contiguous()
        biases_I2J = [mask_bias_T_, triangle_bias_T_]
        x2_out = self.mha_2(q_x=x2, kv_x=x1, biases=biases_I2J, use_lma=use_lma)
        x2 = x2 + x2_out

        return x1, x2
