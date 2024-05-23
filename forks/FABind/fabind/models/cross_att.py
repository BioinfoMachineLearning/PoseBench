import torch
from torch import nn
from torch.nn import LayerNorm, Linear

from models.model_utils import permute_final_dims, Attention, Transition, InteractionModule

class CrossAttentionModule(nn.Module):
    def __init__(self, node_hidden_dim, pair_hidden_dim, rm_layernorm=False, keep_trig_attn=False, dist_hidden_dim=32, normalize_coord=None):
        super().__init__()
        self.pair_hidden_dim = pair_hidden_dim
        self.keep_trig_attn = keep_trig_attn

        if keep_trig_attn:
            self.triangle_block_row = RowTriangleAttentionBlock(pair_hidden_dim, dist_hidden_dim, rm_layernorm=rm_layernorm)
            self.triangle_block_column = RowTriangleAttentionBlock(pair_hidden_dim, dist_hidden_dim, rm_layernorm=rm_layernorm)

        self.p_attention_block = RowAttentionBlock(node_hidden_dim, pair_hidden_dim, rm_layernorm=rm_layernorm)
        self.c_attention_block = RowAttentionBlock(node_hidden_dim, pair_hidden_dim, rm_layernorm=rm_layernorm)
        self.p_transition = Transition(node_hidden_dim, 2, rm_layernorm=rm_layernorm)
        self.c_transition = Transition(node_hidden_dim, 2, rm_layernorm=rm_layernorm)
        self.pair_transition = Transition(pair_hidden_dim, 2, rm_layernorm=rm_layernorm)
        self.inter_layer = InteractionModule(node_hidden_dim, pair_hidden_dim, 32, opm=False, rm_layernorm=rm_layernorm)

    def forward(self, 
                p_embed_batched, p_mask,
                c_embed_batched, c_mask,
                pair_embed, pair_mask,
                c_c_dist_embed=None, p_p_dist_embed=None):
    
        if self.keep_trig_attn:
            pair_embed = self.triangle_block_row(pair_embed=pair_embed,
                                                pair_mask=pair_mask,
                                                dist_embed=c_c_dist_embed)
            pair_embed = self.triangle_block_row(pair_embed=pair_embed.transpose(-2, -3),
                                                pair_mask=pair_mask.transpose(-1, -2),
                                                dist_embed=p_p_dist_embed).transpose(-2, -3)
    
        p_embed_batched = self.p_attention_block(node_embed_i=p_embed_batched,
                                                 node_embed_j=c_embed_batched,
                                                 pair_embed=pair_embed,
                                                 pair_mask=pair_mask,
                                                 node_mask_i=p_mask)
        c_embed_batched = self.c_attention_block(node_embed_i=c_embed_batched,
                                                 node_embed_j=p_embed_batched,
                                                 pair_embed=pair_embed.transpose(-2, -3),
                                                 pair_mask=pair_mask.transpose(-1, -2),
                                                 node_mask_i=c_mask)
        p_embed_batched = p_embed_batched + self.p_transition(p_embed_batched) 
        c_embed_batched = c_embed_batched + self.c_transition(c_embed_batched)

        pair_embed = pair_embed + self.inter_layer(p_embed_batched, c_embed_batched, p_mask, c_mask)[0]

        pair_embed = self.pair_transition(pair_embed) * pair_mask.to(torch.float).unsqueeze(-1)
        return p_embed_batched, c_embed_batched, pair_embed


class RowTriangleAttentionBlock(nn.Module):
    inf = 1e9

    def __init__(self, pair_hidden_dim, dist_hidden_dim, attention_hidden_dim=32, no_heads=4, dropout=0.1, rm_layernorm=False):
        super(RowTriangleAttentionBlock, self).__init__()
        self.no_heads = no_heads
        self.attention_hidden_dim = attention_hidden_dim
        self.dist_hidden_dim = dist_hidden_dim
        self.pair_hidden_dim = pair_hidden_dim

        self.rm_layernorm = rm_layernorm
        if not self.rm_layernorm:
            self.layernorm = LayerNorm(pair_hidden_dim)

        self.linear = Linear(dist_hidden_dim, self.no_heads)
        self.linear_g = Linear(dist_hidden_dim, self.no_heads)
        self.dropout = nn.Dropout(dropout)
        self.mha = Attention(
            pair_hidden_dim, pair_hidden_dim, pair_hidden_dim, attention_hidden_dim, no_heads
        )

    def forward(self, pair_embed, pair_mask, dist_embed):
        if not self.rm_layernorm:
            pair_embed = self.layernorm(pair_embed)  # (*, I, J, C_pair)

        mask_bias = (self.inf * (pair_mask.to(torch.float) - 1))[..., :, None, None, :]  # (*, I, 1, 1, J)
        dist_bias = self.linear(dist_embed) * self.linear_g(dist_embed).sigmoid()  # (*, J, J, H)
        dist_bias = permute_final_dims(dist_bias, [2, 1, 0])[..., None, :, :, :]  # (*, 1, H, J, J)

        pair_embed = pair_embed + self.dropout(self.mha(
            q_x=pair_embed,  # [*, I, J, C_pair]
            kv_x=pair_embed,  # [*, I, J, C_pair]
            biases=[mask_bias, dist_bias]  # List of [*, I, H, J, J]
        )) * pair_mask.to(torch.float).unsqueeze(-1)  # (*, I, J, C_pair)

        return pair_embed


class RowAttentionBlock(nn.Module):
    inf = 1e9

    def __init__(self, node_hidden_dim, pair_hidden_dim, attention_hidden_dim=32, no_heads=4, dropout=0.1, rm_layernorm=False):
        super(RowAttentionBlock, self).__init__()
        self.no_heads = no_heads
        self.attention_hidden_dim = attention_hidden_dim
        self.pair_hidden_dim = pair_hidden_dim
        self.node_hidden_dim = node_hidden_dim

        self.rm_layernorm = rm_layernorm
        if not self.rm_layernorm:
            self.layernorm_node_i = LayerNorm(node_hidden_dim)
            self.layernorm_node_j = LayerNorm(node_hidden_dim)
            self.layernorm_pair = LayerNorm(pair_hidden_dim)

        self.linear = Linear(pair_hidden_dim, self.no_heads)
        self.linear_g = Linear(pair_hidden_dim, self.no_heads)

        self.dropout = nn.Dropout(dropout)

        self.mha = Attention(node_hidden_dim, node_hidden_dim, node_hidden_dim, attention_hidden_dim, no_heads)

    def forward(self, node_embed_i, node_embed_j, pair_embed, pair_mask, node_mask_i):
        if not self.rm_layernorm:
            node_embed_i = self.layernorm_node_i(node_embed_i)  # (*, I, C_node)
            node_embed_j = self.layernorm_node_j(node_embed_j)  # (*, J, C_node)
            pair_embed = self.layernorm_pair(pair_embed)  # (*, I, J, C_pair)

        mask_bias = (self.inf * (pair_mask.to(torch.float) - 1))[..., None, :, :]  # (*, 1, I, J)
        pair_bias = self.linear(pair_embed) * self.linear_g(pair_embed).sigmoid()  # (*, I, J, H)
        pair_bias = permute_final_dims(pair_bias, [2, 0, 1])  # (*, H, I, J)

        node_embed_i = node_embed_i + self.dropout(self.mha(
            q_x=node_embed_i,
            kv_x=node_embed_j,
            biases=[mask_bias, pair_bias]
        )) * node_mask_i.to(torch.float).unsqueeze(-1)

        return node_embed_i
