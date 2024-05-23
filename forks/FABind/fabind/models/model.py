import torch
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch import nn
from torch.nn import Linear
import torch.nn as nn
from models.att_model import EfficientMCAttModel
import torch.nn.functional as F
from utils.utils import get_keepNode_tensor, gumbel_softmax_no_random
import random

class Transition_diff_out_dim(torch.nn.Module):
    # separate left/right edges (block1/block2).
    def __init__(self, embedding_channels=256, out_channels=256, n=4):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.linear1 = Linear(embedding_channels, n*embedding_channels)
        self.linear2 = Linear(n*embedding_channels, out_channels)
        torch.nn.init.xavier_uniform_(self.linear1.weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.linear2.weight, gain=0.001)
    def forward(self, z):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        z = self.layernorm(z)
        z = self.linear2((self.linear1(z)).relu())
        return z

class IaBNet_mean_and_pocket_prediction_cls_coords_dependent(torch.nn.Module):
    def __init__(self, args, embedding_channels=128, pocket_pred_embedding_channels=128):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.args = args
        self.coordinate_scale = args.coordinate_scale
        self.normalize_coord = lambda x: x / self.coordinate_scale
        self.unnormalize_coord = lambda x: x * self.coordinate_scale
        self.stage_prob = args.stage_prob

        
        n_channel = 1 # ligand node has only one coordinate dimension.
        self.complex_model = EfficientMCAttModel(
            args, embedding_channels, embedding_channels, n_channel, n_edge_feats=0, n_layers=args.mean_layers, n_iter=args.n_iter,
            inter_cutoff=args.inter_cutoff, intra_cutoff=args.intra_cutoff, normalize_coord=self.normalize_coord, unnormalize_coord=self.unnormalize_coord,
        )

        self.pocket_pred_model = EfficientMCAttModel(
            args, pocket_pred_embedding_channels, pocket_pred_embedding_channels, n_channel, n_edge_feats=0, n_layers=args.pocket_pred_layers, n_iter=args.pocket_pred_n_iter,
            inter_cutoff=args.inter_cutoff, intra_cutoff=args.intra_cutoff, normalize_coord=self.normalize_coord, unnormalize_coord=self.unnormalize_coord,
        )

        self.protein_to_pocket = Transition_diff_out_dim(embedding_channels=embedding_channels, n=4, out_channels=1)

        # global nodes for protein / compound
        self.glb_c = nn.Parameter(torch.ones(1, embedding_channels))
        self.glb_p = nn.Parameter(torch.ones(1, embedding_channels))
        if args.use_esm2_feat:
            protein_hidden = 1280
        else:
            protein_hidden = 15
        if args.esm2_concat_raw:
            protein_hidden = 1295
        # self.protein_linear = nn.Linear(protein_hidden, embedding_channels) # hard-coded GVP features
        # self.compound_linear = nn.Linear(56, embedding_channels)
        self.protein_linear_whole_protein = nn.Linear(protein_hidden, embedding_channels) # hard-coded GVP features
        self.compound_linear_whole_protein = nn.Linear(56, embedding_channels)

        self.embedding_shrink = nn.Linear(embedding_channels, pocket_pred_embedding_channels)
        self.embedding_enlarge = nn.Linear(pocket_pred_embedding_channels, embedding_channels)
        
        self.distmap_mlp = nn.Sequential(
            nn.Linear(embedding_channels, embedding_channels),
            nn.ReLU(),
            nn.Linear(embedding_channels, 1))
        
        # torch.nn.init.xavier_uniform_(self.protein_linear.weight, gain=0.001)
        # torch.nn.init.xavier_uniform_(self.compound_linear.weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.protein_linear_whole_protein.weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.compound_linear_whole_protein.weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.embedding_shrink.weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.embedding_enlarge.weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.distmap_mlp[0].weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.distmap_mlp[2].weight, gain=0.001)


    def forward(self, data, stage=1, train=False):
        keepNode_less_5 = 0
        compound_batch = data['compound'].batch
        pocket_batch = data['pocket'].batch
        complex_batch = data['complex'].batch
        protein_batch_whole = data['protein_whole'].batch
        complex_batch_whole_protein = data['complex_whole_protein'].batch

        # Pocket Prediction
        # nodes_whole = (data['protein_whole']['node_s'], data['protein_whole']['node_v'])
        # edges_whole = (data[("protein_whole", "p2p", "protein_whole")]["edge_s"], data[("protein_whole", "p2p", "protein_whole")]["edge_v"])
        # protein_out_whole = self.conv_protein(nodes_whole, data[("protein_whole", "p2p", "protein_whole")]["edge_index"], edges_whole, data.seq_whole)
        # protein_out_batched_whole, protein_out_mask_whole = to_dense_batch(protein_out_whole, protein_batch_whole)
        # pocket_cls_pred = self.protein_to_pocket(protein_out_batched_whole)
        # pocket_cls_pred = pocket_cls_pred.squeeze(-1) * protein_out_mask_whole
        # pocket_cls, _ = to_dense_batch(data.pocket_idx, protein_batch_whole)
        batched_complex_coord_whole_protein = self.normalize_coord(data['complex_whole_protein'].node_coords.unsqueeze(-2))
        batched_complex_coord_LAS_whole_protein = self.normalize_coord(data['complex_whole_protein'].node_coords_LAS.unsqueeze(-2))
        batched_compound_emb_whole_protein = self.compound_linear_whole_protein(data['compound'].node_feats)
        batched_protein_emb_whole_protein = self.protein_linear_whole_protein(data['protein_whole'].node_feats)

        # TODO self.glb_c and self.glb_p shared?
        for i in range(complex_batch_whole_protein.max()+1):
            if i == 0:
                new_samples_whole_protein = torch.cat((
                    self.glb_c, batched_compound_emb_whole_protein[compound_batch==i], 
                    self.glb_p, batched_protein_emb_whole_protein[protein_batch_whole==i]
                    ), dim=0)
            else:
                new_sample_whole_protein = torch.cat((
                    self.glb_c, batched_compound_emb_whole_protein[compound_batch==i], 
                    self.glb_p, batched_protein_emb_whole_protein[protein_batch_whole==i]
                    ), dim=0)
                new_samples_whole_protein = torch.cat((new_samples_whole_protein, new_sample_whole_protein), dim=0)

        new_samples_whole_protein = self.embedding_shrink(new_samples_whole_protein)

        complex_coords_whole_protein, complex_out_whole_protein = self.pocket_pred_model(
            batched_complex_coord_whole_protein,
            new_samples_whole_protein, 
            batch_id=complex_batch_whole_protein, 
            segment_id=data['complex_whole_protein'].segment,
            mask=data['complex_whole_protein'].mask, 
            is_global=data['complex_whole_protein'].is_global,
            compound_edge_index=data['complex_whole_protein', 'c2c', 'complex_whole_protein'].edge_index,
            LAS_edge_index=data['complex_whole_protein', 'LAS', 'complex_whole_protein'].edge_index,
            batched_complex_coord_LAS=batched_complex_coord_LAS_whole_protein,
            LAS_mask=None
        )

        complex_out_whole_protein = self.embedding_enlarge(complex_out_whole_protein)

        compound_flag_whole_protein = torch.logical_and(data['complex_whole_protein'].segment == 0, ~data['complex_whole_protein'].is_global)
        compound_out_whole_protein = complex_out_whole_protein[compound_flag_whole_protein]
        protein_flag_whole_protein = torch.logical_and(data['complex_whole_protein'].segment == 1, ~data['complex_whole_protein'].is_global)
        protein_out_whole_protein = complex_out_whole_protein[protein_flag_whole_protein]
        protein_out_batched_whole, protein_out_mask_whole = to_dense_batch(protein_out_whole_protein, protein_batch_whole)
        pocket_cls_pred = self.protein_to_pocket(protein_out_batched_whole)
        pocket_cls_pred = pocket_cls_pred.squeeze(-1) * protein_out_mask_whole
        pocket_cls, _ = to_dense_batch(data.pocket_idx, protein_batch_whole)

        pocket_coords_batched, _ = to_dense_batch(self.normalize_coord(data.node_xyz), pocket_batch)
        protein_coords_batched_whole, protein_coords_mask_whole = to_dense_batch(data.node_xyz_whole, protein_batch_whole)
  
        pred_index_true = pocket_cls_pred.sigmoid().unsqueeze(-1)
        pred_index_false = 1. - pred_index_true
        pred_index_prob = torch.cat([pred_index_false, pred_index_true], dim=-1)
        # For training stability
        pred_index_prob = torch.clamp(pred_index_prob, min=1e-6, max=1-1e-6)
        pred_index_log_prob = torch.log(pred_index_prob)
        if self.pocket_pred_model.training:
            pred_index_one_hot = F.gumbel_softmax(pred_index_log_prob, tau=self.args.gs_tau, hard=self.args.gs_hard)
        else:
            pred_index_one_hot = gumbel_softmax_no_random(pred_index_log_prob, tau=self.args.gs_tau, hard=self.args.gs_hard)
        pred_index_one_hot_true = (pred_index_one_hot[:, :, 1] * protein_out_mask_whole).unsqueeze(-1)
        pred_pocket_center_gumbel = pred_index_one_hot_true * protein_coords_batched_whole
        pred_pocket_center = pred_pocket_center_gumbel.sum(dim=1) / pred_index_one_hot_true.sum(dim=1)

        center_dist_ligand_pocket_batch = torch.norm(data.coords_center - pred_pocket_center, p=2, dim=-1)
        center_dist_mean = center_dist_ligand_pocket_batch.mean(dim=-1)
        if self.pocket_pred_model.training and center_dist_mean < self.args.center_dist_threshold:

            if random.random() < self.stage_prob:
                final_stage = 2
            else:
                final_stage = 1
        elif self.pocket_pred_model.training and center_dist_mean >= self.args.center_dist_threshold:
            final_stage = 1
        else:
            final_stage = stage
            
        if final_stage == 2:
            # Replace raw feature with pocket prediction output
            # batched_compound_emb = self.compound_linear(data['compound'].node_feats)
            batched_compound_emb = compound_out_whole_protein
            # keepNode_batch = torch.tensor([], device=compound_batch.device)
            data['complex'].node_coords = torch.tensor([], device=compound_batch.device)
            data['complex'].node_coords_LAS = torch.tensor([], device=compound_batch.device)
            data['complex'].segment = torch.tensor([], device=compound_batch.device)
            data['complex'].mask = torch.tensor([], device=compound_batch.device)
            data['complex'].is_global = torch.tensor([], device=compound_batch.device)
            complex_batch = torch.tensor([], device=compound_batch.device)
            pocket_batch = torch.tensor([], device=compound_batch.device)
            data['complex', 'c2c', 'complex'].edge_index = torch.tensor([], device=compound_batch.device)
            data['complex', 'LAS', 'complex'].edge_index = torch.tensor([], device=compound_batch.device)
            pocket_coords_concats = torch.tensor([], device=compound_batch.device)
            dis_map = torch.tensor([], device=compound_batch.device)

            if self.args.local_eval:
                pred_pocket_center += self.args.train_pred_pocket_noise * (2 * torch.rand_like(pred_pocket_center) - 1)
            if self.args.train_pred_pocket_noise and train:
                pred_pocket_center += self.args.train_pred_pocket_noise * (2 * torch.rand_like(pred_pocket_center) - 1)

            for i in range(pred_pocket_center.shape[0]):
                protein_i = data.node_xyz_whole[protein_batch_whole==i].detach()
                keepNode = get_keepNode_tensor(protein_i, self.args.pocket_radius, None, pred_pocket_center[i].detach())
                # TODO Check the case
                if keepNode.sum() < 5:
                    # if only include less than 5 residues, simply add first 100 residues.
                    keepNode[:100] = True
                    keepNode_less_5 += 1
                pocket_emb = protein_out_batched_whole[i][protein_out_mask_whole[i]][keepNode]
                # node emb
                if i == 0:
                    new_samples = torch.cat((
                        self.glb_c, batched_compound_emb[compound_batch==i], 
                        self.glb_p, pocket_emb
                        ), dim=0)
                else:
                    new_sample = torch.cat((
                        self.glb_c, batched_compound_emb[compound_batch==i], 
                        self.glb_p, pocket_emb
                        ), dim=0)
                    new_samples = torch.cat((new_samples, new_sample), dim=0)

                # Node coords. 
                # Ligand coords are initialized at pocket center with rdkit random conformation.
                # Pocket coords are from origin protein coords.
                pocket_coords = protein_coords_batched_whole[i][protein_coords_mask_whole[i]][keepNode]
                pocket_coords_concats = torch.cat((pocket_coords_concats, pocket_coords), dim=0)
                
                data['complex'].node_coords = torch.cat( # [glb_c || compound || glb_p || protein]
                    (
                        data['complex'].node_coords,
                        torch.zeros((1, 3), device=compound_batch.device),
                        data['compound'].node_coords[compound_batch==i] - data['compound'].node_coords[compound_batch==i].mean(dim=0).reshape(1, 3) + pocket_coords.mean(dim=0).reshape(1, 3),
                        torch.zeros((1, 3), device=compound_batch.device), 
                        pocket_coords,
                    ), dim=0
                ).float()


                if self.args.compound_coords_init_mode == 'redocking' or self.args.compound_coords_init_mode == 'redocking_no_rotate':
                    data['complex'].node_coords_LAS = torch.cat( # [glb_c || compound || glb_p || protein]
                        (
                            data['complex'].node_coords_LAS,
                            torch.zeros((1, 3), device=compound_batch.device),
                            torch.tensor(data['compound'].node_coords[compound_batch==i]),
                            torch.zeros((1, 3), device=compound_batch.device), 
                            torch.zeros_like(pocket_coords)
                        ), dim=0
                    ).float()
                else:
                    data['complex'].node_coords_LAS = torch.cat( # [glb_c || compound || glb_p || protein]
                        (
                            data['complex'].node_coords_LAS,
                            torch.zeros((1, 3), device=compound_batch.device),
                            data['compound'].rdkit_coords[compound_batch==i],
                            torch.zeros((1, 3), device=compound_batch.device), 
                            torch.zeros_like(pocket_coords)
                        ), dim=0
                    ).float()

                # masks
                n_protein = pocket_emb.shape[0]
                n_compound = batched_compound_emb[compound_batch==i].shape[0]
                segment = torch.zeros((n_protein + n_compound + 2), device=complex_batch.device)
                segment[n_compound+1:] = 1 # compound: 0, protein: 1
                data['complex'].segment = torch.cat((data['complex'].segment, segment), dim=0) # protein or ligand
                mask = torch.zeros((n_protein + n_compound + 2), device=complex_batch.device)
                mask[:n_compound+2] = 1 # glb_p can be updated
                data['complex'].mask = torch.cat((data['complex'].mask, mask.bool()), dim=0)
                is_global = torch.zeros((n_protein + n_compound + 2), device=complex_batch.device)
                is_global[0] = 1
                is_global[n_compound+1] = 1
                data['complex'].is_global = torch.cat((data['complex'].is_global, is_global.bool()), dim=0)

                # edge_index
                data['complex', 'c2c', 'complex'].edge_index = torch.cat(
                    (
                        data['complex', 'c2c', 'complex'].edge_index, 
                        data['compound_atom_edge_list'].x[data['compound_atom_edge_list'].batch==i].t() + complex_batch.shape[0]
                    ), dim=1)
                data['complex', 'LAS', 'complex'].edge_index = torch.cat(
                    (
                        data['complex', 'LAS', 'complex'].edge_index, 
                        data['LAS_edge_list'].x[data['LAS_edge_list'].batch==i].t() + complex_batch.shape[0]
                    ), dim=1)
                
                # batch_id
                complex_batch = torch.cat((complex_batch, torch.ones((n_compound + n_protein + 2), device=compound_batch.device)*i), dim=0)
                pocket_batch = torch.cat((pocket_batch, torch.ones((n_protein), device=compound_batch.device)*i), dim=0)

                # distance map
                dis_map_i = torch.cdist(pocket_coords, data['compound'].node_coords[compound_batch==i].to(torch.float32)).flatten()
                dis_map_i[dis_map_i>10] = 10
                dis_map = torch.cat((dis_map, dis_map_i), dim=0)

            # construct inputs
            batched_complex_coord = self.normalize_coord(data['complex'].node_coords.unsqueeze(-2))
            batched_complex_coord_LAS = self.normalize_coord(data['complex'].node_coords_LAS.unsqueeze(-2))
            complex_batch = complex_batch.to(torch.int64)
            pocket_batch = pocket_batch.to(torch.int64)
            pocket_coords_batched, _ = to_dense_batch(self.normalize_coord(pocket_coords_concats), pocket_batch)
            data['complex', 'c2c', 'complex'].edge_index = data['complex', 'c2c', 'complex'].edge_index.to(torch.int64)
            data['complex', 'LAS', 'complex'].edge_index = data['complex', 'LAS', 'complex'].edge_index.to(torch.int64)
            data['complex'].segment = data['complex'].segment.to(torch.bool)
            data['complex'].mask = data['complex'].mask.to(torch.bool)
            data['complex'].is_global = data['complex'].is_global.to(torch.bool)

        elif final_stage == 1:
            batched_compound_emb = compound_out_whole_protein
            batched_pocket_emb = protein_out_whole_protein[data['pocket'].keepNode]
            batched_complex_coord = self.normalize_coord(data['complex'].node_coords.unsqueeze(-2))
            batched_complex_coord_LAS = self.normalize_coord(data['complex'].node_coords_LAS.unsqueeze(-2))

            for i in range(complex_batch.max()+1):
                if i == 0:
                    new_samples = torch.cat((
                        self.glb_c, batched_compound_emb[compound_batch==i], 
                        self.glb_p, batched_pocket_emb[pocket_batch==i]
                        ), dim=0)
                else:
                    new_sample = torch.cat((
                        self.glb_c, batched_compound_emb[compound_batch==i], 
                        self.glb_p, batched_pocket_emb[pocket_batch==i]
                        ), dim=0)
                    new_samples = torch.cat((new_samples, new_sample), dim=0)
            dis_map = data.dis_map


        complex_coords, complex_out = self.complex_model(
            batched_complex_coord,
            new_samples, 
            batch_id=complex_batch, 
            segment_id=data['complex'].segment,
            mask=data['complex'].mask, 
            is_global=data['complex'].is_global,
            compound_edge_index=data['complex', 'c2c', 'complex'].edge_index,
            LAS_edge_index=data['complex', 'LAS', 'complex'].edge_index,
            batched_complex_coord_LAS=batched_complex_coord_LAS,
            LAS_mask=None
        )

        compound_flag = torch.logical_and(data['complex'].segment == 0, ~data['complex'].is_global)
        protein_flag  = torch.logical_and(data['complex'].segment == 1, ~data['complex'].is_global)
        pocket_out  = complex_out[protein_flag]
        compound_out = complex_out[compound_flag]
        compound_coords_out = complex_coords[compound_flag].squeeze(-2)

        # pocket_batch version could further process b matrix. better than for loop.
        # pocket_out_batched of shape (b, n, c). to_dense_batch is torch geometric function.
        pocket_out_batched, pocket_out_mask = to_dense_batch(pocket_out, pocket_batch)
        compound_out_batched, compound_out_mask = to_dense_batch(compound_out, compound_batch)
        compound_coords_out_batched, compound_coords_out_mask = to_dense_batch(compound_coords_out, compound_batch)

        # get the pair distance of protein and compound
        pocket_com_dis_map = torch.cdist(pocket_coords_batched, compound_coords_out_batched)

        # Assume self.args.distmap_pred == 'mlp':
        pocket_out_batched = self.layernorm(pocket_out_batched)
        compound_out_batched = self.layernorm(compound_out_batched)
        # z of shape, b, protein_length, compound_length, channels.
        z = torch.einsum("bik,bjk->bijk", pocket_out_batched, compound_out_batched)
        z_mask = torch.einsum("bi,bj->bij", pocket_out_mask, compound_out_mask)

        b = self.distmap_mlp(z).squeeze(-1)

        y_pred = b[z_mask]
        y_pred = y_pred.sigmoid() * 10   # normalize to 0 to 10.

        y_pred_by_coords = pocket_com_dis_map[z_mask]
        y_pred_by_coords = self.unnormalize_coord(y_pred_by_coords)
        y_pred_by_coords = torch.clamp(y_pred_by_coords, 0, 10)
        
        compound_coords_out = self.unnormalize_coord(compound_coords_out)
        
        return compound_coords_out, compound_batch, y_pred, y_pred_by_coords, pocket_cls_pred, pocket_cls, protein_out_mask_whole, protein_coords_batched_whole, pred_pocket_center, dis_map, keepNode_less_5

    def inference(self, data):
        compound_batch = data['compound'].batch
        protein_batch_whole = data['protein_whole'].batch
        complex_batch_whole_protein = data['complex_whole_protein'].batch

        # Pocket Prediction
        batched_complex_coord_whole_protein = self.normalize_coord(data['complex_whole_protein'].node_coords.unsqueeze(-2))
        batched_complex_coord_LAS_whole_protein = self.normalize_coord(data['complex_whole_protein'].node_coords_LAS.unsqueeze(-2))
        batched_compound_emb_whole_protein = self.compound_linear_whole_protein(data['compound'].node_feats)
        batched_protein_emb_whole_protein = self.protein_linear_whole_protein(data['protein_whole'].node_feats)

        # TODO self.glb_c and self.glb_p shared?
        for i in range(complex_batch_whole_protein.max()+1):
            if i == 0:
                new_samples_whole_protein = torch.cat((
                    self.glb_c, batched_compound_emb_whole_protein[compound_batch==i], 
                    self.glb_p, batched_protein_emb_whole_protein[protein_batch_whole==i]
                    ), dim=0)
            else:
                new_sample_whole_protein = torch.cat((
                    self.glb_c, batched_compound_emb_whole_protein[compound_batch==i], 
                    self.glb_p, batched_protein_emb_whole_protein[protein_batch_whole==i]
                    ), dim=0)
                new_samples_whole_protein = torch.cat((new_samples_whole_protein, new_sample_whole_protein), dim=0)

        new_samples_whole_protein = self.embedding_shrink(new_samples_whole_protein)

        complex_coords_whole_protein, complex_out_whole_protein = self.pocket_pred_model(
            batched_complex_coord_whole_protein,
            new_samples_whole_protein, 
            batch_id=complex_batch_whole_protein, 
            segment_id=data['complex_whole_protein'].segment,
            mask=data['complex_whole_protein'].mask, 
            is_global=data['complex_whole_protein'].is_global,
            compound_edge_index=data['complex_whole_protein', 'c2c', 'complex_whole_protein'].edge_index,
            LAS_edge_index=data['complex_whole_protein', 'LAS', 'complex_whole_protein'].edge_index,
            batched_complex_coord_LAS=batched_complex_coord_LAS_whole_protein,
            LAS_mask=None
        )

        complex_out_whole_protein = self.embedding_enlarge(complex_out_whole_protein)

        compound_flag_whole_protein = torch.logical_and(data['complex_whole_protein'].segment == 0, ~data['complex_whole_protein'].is_global)
        compound_out_whole_protein = complex_out_whole_protein[compound_flag_whole_protein]
        protein_flag_whole_protein = torch.logical_and(data['complex_whole_protein'].segment == 1, ~data['complex_whole_protein'].is_global)
        protein_out_whole_protein = complex_out_whole_protein[protein_flag_whole_protein]
        protein_out_batched_whole, protein_out_mask_whole = to_dense_batch(protein_out_whole_protein, protein_batch_whole)
        pocket_cls_pred = self.protein_to_pocket(protein_out_batched_whole)
        pocket_cls_pred = pocket_cls_pred.squeeze(-1) * protein_out_mask_whole
        
        protein_coords_batched_whole, protein_coords_mask_whole = to_dense_batch(data.node_xyz_whole, protein_batch_whole)
  
        pred_pocket_center = torch.zeros((pocket_cls_pred.shape[0], 3)).to(pocket_cls_pred.device)
        batch_len = protein_out_mask_whole.sum(dim=1).detach()
        for i, j in enumerate(batch_len):
            pred_index_bool = (pocket_cls_pred.detach()[i][:j].sigmoid().round().int() == 1)
            if pred_index_bool.sum() != 0:
                pred_pocket_center[i] = protein_coords_batched_whole.detach()[i][:j][pred_index_bool].mean(dim=0)
            else: # all the prediction is False, use gumbel soft
                pred_index_true = pocket_cls_pred[i][:j].sigmoid().unsqueeze(-1)
                pred_index_false = 1. - pred_index_true
                pred_index_prob = torch.cat([pred_index_false, pred_index_true], dim=-1)
                pred_index_log_prob = torch.log(pred_index_prob)
                pred_index_one_hot = gumbel_softmax_no_random(pred_index_log_prob, tau=self.args.gs_tau, hard=self.args.gs_hard)
                pred_index_one_hot_true = pred_index_one_hot[:, 1].unsqueeze(-1)
                pred_pocket_center_gumbel = pred_index_one_hot_true * protein_coords_batched_whole[i][:j]
                pred_pocket_center[i] = pred_pocket_center_gumbel.sum(dim=0) / pred_index_one_hot_true.sum(dim=0)
        
        # Replace raw feature with pocket prediction output
        # batched_compound_emb = self.compound_linear(data['compound'].node_feats)
        batched_compound_emb = compound_out_whole_protein
        # keepNode_batch = torch.tensor([], device=compound_batch.device)
        data['complex'].node_coords = torch.tensor([], device=compound_batch.device)
        data['complex'].node_coords_LAS = torch.tensor([], device=compound_batch.device)
        data['complex'].segment = torch.tensor([], device=compound_batch.device)
        data['complex'].mask = torch.tensor([], device=compound_batch.device)
        data['complex'].is_global = torch.tensor([], device=compound_batch.device)
        complex_batch = torch.tensor([], device=compound_batch.device)
        pocket_batch = torch.tensor([], device=compound_batch.device)
        data['complex', 'c2c', 'complex'].edge_index = torch.tensor([], device=compound_batch.device)
        data['complex', 'LAS', 'complex'].edge_index = torch.tensor([], device=compound_batch.device)
        pocket_coords_concats = torch.tensor([], device=compound_batch.device)
        dis_map = torch.tensor([], device=compound_batch.device)
        
        for i in range(pred_pocket_center.shape[0]):
            protein_i = data.node_xyz_whole[protein_batch_whole==i].detach()
            keepNode = get_keepNode_tensor(protein_i, self.args.pocket_radius, None, pred_pocket_center[i].detach())
            # TODO Check the case
            if keepNode.sum() < 5:
                # if only include less than 5 residues, simply add first 100 residues.
                keepNode[:100] = True
            pocket_emb = protein_out_batched_whole[i][protein_out_mask_whole[i]][keepNode]
            # node emb
            if i == 0:
                new_samples = torch.cat((
                    self.glb_c, batched_compound_emb[compound_batch==i], 
                    self.glb_p, pocket_emb
                    ), dim=0)
            else:
                new_sample = torch.cat((
                    self.glb_c, batched_compound_emb[compound_batch==i], 
                    self.glb_p, pocket_emb
                    ), dim=0)
                new_samples = torch.cat((new_samples, new_sample), dim=0)

            # Node coords. 
            # Ligand coords are initialized at pocket center with rdkit random conformation.
            # Pocket coords are from origin protein coords.
            pocket_coords = protein_coords_batched_whole[i][protein_coords_mask_whole[i]][keepNode]
            pocket_coords_concats = torch.cat((pocket_coords_concats, pocket_coords), dim=0)
            
            data['complex'].node_coords = torch.cat( # [glb_c || compound || glb_p || protein]
                (
                    data['complex'].node_coords,
                    torch.zeros((1, 3), device=compound_batch.device),
                    data['compound'].node_coords[compound_batch==i] - data['compound'].node_coords[compound_batch==i].mean(dim=0).reshape(1, 3) + pocket_coords.mean(dim=0).reshape(1, 3),
                    torch.zeros((1, 3), device=compound_batch.device), 
                    pocket_coords,
                ), dim=0
            ).float()


            if self.args.compound_coords_init_mode == 'redocking' or self.args.compound_coords_init_mode == 'redocking_no_rotate':
                data['complex'].node_coords_LAS = torch.cat( # [glb_c || compound || glb_p || protein]
                    (
                        data['complex'].node_coords_LAS,
                        torch.zeros((1, 3), device=compound_batch.device),
                        data['compound'].node_coords[compound_batch==i],
                        torch.zeros((1, 3), device=compound_batch.device), 
                        torch.zeros_like(pocket_coords)
                    ), dim=0
                ).float()
            else:
                data['complex'].node_coords_LAS = torch.cat( # [glb_c || compound || glb_p || protein]
                    (
                        data['complex'].node_coords_LAS,
                        torch.zeros((1, 3), device=compound_batch.device),
                        data['compound'].rdkit_coords[compound_batch==i],
                        torch.zeros((1, 3), device=compound_batch.device), 
                        torch.zeros_like(pocket_coords)
                    ), dim=0
                ).float()

            # masks
            n_protein = pocket_emb.shape[0]
            n_compound = batched_compound_emb[compound_batch==i].shape[0]
            segment = torch.zeros((n_protein + n_compound + 2), device=complex_batch.device)
            segment[n_compound+1:] = 1 # compound: 0, protein: 1
            data['complex'].segment = torch.cat((data['complex'].segment, segment), dim=0) # protein or ligand
            mask = torch.zeros((n_protein + n_compound + 2), device=complex_batch.device)
            mask[:n_compound+2] = 1 # glb_p can be updated
            data['complex'].mask = torch.cat((data['complex'].mask, mask.bool()), dim=0)
            is_global = torch.zeros((n_protein + n_compound + 2), device=complex_batch.device)
            is_global[0] = 1
            is_global[n_compound+1] = 1
            data['complex'].is_global = torch.cat((data['complex'].is_global, is_global.bool()), dim=0)

            # edge_index
            data['complex', 'c2c', 'complex'].edge_index = torch.cat(
                (
                    data['complex', 'c2c', 'complex'].edge_index, 
                    data['compound_atom_edge_list'].x[data['compound_atom_edge_list'].batch==i].t() + complex_batch.shape[0]
                ), dim=1)
            data['complex', 'LAS', 'complex'].edge_index = torch.cat(
                (
                    data['complex', 'LAS', 'complex'].edge_index, 
                    data['LAS_edge_list'].x[data['LAS_edge_list'].batch==i].t() + complex_batch.shape[0]
                ), dim=1)
            
            # batch_id
            complex_batch = torch.cat((complex_batch, torch.ones((n_compound + n_protein + 2), device=compound_batch.device)*i), dim=0)
            pocket_batch = torch.cat((pocket_batch, torch.ones((n_protein), device=compound_batch.device)*i), dim=0)

            # distance map
            dis_map_i = torch.cdist(pocket_coords, data['compound'].node_coords[compound_batch==i].to(torch.float32)).flatten()
            dis_map_i[dis_map_i>10] = 10
            dis_map = torch.cat((dis_map, dis_map_i), dim=0)

        # construct inputs
        batched_complex_coord = self.normalize_coord(data['complex'].node_coords.unsqueeze(-2))
        batched_complex_coord_LAS = self.normalize_coord(data['complex'].node_coords_LAS.unsqueeze(-2))
        complex_batch = complex_batch.to(torch.int64)
        pocket_batch = pocket_batch.to(torch.int64)
        pocket_coords_batched, _ = to_dense_batch(self.normalize_coord(pocket_coords_concats), pocket_batch)
        data['complex', 'c2c', 'complex'].edge_index = data['complex', 'c2c', 'complex'].edge_index.to(torch.int64)
        data['complex', 'LAS', 'complex'].edge_index = data['complex', 'LAS', 'complex'].edge_index.to(torch.int64)
        data['complex'].segment = data['complex'].segment.to(torch.bool)
        data['complex'].mask = data['complex'].mask.to(torch.bool)
        data['complex'].is_global = data['complex'].is_global.to(torch.bool)
        data['complex'].batch = complex_batch


        complex_coords, complex_out = self.complex_model(
            batched_complex_coord,
            new_samples, 
            batch_id=complex_batch, 
            segment_id=data['complex'].segment,
            mask=data['complex'].mask, 
            is_global=data['complex'].is_global,
            compound_edge_index=data['complex', 'c2c', 'complex'].edge_index,
            LAS_edge_index=data['complex', 'LAS', 'complex'].edge_index,
            batched_complex_coord_LAS=batched_complex_coord_LAS,
            LAS_mask=None
        )

        compound_flag = torch.logical_and(data['complex'].segment == 0, ~data['complex'].is_global)
        compound_coords_out = complex_coords[compound_flag].squeeze(-2)
        compound_coords_out = self.unnormalize_coord(compound_coords_out)
        
        return compound_coords_out, compound_batch

def get_model(args, logger, device):
    if args.mode == 5:
        logger.log_message("FABind")
        model = IaBNet_mean_and_pocket_prediction_cls_coords_dependent(args, args.hidden_size, args.pocket_pred_hidden_size)
    return model
