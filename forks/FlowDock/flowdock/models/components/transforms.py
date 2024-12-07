# Adapted from: https://github.com/zrqiao/NeuralPLexer

import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from flowdock.utils.model_utils import segment_mean


class LatentCoordinateConverter:
    """Transform the batched feature dict to latent coordinate arrays."""

    def __init__(self, config, prot_atom37_namemap, lig_namemap):
        """Initialize the converter."""
        super().__init__()
        self.config = config
        self.prot_namemap = prot_atom37_namemap
        self.lig_namemap = lig_namemap
        self.cached_noise = None
        self._last_pred_ca_trace = None

    @staticmethod
    def nested_get(dic, keys):
        """Get the value in the nested dictionary."""
        for key in keys:
            dic = dic[key]
        return dic

    @staticmethod
    def nested_set(dic, keys, value):
        """Set the value in the nested dictionary."""
        for key in keys[:-1]:
            dic = dic.setdefault(key, {})
        dic[keys[-1]] = value

    def to_latent(self, batch):
        """Convert the batched feature dict to latent coordinates."""
        return None

    def assign_to_batch(self, batch, x_int):
        """Assign the latent coordinates to the batched feature dict."""
        return None


class DefaultPLCoordinateConverter(LatentCoordinateConverter):
    """Minimal conversion, using internal coords for sidechains and global coords for others."""

    def __init__(self, config, prot_atom37_namemap, lig_namemap):
        """Initialize the converter."""
        super().__init__(config, prot_atom37_namemap, lig_namemap)
        # Scale parameters in Angstrom
        self.ca_scale = config.global_max_sigma
        self.other_scale = config.internal_max_sigma

    def to_latent(self, batch: dict):
        """Convert the batched feature dict to latent coordinates."""
        indexer = batch["indexer"]
        metadata = batch["metadata"]
        self._batch_size = metadata["num_structid"]
        atom37_mask = batch["features"]["res_atom_mask"].bool()
        self._cother_mask = atom37_mask.clone()
        self._cother_mask[:, 1] = False
        atom37_coords = self.nested_get(batch, self.prot_namemap[0])
        try:
            apo_available = True
            apo_atom37_coords = self.nested_get(
                batch, self.prot_namemap[0][:-1] + ("apo_" + self.prot_namemap[0][-1],)
            )
        except KeyError:
            apo_available = False
            apo_atom37_coords = torch.zeros_like(atom37_coords)
        ca_atom_centroid_coords = segment_mean(
            # NOTE: in contrast to NeuralPLexer, we center all coordinates at the origin using the Ca atom centroids
            atom37_coords[:, 1],
            indexer["gather_idx_a_structid"],
            self._batch_size,
        )
        if apo_available:
            apo_ca_atom_centroid_coords = segment_mean(
                apo_atom37_coords[:, 1],
                indexer["gather_idx_a_structid"],
                self._batch_size,
            )
        else:
            apo_ca_atom_centroid_coords = torch.zeros_like(ca_atom_centroid_coords)
        ca_coords_glob = (
            (atom37_coords[:, 1] - ca_atom_centroid_coords[indexer["gather_idx_a_structid"]])
            .contiguous()
            .view(self._batch_size, -1, 3)
        )
        if apo_available:
            apo_ca_coords_glob = (
                (
                    apo_atom37_coords[:, 1]
                    - apo_ca_atom_centroid_coords[indexer["gather_idx_a_structid"]]
                )
                .contiguous()
                .view(self._batch_size, -1, 3)
            )
        else:
            apo_ca_coords_glob = torch.zeros_like(ca_coords_glob)
        cother_coords_int = (
            (atom37_coords - ca_atom_centroid_coords[indexer["gather_idx_a_structid"], None])[
                self._cother_mask
            ]
            .contiguous()
            .view(self._batch_size, -1, 3)
        )
        if apo_available:
            apo_cother_coords_int = (
                (
                    apo_atom37_coords
                    - apo_ca_atom_centroid_coords[indexer["gather_idx_a_structid"], None]
                )[self._cother_mask]
                .contiguous()
                .view(self._batch_size, -1, 3)
            )
        else:
            apo_cother_coords_int = torch.zeros_like(cother_coords_int)
        self._n_res_per_sample = ca_coords_glob.shape[1]
        self._n_cother_per_sample = cother_coords_int.shape[1]
        if batch["misc"]["protein_only"]:
            self._n_ligha_per_sample = 0
            x_int = torch.cat(
                [
                    ca_coords_glob / self.ca_scale,
                    apo_ca_coords_glob / self.ca_scale,
                    cother_coords_int / self.other_scale,
                    apo_cother_coords_int / self.other_scale,
                ],
                dim=1,
            )
            return x_int
        lig_ha_coords = self.nested_get(batch, self.lig_namemap[0])
        lig_ha_coords_int = (
            lig_ha_coords - ca_atom_centroid_coords[indexer["gather_idx_i_structid"]]
        )
        lig_ha_coords_int = lig_ha_coords_int.contiguous().view(self._batch_size, -1, 3)
        ca_atom_centroid_coords = ca_atom_centroid_coords.contiguous().view(
            self._batch_size, -1, 3
        )
        apo_ca_atom_centroid_coords = apo_ca_atom_centroid_coords.contiguous().view(
            self._batch_size, -1, 3
        )
        x_int = torch.cat(
            [
                ca_coords_glob / self.ca_scale,
                apo_ca_coords_glob / self.ca_scale,
                cother_coords_int / self.other_scale,
                apo_cother_coords_int / self.other_scale,
                ca_atom_centroid_coords / self.ca_scale,
                apo_ca_atom_centroid_coords / self.ca_scale,
                lig_ha_coords_int / self.other_scale,
            ],
            dim=1,
        )
        # NOTE: since we use the Ca atom centroids for centralization, we have only one molid per sample
        self._n_molid_per_sample = ca_atom_centroid_coords.shape[1]
        self._n_ligha_per_sample = lig_ha_coords_int.shape[1]
        return x_int

    def assign_to_batch(self, batch: dict, x_lat: torch.Tensor):
        """Assign the latent coordinates to the batched feature dict."""
        indexer = batch["indexer"]
        new_atom37_coords = x_lat.new_zeros(self._batch_size * self._n_res_per_sample, 37, 3)
        apo_new_atom37_coords = x_lat.new_zeros(self._batch_size * self._n_res_per_sample, 37, 3)
        if batch["misc"]["protein_only"]:
            ca_lat, apo_ca_lat, cother_lat, apo_cother_lat = torch.split(
                x_lat,
                [
                    self._n_res_per_sample,
                    self._n_res_per_sample,
                    self._n_cother_per_sample,
                    self._n_cother_per_sample,
                ],
                dim=1,
            )
        else:
            (
                ca_lat,
                apo_ca_lat,
                cother_lat,
                apo_cother_lat,
                ca_cent_lat,
                _,
                lig_lat,
            ) = torch.split(
                x_lat,
                [
                    self._n_res_per_sample,
                    self._n_res_per_sample,
                    self._n_cother_per_sample,
                    self._n_cother_per_sample,
                    self._n_molid_per_sample,
                    self._n_molid_per_sample,
                    self._n_ligha_per_sample,
                ],
                dim=1,
            )
        new_ca_glob = (ca_lat * self.ca_scale).contiguous().flatten(0, 1)
        apo_new_ca_glob = (apo_ca_lat * self.ca_scale).contiguous().flatten(0, 1)
        new_atom37_coords[self._cother_mask] = (
            (cother_lat * self.other_scale).contiguous().flatten(0, 1)
        )
        apo_new_atom37_coords[self._cother_mask] = (
            (apo_cother_lat * self.other_scale).contiguous().flatten(0, 1)
        )
        new_atom37_coords = new_atom37_coords
        apo_new_atom37_coords = apo_new_atom37_coords
        new_atom37_coords[~self._cother_mask] = 0
        apo_new_atom37_coords[~self._cother_mask] = 0
        new_atom37_coords[:, 1] = new_ca_glob
        apo_new_atom37_coords[:, 1] = apo_new_ca_glob
        self.nested_set(batch, self.prot_namemap[1], new_atom37_coords)
        self.nested_set(
            batch,
            self.prot_namemap[1][:-1] + ("apo_" + self.prot_namemap[1][-1],),
            apo_new_atom37_coords,
        )
        if batch["misc"]["protein_only"]:
            self.nested_set(batch, self.lig_namemap[1], None)
            self.empty_cache()
            return batch
        new_ligha_coords_int = (lig_lat * self.other_scale).contiguous().flatten(0, 1)
        new_ligha_coords_cent = (ca_cent_lat * self.ca_scale).contiguous().flatten(0, 1)
        new_ligha_coords = (
            new_ligha_coords_int + new_ligha_coords_cent[indexer["gather_idx_i_structid"]]
        )
        self.nested_set(batch, self.lig_namemap[1], new_ligha_coords)
        self.empty_cache()
        return batch

    def empty_cache(self):
        """Empty the cached variables."""
        self._batch_size = None
        self._cother_mask = None
        self._n_res_per_sample = None
        self._n_cother_per_sample = None
        self._n_ligha_per_sample = None
        self._n_molid_per_sample = None
