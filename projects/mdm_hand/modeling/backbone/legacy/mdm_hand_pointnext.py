import numpy as np
import torch
import copy
import math
import torch.nn as nn
from torch import Tensor
from typing import Optional, Any, Union, Callable
from d2.modeling import BACKBONE_REGISTRY
from utils.rot import axis_angle_to_quaternion, quaternion_apply, quaternion_to_rotation_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix

from .mdm_module import PositionalEncoding, TimestepEmbedder
from .mdm import MDMHandMANO
from .pcd.spconvunet import SpUNetEncoder, SpUNetDecoder
from .pcd.ptv2 import PointTransformerV2
from .pcd.pointnext import get_pointnext, get_pointnext_96

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class PTEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers
        Copied from torch official implementation
    """
    __constants__ = ['norm']

    def __init__(self, pcd_in_dim, latent_dim, temporal_layer, num_layers, norm=None, enable_nested_tensor=False):
        super(PTEncoder, self).__init__()
        self.temporal_layers = _get_clones(temporal_layer, num_layers)
        self.num_layers = num_layers
        self.norm1 = norm
        self.norm2 = copy.deepcopy(self.norm1)

        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout=0.1)

        self.pcd_mid_dim = pcd_in_dim  * 8

        self.pcd_2_temporal = nn.Sequential(
            nn.Linear(2*self.pcd_mid_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        self.temporal_2_pcd = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, self.pcd_mid_dim)
        )

        self.hand_embedding = nn.Embedding(2, pcd_in_dim)
        self.pcd_arti_embedding = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, self.pcd_mid_dim)
        )

        self.pointnext = get_pointnext_96(pcd_in_dim)

    def forward(self, 
                pcd_feat,
                pcd_pos,
                rot_feat,
                diff_cond
        ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        bsz, nf, n_pcd, _,  pcd_in_dim = pcd_feat.shape
        hand_emb = self.hand_embedding.weight.view(1, 1, 1, 2, -1)
        output = pcd_feat

        # Object Info as pcd
        output = output + hand_emb + diff_cond["timestep_pcd"]
        output = output.transpose(2, 3).reshape(bsz*nf*2, n_pcd, pcd_in_dim)
        pcd_pos = pcd_pos.transpose(2, 3).reshape(bsz*nf*2, n_pcd, 3)
        p, f = self.pointnext.encoder.forward_seg_feat(
            {"pos": pcd_pos, "x": output.transpose(1, 2).contiguous()}
        )
        pcd_embedding = f[-1].view(bsz, nf, 2, self.pcd_mid_dim)

        # Temporal transformer
        output = self.pcd_2_temporal(pcd_embedding.view(bsz, nf, -1))
        output = output + diff_cond["obj_motion_cond"] + rot_feat
        seq = torch.cat([diff_cond["timestep_temporal"].view(bsz, 1, -1), output], dim=1)
        seq = seq + self.sequence_pos_encoder.pe[:nf+1].view(1, nf+1, -1)
        for temporal_layer in self.temporal_layers:
            seq = temporal_layer(seq)
        motion_output = seq[:, 1:]
        pcd_motion_output = self.temporal_2_pcd(seq[:, 1:]).reshape(bsz, nf, 1, -1)

        pcd_motion_output = (pcd_motion_output + pcd_embedding)
        f[-1] = pcd_motion_output.view(bsz*nf*2, -1, 1)
        pcd_output = self.pointnext.decoder(p, f).transpose(-1, -2)
        # pcd_feats = self.pt_transformer.forward_decoder(motion_output.view(bsz*nf*2, -1), pt_skips)
    
        if self.norm1 is not None:
            pcd_output = self.norm1(pcd_output)
            motion_output = self.norm2(motion_output)
        pcd_output = pcd_output.view(bsz, nf, 2, n_pcd, -1).transpose(2, 3)
        return pcd_output, motion_output


class STHandformer2(MDMHandMANO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Hack to drop unused layer
        del self.output_process
        del self.input_process
        del self.sequence_pos_encoder

        self.contact_mask_std = kwargs['contact_mask_std']
        in_dim = 66
        pcd_dim = 64

        self.input_feat = nn.Linear(in_dim, pcd_dim)
        self.input_rot = nn.Sequential(
            nn.Linear(12, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )
        self.embed_timestep_pcd = TimestepEmbedder(pcd_dim)

        temporal_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                    nhead=self.num_heads,
                                                    dim_feedforward=self.ff_size*2,
                                                    dropout=self.dropout,
                                                    activation=self.activation,
                                                    batch_first=True)
        self.st_encoder = PTEncoder(pcd_dim, self.latent_dim, temporal_layer, num_layers=self.num_layers)
        self.pcd_readout = nn.Sequential(
            nn.Linear(pcd_dim, pcd_dim),
            nn.SiLU(),
            nn.Linear(pcd_dim, 63)
        )
        self.rot_readout = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, 12)
        )

    def normal_to_axis_angle(self, normals):
        """
        Converts a batch of normal vectors to rotation vectors (axis-angle representation) in PyTorch.
        Assumes each normal vector in the batch is normalized.
        """
        normals = normals / torch.norm(normals, dim=-1, keepdim=True)
        z_axes = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32).repeat(normals.size(0), 1).to(normals)

        # Calculate rotation axes (cross products)
        rotation_axes = torch.cross(z_axes, normals)
        axis_norms = torch.norm(rotation_axes, dim=1, keepdim=True).clamp_min(1e-12)
        rotation_axes = rotation_axes / axis_norms

        # Calculate rotation angles
        angles = torch.acos(torch.sum(z_axes * normals, dim=1, keepdim=True))

        return rotation_axes * angles

    def dict_2_tensor(self, data, condition):
        pcd_pos = condition["pcd_pos"]
        pcd_normal = condition["pcd_normal"]
        bsz, nf, n_pcd, _ = pcd_pos.shape

        # get pcd relative joint pos 
        pos_lr = torch.cat([
            data["pos_left_g"], data["pos_left_j"],
            data["pos_right_g"], data["pos_right_j"]], dim=-1
        ).view(bsz, nf, 1, 42, 3) # l->r # l->r
        relative_pcd_pos = pos_lr - pcd_pos.view(bsz, nf, n_pcd, 1, 3)

        # pcd normal corrdinate relative pos
        rot_normal = self.normal_to_axis_angle(pcd_normal.view(-1, 3))
        mat_normal = quaternion_to_rotation_matrix(axis_angle_to_quaternion(rot_normal)) #bsz*nf*n_pcd, 3, 3
        relative_pcd_canon_pos = torch.bmm(
            mat_normal.transpose(1, 2),
            relative_pcd_pos.view(bsz*nf*n_pcd, 42, 3).transpose(1, 2)
        ).transpose(1, 2)
        pcd_pos_feat = relative_pcd_canon_pos.reshape(bsz, nf, n_pcd, 2, 63).view(bsz, nf, -1)
        rot_lr_g = torch.cat([data['rot_l'], data['rot_r']], dim=-1).view(bsz, nf, 12)
        pcd_feat = torch.cat([pcd_pos_feat.view(bsz, nf, -1), rot_lr_g], dim=-1)
        return pcd_feat

    def tensor_2_dict(self, output_tensor, condition, merge_pcd=False):
        bsz, nf, _ = output_tensor.shape
        pcd_pos_feat = output_tensor[..., :-12].reshape(bsz, nf, -1, 126)
        rot_lr_g = output_tensor[..., -12:].reshape(bsz, nf, 12)
        if merge_pcd:
            pcd_pos = condition["pcd_pos"]
            bsz, nf, n_pcd, _ = pcd_pos.shape

            rot_normal = self.normal_to_axis_angle(condition["pcd_normal"].view(-1, 3))
            mat_normal = quaternion_to_rotation_matrix(axis_angle_to_quaternion(rot_normal)) #bsz*nf*n_pcd, 3, 3
            relative_pcd_pos = torch.bmm(
                mat_normal,
                pcd_pos_feat.reshape(-1, 42, 3).transpose(1, 2),
            ).transpose(1, 2)
            pos_lr = relative_pcd_pos.reshape(bsz, nf, n_pcd, 42, 3) + pcd_pos.view(bsz, nf, n_pcd, 1, 3)
            pos_lr = pos_lr.mean(dim=2).view(bsz, nf, 126)
            data = {
                "pos_left_g": pos_lr[..., :3],
                "pos_left_j": pos_lr[..., 3:63],
                "pos_right_g": pos_lr[..., 63:66],
                "pos_right_j": pos_lr[..., 66:126],
                "rot_l": rot_lr_g[..., :6],
                "rot_r": rot_lr_g[..., 6:],
            }
        else:
            data = {
                "pos_left_g": pcd_pos_feat[..., :3],
                "pos_left_j": pcd_pos_feat[..., 3:63],
                "pos_right_g": pcd_pos_feat[..., 63:66],
                "pos_right_j": pcd_pos_feat[..., 66:126],
                "rot_l": rot_lr_g[..., :6],
                "rot_r": rot_lr_g[..., 6:],
            }
        return data

    def forward(self, x, timesteps, condition, y=None, is_train=False, x_target=None, **kwargs):
        pcd_pos_ca = condition["pcd_pos_ca"]
        pcd_pos_ca = pcd_pos_ca.unsqueeze(dim=-2).repeat(1, 1, 1, 2, 1)
        bsz, nf, n_pcd, _, _ = pcd_pos_ca.shape

        pcd_normal_ca = condition["pcd_normal_ca"]
        pcd_normal_ca = pcd_normal_ca.unsqueeze(dim=-2).repeat(1, 1, 1, 2, 1)

        pcd_pos_feat = x[..., :-12].reshape(bsz, nf, n_pcd, 2, 63)
        pcd_feat = torch.cat([pcd_pos_feat, pcd_normal_ca], dim=-1)
        pcd_feat = self.input_feat(pcd_feat)
        rot_feat = self.input_rot(x[..., -12:].reshape(bsz, nf, 12))

        diff_cond = {
            "timestep_pcd": self.embed_timestep_pcd(timesteps).view(bsz, 1, 1, 1, -1),
            "timestep_temporal": self.embed_timestep(timesteps).view(bsz, 1, 1, 1, -1)  
        }
        cond = condition
        if len(cond.keys()) > 0:
            if self.condition_dim > 0:
                cond_tensor = torch.cat([
                    condition['obj_trans'],
                    condition['obj_rot_6D'],
                    condition['obj_arti']
                ], dim=-1).to(x)
                diff_cond["obj_motion_cond"] = self.condition_embed(cond_tensor) # bs, nf, dim
            if 'obj_type' in cond:
                obj_tensor = self.obj_embedding(cond['obj_type']).view(bsz, 1, -1) # 1, bs, dim
                diff_cond["obj_shape_cond"] = obj_tensor

        pcd_out, motion_out = self.st_encoder(pcd_feat, pcd_pos_ca, rot_feat, diff_cond)
        pcd_out = self.pcd_readout(pcd_out) # bsz, nf, n_pcd, 2, 63
        rot_out = self.rot_readout(motion_out) # bsz, nf, 2, 6
        out = torch.cat([pcd_out.view(bsz, nf, -1), rot_out.view(bsz, nf, -1)], dim=-1)
        return out


@BACKBONE_REGISTRY.register()
def pcd_mdm(cfg, input_shape):
    return STHandformer2(input_dim=150, condition_dim=10, latent_dim=256, ff_size=512, num_layers=8,
                         contact_mask_std=cfg.MODEL.CONTACT_MASK_STD,
                         target_keys=["pos_left_g", "pos_left_j",
                                      "pos_right_g", "pos_right_j",
                                      "rot_l", "rot_r"],
                        # trans_obj may be can be improved? 
                         cond_keys=["obj_trans", "obj_rot_6D", "obj_arti", # condition
                                    "pcd_pos_ca", "pcd_normal_ca", "pcd_pos", "pcd_normal"
                                    ])
