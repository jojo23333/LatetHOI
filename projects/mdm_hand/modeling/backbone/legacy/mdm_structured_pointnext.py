import numpy as np
import torch
import copy
import math
import torch.nn as nn
from torch import Tensor
from typing import Optional, Any, Union, Callable
from d2.modeling import BACKBONE_REGISTRY

from .mdm_module import PositionalEncoding, TimestepEmbedder
from .mdm import MDMHandMANOCO
from .pcd.spconvunet import SpUNetEncoder, SpUNetDecoder
from .pcd.ptv2 import PointTransformerV2
from .pcd.pointnext import get_pointnext, get_pointnext_normal

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class STEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers
        Copied from torch official implementation
    """
    __constants__ = ['norm']

    def __init__(self, latent_dim, spatial_layer, temporal_layer, num_layers, norm=None, enable_nested_tensor=False):
        super(STEncoder, self).__init__()
        self.temporal_layers = _get_clones(temporal_layer, num_layers)
        self.spatial_layers = _get_clones(spatial_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.enable_nested_tensor = enable_nested_tensor

        self.sequence_pos_encoder = PositionalEncoding(latent_dim*2, dropout=0.1)

        self.spatial_2_temporal = nn.Sequential(
            nn.Linear(2*latent_dim, 2*latent_dim),
            nn.SiLU(),
            nn.Linear(2*latent_dim, 2*latent_dim)
        )

        self.temporal_2_spatial = nn.Sequential(
            nn.Linear(2*latent_dim, 2*latent_dim),
            nn.SiLU(),
            nn.Linear(2*latent_dim, 2*latent_dim)
        )

        self.pcd_feat_2_latent = nn.Sequential(
            nn.Linear(1024, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        self.latent_2_pcd_feat = nn.Linear(2*latent_dim, 1024)

        self.hand_embedding = nn.Embedding(2, latent_dim)
        self.joint_embedding = nn.Embedding(21, latent_dim)
        self.pcd_arti_embedding = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, 1024)
        )

        self.pointnext = get_pointnext()

    def forward(self, 
                src: Tensor,
                pcd,
                diff_cond,
                condition,
                mask: Optional[Tensor] = None, 
                src_key_padding_mask: Optional[Tensor] = None
        ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        bsz, nf, nhands, nj, hidden_dim = src.shape
        hand_emb = self.hand_embedding.weight.view(1, 1, 2, 1, -1)
        joint_emb = self.joint_embedding.weight.view(1, 1, 1, 21, -1)
        num_encoder_layer = len(self.spatial_layers) // 2
        output = src
        assert nhands == 2 and nj == 21

        # Object Info as pcd
        p, f = self.pointnext.encoder.forward_seg_feat(pcd)
        pooled_f = f[-1].view(2, bsz, 1024)
        f_obj_t = pooled_f[0]
        f_obj_b = pooled_f[1]
        obj_pcd_embedding = self.pcd_feat_2_latent(f_obj_t) + self.pcd_feat_2_latent(f_obj_b)

        # Spatial for all the joints
        # [bsz, 2, 21, nf, hidden_dim]
        #          ↑↑
        # Operate on joint dimension
        output = output + diff_cond["timestep"]
        output = output + hand_emb + joint_emb + obj_pcd_embedding.view(bsz, 1, 1, 1, -1)
        output = output.view(bsz*nf*2, 21, -1)

        for spatial_layer in self.spatial_layers[:num_encoder_layer]:
            output = spatial_layer(output, src_mask=mask)
        out_skip = output.view(bsz, nf, 2, 21, -1)

        # Temporal for wrist only
        # [bsz, 2, 1, nf, hidden_dim]
        #             ↑↑
        # Operate on motion dimension
        output = out_skip[..., 0, :].reshape(bsz, nf, -1)
        output = self.spatial_2_temporal(output)
        output = output + diff_cond["obj_motion_cond"]
        seq = torch.cat([diff_cond["timestep"].view(bsz, 1, -1).repeat(1, 1, 2), output], dim=1)
        seq = seq + self.sequence_pos_encoder.pe[:nf+1].view(1, nf+1, -1)
        for temporal_layer in self.temporal_layers:
            seq = temporal_layer(seq)
        motion_output = self.temporal_2_spatial(seq[:, 1:]).view(bsz, nf, 2, 1, -1)

        # Spatial for all the joints
        # [bsz, 2, 21, nf, hidden_dim]
        #          ↑↑
        # Operate on joint dimension
        output = motion_output + out_skip
        output = output.view(bsz*nf*2, 21, -1)
        for spatial_layer in self.spatial_layers[num_encoder_layer:]:
            output = spatial_layer(output, src_mask=mask)
        output = output.view(bsz, nf, 2, 21, -1)

        # bsz, nf, n, 2, 42
        # comment this for memory save variant
        motion_pcd = self.latent_2_pcd_feat(motion_output.view(bsz, nf, -1))
        arti_embedding = self.pcd_arti_embedding(condition["obj_arti"].to(motion_pcd))
        p, f = self.expand(p, f, motion_pcd, arti_embedding)

        f = self.pointnext.decoder(p, f).view(bsz, 2, nf, 64, -1).transpose(-1, -2)
        # pcd_feats = self.pt_transformer.forward_decoder(motion_output.view(bsz*nf*2, -1), pt_skips)
    
        if self.norm is not None:
            output = self.norm(output)
        return output, torch.cat([f[:, 0], f[:, 1]], dim=-2)

    def expand(self, p, f, motion_out, arti_embedding):
        """
            p list of [2B, N, 3] -> [2B*nf, N, 3]
            f list of [2B, C, N] -> [2B*nf, C, N]
        """
        bsz, nf, _ = motion_out.shape
        motion_out = motion_out.unsqueeze(1).repeat(1, 2, 1, 1).view(bsz*2*nf, -1, 1)
        len_p = len(p)
        for i in range(len_p):
            p[i] = p[i].unsqueeze(1).repeat(1, nf, 1, 1).view(bsz*2*nf, p[i].shape[1], -1)# + motion_out[:, :p[i].shape[1], :]
            f[i] = f[i].unsqueeze(1).repeat(1, nf, 1, 1).view(bsz*2*nf, f[i].shape[1], -1)
        f_pooled = f[-1] + motion_out
        f_pooled = f_pooled.view(bsz, 2, nf, -1)
        f_pooled[:, 0, :, :] = f_pooled[:, 0, :, :] + arti_embedding
        f[-1] = f_pooled.view(bsz*2*nf, -1, 1)
        return p, f

class STHandformer2(MDMHandMANOCO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Hack to drop unused layer
        del self.output_process
        del self.input_process
        del self.condition_embed

        self.contact_mask_std = kwargs['contact_mask_std']
        in_dim = 12
        # TODO
        self.input_joints = nn.Linear(in_dim, self.latent_dim)
        self.input_wrists = nn.Linear(in_dim, self.latent_dim)
        self.readout_joints = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, in_dim),
        )

        self.readout_wrist = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, in_dim),
        )
        self.latent_2_pcd = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        self.pcd_readout = nn.Sequential(
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 3)
        )

        temporal_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim*2,
                                                    nhead=self.num_heads,
                                                    dim_feedforward=self.ff_size*2,
                                                    dropout=self.dropout,
                                                    activation=self.activation,
                                                    batch_first=True)
        spatial_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                    nhead=self.num_heads,
                                                    dim_feedforward=self.ff_size,
                                                    dropout=self.dropout,
                                                    activation=self.activation,
                                                    batch_first=True)
        self.st_encoder = STEncoder(self.latent_dim, spatial_layer, temporal_layer, num_layers=self.num_layers)
        self.condition_embed = nn.Sequential(
            nn.Linear(self.condition_dim, self.latent_dim*2),
            nn.SiLU(),
            nn.Linear(2*self.latent_dim, 2*self.latent_dim),
        )

    def process_input(self, x):
        def merge_feat(pos, rot, vel):
            bsz, nf, _ = pos.shape
            pos = pos.view(bsz, nf, 21, 3)
            rot = rot.view(bsz, nf, 16, 6)
            vel = vel.view(bsz, nf, 21, 3)
            rot_aug = rot[:,:,[15, 3, 6, 12, 9],:]
            rot = torch.cat([rot, rot_aug], dim=2)
            return torch.cat([pos, rot, vel], dim=-1)

        bsz, nf, _ = x.shape
        pos_left = x[..., 0:63]
        pos_right = x[..., 63:126]
        rot_left = x[..., 126:222]
        rot_right = x[..., 222:318]
        local_vel_left = x[..., 318:381]
        local_vel_right = x[..., 381:444]
        lfeat = merge_feat(pos_left, rot_left, local_vel_left)
        rfeat = merge_feat(pos_right, rot_right, local_vel_right)
        xseq = torch.stack([lfeat, rfeat], dim=1).transpose(2, 3) # bsz, 2, 21, nf, dim 
        xseq_wrist = self.input_wrists(xseq[:, :, :1])
        xseq_joint = self.input_joints(xseq[:, :, 1:])
        return torch.cat([xseq_wrist, xseq_joint], dim=2)

    def process_pcd_input(self, condition, x):
        pcd_top = condition["pcd_top"]
        pcd_bottom = condition["pcd_bottom"]
        pos = torch.cat([pcd_top[..., :3], pcd_bottom[..., :3]], dim=0).contiguous()
        normal = torch.cat([pcd_top[..., 3:], pcd_bottom[..., 3:]], dim=0).contiguous()

        return {
            "pos": pos.to(torch.float32),
            "x": pos.transpose(1, 2).contiguous().to(x),
            "normal": normal.to(torch.float32)
        }

    def gaussian_density(self, x):
        """Calculate the Gaussian density function for a given x, mean, and variance."""
        variance = self.contact_mask_std ** 2
        return torch.exp(- x.to(torch.float64) ** 2 / (2 * variance)).to(x)

    def output(self, output, pcd_feat, pcd_pos, gt_pos_lr=None):
        bsz, nf, _, _, hidden_dim = output.shape
        n_pcd = pcd_feat.shape[2]
        # output (bsz, nf, 2, 21, hidden_dim)
        # pcd_feat (bsz, nf, N, pcd_dim)
        # pcd_pos (bsz, nf, N, 3)
        # gt_lr_pos (bsz, nf, 42, 3)
        pcd_out = self.latent_2_pcd(output).view(bsz, nf, 42, 1, -1) - pcd_feat.view(bsz, nf, 1, n_pcd, -1)
        pcd_out = self.pcd_readout(pcd_out) # bsz, nf, 42, N, 3 
        pcd_trans = pcd_out + pcd_pos.view(bsz, nf, 1, -1, 3)
        if self.contact_mask_std > 0:
            if gt_pos_lr is None:
                # test time coarse evaluation setting using all the node
                # TODO
                gt_dist = pcd_pos.view(bsz, nf, 1, n_pcd, 3) - pcd_trans.mean(dim=-2).view(bsz, nf, 42, 1, 3)
                gt_dist = torch.linalg.norm(gt_dist, dim=-1, keepdim=True)
                contact_mask = self.gaussian_density(gt_dist)
                contact_mask = (contact_mask + 1e-6) / (contact_mask.sum(dim=-2, keepdim=True) + n_pcd*1e-6) # (bsz, nf, 42, n_pcd, 1)
                pos_lr = (pcd_trans * contact_mask).sum(dim=-2).view(bsz, nf, 42*3)
            else:
                # training time
                gt_dist = pcd_pos.view(bsz, nf, 1, n_pcd, 3) - gt_pos_lr.view(bsz, nf, 42, 1, 3)
                gt_dist = torch.linalg.norm(gt_dist, dim=-1, keepdim=True)
                contact_mask = self.gaussian_density(gt_dist)
                contact_mask = (contact_mask + 1e-6) / (contact_mask.sum(dim=-2, keepdim=True) + n_pcd*1e-6)  # (bsz, nf, 42, n_pcd, 1)
                pos_lr = (pcd_trans * contact_mask).sum(dim=-2).view(bsz, nf, 42*3)
        else:
            pos_lr = pcd_trans.mean(dim=-2).view(bsz, nf, 42*3)

        joint = self.readout_joints(output[..., 1:, :])
        wrist = self.readout_wrist(output[..., :1, :]) # bsz, nf, 2, j, hidden_dim
        x = torch.cat([wrist, joint], dim=3)
        rot_lr = x[..., :16, :6].reshape(bsz, nf, 32*6)
        vel_lr = x[..., 6:9].reshape(bsz, nf, 42*3)
        return torch.cat([pos_lr, rot_lr, vel_lr], dim=-1), pcd_trans

    def forward(self, x, timesteps, condition, y=None, is_train=False, x_target=None, **kwargs):
        xseq = self.process_input(x) # bsz, 2, 21, nf, dim 
        pcd = self.process_pcd_input(condition, xseq)
        bsz, nhand, njoint, nf, hidden_dim = xseq.shape
        xseq = xseq.permute(0, 3, 1, 2, 4).contiguous()# bsz, nf, 2, 21, dim 

        diff_cond = {
            "timestep": self.embed_timestep(timesteps).view(bsz, 1, 1, 1, hidden_dim)  
        }
        # TODO 
        cond = condition
        if len(cond.keys()) > 0:
            if self.condition_dim > 0:
                # cond_tensor = [
                #     condition['relative_obj_positions'],
                #     condition['obj_rot_6D'],
                #     condition['obj_arti']
                # ]
                # cond_tensor = []
                # for k, v in cond.items():
                #     if k == "obj_type" or 'pcd' in k:
                #         continue
                #     assert v.ndim == 3 and v.shape[1] == nf, (k, v.shape)
                #     cond_tensor.append(v)
                # cond_tensor = torch.cat(cond_tensor, dim=-1).to(x)
                cond_tensor = condition['cond_obj_10d'].to(x)
                cond_tensor = self.condition_embed(cond_tensor) # bs, nf, dim
                diff_cond["obj_motion_cond"] = cond_tensor
            if 'obj_type' in cond:
                obj_tensor = self.obj_embedding(cond['obj_type']).view(bsz, 1, -1) # 1, bs, dim
                diff_cond["obj_shape_cond"] = obj_tensor

        out, pcd_feat = self.st_encoder(xseq, pcd, diff_cond, condition)
        pcd_offset = torch.cat([condition["pcd_top_pos"], condition["pcd_bottom_pos"]], dim=-2)
        out, pcd_out = self.output(
            out,
            pcd_feat,
            pcd_offset,
            condition['gt_pos_lr'].to(out) if is_train else None
        )
        if is_train:
            return out, pcd_out
        else:
            return out


@BACKBONE_REGISTRY.register()
def pcd_structured_decoder_2(cfg, input_shape):
    return STHandformer2(input_dim=444, condition_dim=10, latent_dim=256, ff_size=512, num_layers=8,
                         contact_mask_std=cfg.MODEL.CONTACT_MASK_STD,
                         target_keys=["pos_left_g", "pos_left_j",
                                      "pos_right_g", "pos_right_j",
                                      "rot_left_g", "rot_left_j",
                                      "rot_right_g", "rot_right_j",
                                      "local_vel_left", "local_vel_right"],
                         cond_keys=["cond_obj_10d", "obj_arti", "obj_init_rot",
                                    'pcd_top', 'pcd_bottom', 'pcd_top_pos', 'pcd_bottom_pos', 'gt_pos_lr'])


class STEncoderGrab(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers
        Copied from torch official implementation
    """
    __constants__ = ['norm']

    def __init__(self, latent_dim, spatial_layer, temporal_layer, num_layers, norm=None, enable_nested_tensor=False):
        super(STEncoderGrab, self).__init__()
        self.temporal_layers = _get_clones(temporal_layer, num_layers)
        self.spatial_layers = _get_clones(spatial_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.enable_nested_tensor = enable_nested_tensor

        self.sequence_pos_encoder = PositionalEncoding(latent_dim*2, dropout=0.1)

        self.spatial_2_temporal = nn.Sequential(
            nn.Linear(2*latent_dim, 2*latent_dim),
            nn.SiLU(),
            nn.Linear(2*latent_dim, 2*latent_dim)
        )

        self.temporal_2_spatial = nn.Sequential(
            nn.Linear(2*latent_dim, 2*latent_dim),
            nn.SiLU(),
            nn.Linear(2*latent_dim, 2*latent_dim)
        )

        self.pcd_feat_2_latent = nn.Sequential(
            nn.Linear(1024, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        self.latent_2_pcd_feat = nn.Linear(2*latent_dim, 1024)

        self.hand_embedding = nn.Embedding(2, latent_dim)
        self.joint_embedding = nn.Embedding(21, latent_dim)

        self.pointnext = get_pointnext()

    def forward(self, 
                src: Tensor,
                pcd,
                diff_cond,
                condition,
                mask: Optional[Tensor] = None, 
                src_key_padding_mask: Optional[Tensor] = None
        ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        bsz, nf, nhands, nj, hidden_dim = src.shape
        hand_emb = self.hand_embedding.weight.view(1, 1, 2, 1, -1)
        joint_emb = self.joint_embedding.weight.view(1, 1, 1, 21, -1)
        num_encoder_layer = len(self.spatial_layers) // 2
        output = src
        assert nhands == 2 and nj == 21

        # Object Info as pcd
        p, f = self.pointnext.encoder.forward_seg_feat(pcd)
        pooled_f = f[-1].view(bsz, 1024)
        obj_pcd_embedding = self.pcd_feat_2_latent(pooled_f)

        # Spatial for all the joints
        # [bsz, 2, 21, nf, hidden_dim]
        #          ↑↑
        # Operate on joint dimension
        output = output + diff_cond["timestep"]
        output = output + hand_emb + joint_emb + obj_pcd_embedding.view(bsz, 1, 1, 1, -1)
        output = output.view(bsz*nf*2, 21, -1)

        for spatial_layer in self.spatial_layers[:num_encoder_layer]:
            output = spatial_layer(output, src_mask=mask)
        out_skip = output.view(bsz, nf, 2, 21, -1)

        # Temporal for wrist only
        # [bsz, 2, 1, nf, hidden_dim]
        #             ↑↑
        # Operate on motion dimension
        output = out_skip[..., 0, :].reshape(bsz, nf, -1)
        output = self.spatial_2_temporal(output)
        output = output + diff_cond["obj_motion_cond"]
        seq = torch.cat([diff_cond["timestep"].view(bsz, 1, -1).repeat(1, 1, 2), output], dim=1)
        seq = seq + self.sequence_pos_encoder.pe[:nf+1].view(1, nf+1, -1)
        for temporal_layer in self.temporal_layers:
            seq = temporal_layer(seq)
        motion_output = self.temporal_2_spatial(seq[:, 1:]).view(bsz, nf, 2, 1, -1)

        # Spatial for all the joints
        # [bsz, 2, 21, nf, hidden_dim]
        #          ↑↑
        # Operate on joint dimension
        output = motion_output + out_skip
        output = output.view(bsz*nf*2, 21, -1)
        for spatial_layer in self.spatial_layers[num_encoder_layer:]:
            output = spatial_layer(output, src_mask=mask)
        output = output.view(bsz, nf, 2, 21, -1)

        # bsz, nf, n, 2, 42
        # comment this for memory save variant
        motion_pcd = self.latent_2_pcd_feat(motion_output.view(bsz, nf, -1))
        p, f = self.expand(p, f, motion_pcd)

        f = self.pointnext.decoder(p, f).view(bsz, nf, 64, -1).transpose(-1, -2)
        # pcd_feats = self.pt_transformer.forward_decoder(motion_output.view(bsz*nf*2, -1), pt_skips)
    
        if self.norm is not None:
            output = self.norm(output)
        return output, f

    def expand(self, p, f, motion_out):
        """
            p list of [2B, N, 3] -> [2B*nf, N, 3]
            f list of [2B, C, N] -> [2B*nf, C, N]
        """
        bsz, nf, _ = motion_out.shape
        motion_out = motion_out.view(bsz*nf, -1, 1)
        len_p = len(p)
        for i in range(len_p):
            p[i] = p[i].unsqueeze(1).repeat(1, nf, 1, 1).view(bsz*nf, p[i].shape[1], -1)# + motion_out[:, :p[i].shape[1], :]
            f[i] = f[i].unsqueeze(1).repeat(1, nf, 1, 1).view(bsz*nf, f[i].shape[1], -1)
        f_pooled = f[-1] + motion_out
        f[-1] = f_pooled.view(bsz*nf, -1, 1)
        return p, f


class STHandformerGrab(MDMHandMANOCO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Hack to drop unused layer
        del self.output_process
        del self.input_process
        del self.condition_embed

        self.contact_mask_std = kwargs['contact_mask_std']
        in_dim = 12
        # TODO
        self.input_joints = nn.Linear(in_dim, self.latent_dim)
        self.input_wrists = nn.Linear(in_dim, self.latent_dim)
        self.readout_joints = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, in_dim),
        )

        self.readout_wrist = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, in_dim),
        )
        self.latent_2_pcd = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        self.pcd_readout = nn.Sequential(
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 3)
        )

        temporal_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim*2,
                                                    nhead=self.num_heads,
                                                    dim_feedforward=self.ff_size*2,
                                                    dropout=self.dropout,
                                                    activation=self.activation,
                                                    batch_first=True)
        spatial_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                    nhead=self.num_heads,
                                                    dim_feedforward=self.ff_size,
                                                    dropout=self.dropout,
                                                    activation=self.activation,
                                                    batch_first=True)
        self.st_encoder = STEncoderGrab(self.latent_dim, spatial_layer, temporal_layer, num_layers=self.num_layers)
        self.condition_embed = nn.Sequential(
            nn.Linear(self.condition_dim, self.latent_dim*2),
            nn.SiLU(),
            nn.Linear(2*self.latent_dim, 2*self.latent_dim),
        )

    def process_input(self, x):
        def merge_feat(pos, rot, vel):
            bsz, nf, _ = pos.shape
            pos = pos.view(bsz, nf, 21, 3)
            rot = rot.view(bsz, nf, 16, 6)
            vel = vel.view(bsz, nf, 21, 3)
            rot_aug = rot[:,:,[15, 3, 6, 12, 9],:]
            rot = torch.cat([rot, rot_aug], dim=2)
            return torch.cat([pos, rot, vel], dim=-1)

        bsz, nf, _ = x.shape
        pos_left = x[..., 0:63]
        pos_right = x[..., 63:126]
        rot_left = x[..., 126:222]
        rot_right = x[..., 222:318]
        local_vel_left = x[..., 318:381]
        local_vel_right = x[..., 381:444]
        lfeat = merge_feat(pos_left, rot_left, local_vel_left)
        rfeat = merge_feat(pos_right, rot_right, local_vel_right)
        xseq = torch.stack([lfeat, rfeat], dim=1).transpose(2, 3) # bsz, 2, 21, nf, dim 
        xseq_wrist = self.input_wrists(xseq[:, :, :1])
        xseq_joint = self.input_joints(xseq[:, :, 1:])
        return torch.cat([xseq_wrist, xseq_joint], dim=2)

    def process_pcd_input(self, condition, x):
        pcd_top = condition["pcd_top"]
        pos = pcd_top[..., :3]
        normal = pcd_top[..., 3:]

        return {
            "pos": pos.to(torch.float32),
            "x": pos.transpose(1, 2).contiguous().to(x),
            "normal": normal.to(torch.float32)
        }

    def gaussian_density(self, x):
        """Calculate the Gaussian density function for a given x, mean, and variance."""
        variance = self.contact_mask_std ** 2
        return torch.exp(- x.to(torch.float64) ** 2 / (2 * variance)).to(x)

    def output(self, output, pcd_feat, pcd_pos, gt_pos_lr=None):
        bsz, nf, _, _, hidden_dim = output.shape
        n_pcd = pcd_feat.shape[2]
        # output (bsz, nf, 2, 21, hidden_dim)
        # pcd_feat (bsz, nf, N, pcd_dim)
        # pcd_pos (bsz, nf, N, 3)
        # gt_lr_pos (bsz, nf, 42, 3)
        pcd_out = self.latent_2_pcd(output).view(bsz, nf, 42, 1, -1) - pcd_feat.view(bsz, nf, 1, n_pcd, -1)
        pcd_out = self.pcd_readout(pcd_out) # bsz, nf, 42, N, 3 
        pcd_trans = pcd_out + pcd_pos.view(bsz, 1, 1, -1, 3)
        if self.contact_mask_std > 0:
            if gt_pos_lr is None:
                # test time coarse evaluation setting using all the node
                # TODO
                gt_dist = pcd_pos.view(bsz, nf, 1, n_pcd, 3) - pcd_trans.mean(dim=-2).view(bsz, nf, 42, 1, 3)
                gt_dist = torch.linalg.norm(gt_dist, dim=-1, keepdim=True)
                contact_mask = self.gaussian_density(gt_dist)
                contact_mask = (contact_mask + 1e-6) / (contact_mask.sum(dim=-2, keepdim=True) + n_pcd*1e-6) # (bsz, nf, 42, n_pcd, 1)
                pos_lr = (pcd_trans * contact_mask).sum(dim=-2).view(bsz, nf, 42*3)
            else:
                # training time
                gt_dist = pcd_pos.view(bsz, nf, 1, n_pcd, 3) - gt_pos_lr.view(bsz, nf, 42, 1, 3)
                gt_dist = torch.linalg.norm(gt_dist, dim=-1, keepdim=True)
                contact_mask = self.gaussian_density(gt_dist)
                contact_mask = (contact_mask + 1e-6) / (contact_mask.sum(dim=-2, keepdim=True) + n_pcd*1e-6)  # (bsz, nf, 42, n_pcd, 1)
                pos_lr = (pcd_trans * contact_mask).sum(dim=-2).view(bsz, nf, 42*3)
        else:
            pos_lr = pcd_trans.mean(dim=-2).view(bsz, nf, 42*3)

        joint = self.readout_joints(output[..., 1:, :])
        wrist = self.readout_wrist(output[..., :1, :]) # bsz, nf, 2, j, hidden_dim
        x = torch.cat([wrist, joint], dim=3)
        rot_lr = x[..., :16, :6].reshape(bsz, nf, 32*6)
        vel_lr = x[..., 6:9].reshape(bsz, nf, 42*3)
        return torch.cat([pos_lr, rot_lr, vel_lr], dim=-1), pcd_trans

    def forward(self, x, timesteps, condition, y=None, is_train=False, x_target=None, **kwargs):
        xseq = self.process_input(x) # bsz, 2, 21, nf, dim 
        pcd = self.process_pcd_input(condition, xseq)
        bsz, nhand, njoint, nf, hidden_dim = xseq.shape
        xseq = xseq.permute(0, 3, 1, 2, 4).contiguous()# bsz, nf, 2, 21, dim 

        diff_cond = {
            "timestep": self.embed_timestep(timesteps).view(bsz, 1, 1, 1, hidden_dim)  
        }
        # TODO 
        cond = condition
        if len(cond.keys()) > 0:
            if self.condition_dim > 0:
                cond_tensor = condition['cond_obj_9d'].to(x)
                cond_tensor = self.condition_embed(cond_tensor) # bs, nf, dim
                diff_cond["obj_motion_cond"] = cond_tensor

        out, pcd_feat = self.st_encoder(xseq, pcd, diff_cond, condition)
        out, pcd_out = self.output(
            out,
            pcd_feat,
            condition["pcd_top_pos"],
            condition['gt_pos_lr'].to(out) if is_train else None
        )
        if is_train:
            return out, pcd_out
        else:
            return out


@BACKBONE_REGISTRY.register()
def pcd_structured_decoder_grab(cfg, input_shape):
    return STHandformerGrab(input_dim=444, condition_dim=9, latent_dim=256, ff_size=512, num_layers=8,
                         contact_mask_std=cfg.MODEL.CONTACT_MASK_STD,
                         target_keys=["pos_left_g", "pos_left_j",
                                      "pos_right_g", "pos_right_j",
                                      "rot_left_g", "rot_left_j",
                                      "rot_right_g", "rot_right_j",
                                      "local_vel_left", "local_vel_right"],
                         cond_keys=["cond_obj_9d", 
                                    'pcd_top', 'pcd_top_pos', 'gt_pos_lr'])