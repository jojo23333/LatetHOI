import numpy as np
import torch
import copy
import torch.nn as nn
from torch import Tensor
from typing import Optional, Any, Union, Callable
from d2.modeling import BACKBONE_REGISTRY

from .mdm_module import PositionalEncoding, TimestepEmbedder
from .mdm import MDMHandMANOCO


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

    def forward(self, 
                src: Tensor, 
                hand_emb,
                joint_emb,
                diff_cond,
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
        output = src
        assert nhands == 2 and nj == 21
        num_encoder_layer = len(self.spatial_layers) // 2

        output = output + diff_cond["timestep"]
        output = output + hand_emb + joint_emb + (diff_cond["obj_shape_cond"].view(bsz, 1, 1, 1, -1) if "obj_shape_cond" in diff_cond else 0)
        output = output.view(bsz*nf*2, 21, -1)
        # Spatial for all the joints
        # [bsz, 2, 21, nf, hidden_dim]
        #          ↑↑
        # Operate on joint dimension
        for spatial_layer in self.spatial_layers[:num_encoder_layer]:
            output = spatial_layer(output, src_mask=mask)
        out_skip = output.view(bsz, nf, 2, 21, -1)

        output = out_skip[..., 0, :].reshape(bsz, nf, -1)
        output = self.spatial_2_temporal(output)
        output = output + diff_cond["obj_motion_cond"]
        seq = torch.cat([diff_cond["timestep"].view(bsz, 1, -1).repeat(1, 1, 2), output], dim=1)
        seq = seq + self.sequence_pos_encoder.pe[:nf+1].view(1, nf+1, -1)
        # Temporal for wrist only
        # [bsz, 2, 1, nf, hidden_dim]
        #             ↑↑
        # Operate on motion dimension
        for temporal_layer in self.temporal_layers:
            seq = temporal_layer(seq)
        output = self.temporal_2_spatial(seq[:, 1:]).view(bsz, nf, 2, 1, -1)

        output = output + out_skip
        output = output.view(bsz*nf*2, 21, -1)
        # Spatial for all the joints
        # [bsz, 2, 21, nf, hidden_dim]
        #          ↑↑
        # Operate on joint dimension
        for spatial_layer in self.spatial_layers[num_encoder_layer:]:
            output = spatial_layer(output, src_mask=mask)
        output = output.view(bsz, nf, 2, 21, -1)

        if self.norm is not None:
            output = self.norm(output)
        return output

class STHandformer2(MDMHandMANOCO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Hack to drop unused layer
        del self.output_process
        del self.input_process
        del self.condition_embed

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
        self.hand_embedding = nn.Embedding(2, self.latent_dim)
        self.joint_embedding = nn.Embedding(21, self.latent_dim)

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
        self.seqTransEncoder = STEncoder(self.latent_dim, spatial_layer, temporal_layer, num_layers=self.num_layers)
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

    def output(self, x):
        # x (bsz, nf, 2, 21, hidden_dim)
        bsz, nf, _, _, hidden_dim = x.shape
        joint = self.readout_joints(x[..., 1:, :])
        wrist = self.readout_wrist(x[..., :1, :]) # bsz, nf, 2, j, hidden_dim
        x = torch.cat([wrist, joint], dim=3)
        pos_lr = x[..., :3].reshape(bsz, nf, 42*3)
        rot_lr = x[..., :16, 3:9].reshape(bsz, nf, 32*6)
        vel_lr = x[..., 9:].reshape(bsz, nf, 42*3)
        return torch.cat([pos_lr, rot_lr, vel_lr], dim=-1)

    def forward(self, x, timesteps, condition, y=None, **kwargs):
        xseq = self.process_input(x) # bsz, 2, 21, nf, dim 
        bsz, nhand, njoint, nf, hidden_dim = xseq.shape
        xseq = xseq.permute(0, 3, 1, 2, 4).contiguous()# bsz, nf, 2, 21, dim 
        
        joint_emb = self.joint_embedding.weight.view(1, 1, 1, 21, -1)
        hand_emb = self.hand_embedding.weight.view(1, 1, 2, 1, -1)

        diff_cond = {
            "timestep": self.embed_timestep(timesteps).view(bsz, 1, 1, 1, hidden_dim)  
        }
        # TODO 
        cond = condition
        if len(cond.keys()) > 0:
            if self.condition_dim > 0:
                cond_tensor = condition['cond_obj_10d'].to(x)
                cond_tensor = self.condition_embed(cond_tensor) # bs, nf, dim
                diff_cond["obj_motion_cond"] = cond_tensor
            if 'obj_type' in cond:
                obj_tensor = self.obj_embedding(cond['obj_type']).view(bsz, 1, -1) # 1, bs, dim
                diff_cond["obj_shape_cond"] = obj_tensor

        out = self.seqTransEncoder(xseq, hand_emb, joint_emb, diff_cond)
        out = self.output(out)
        # TODO add output
        return out


@BACKBONE_REGISTRY.register()
def mdm_structured_decoder(cfg, input_shape):
    return STHandformer2(input_dim=444, condition_dim=10, latent_dim=256, ff_size=512, num_layers=8,
                         target_keys=["pos_left_g", "pos_left_j",
                                      "pos_right_g", "pos_right_j",
                                      "rot_left_g", "rot_left_j",
                                      "rot_right_g", "rot_right_j",
                                      "local_vel_left", "local_vel_right"],
                         cond_keys=["cond_obj_10d"])