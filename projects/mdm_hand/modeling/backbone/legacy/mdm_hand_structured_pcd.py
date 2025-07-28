import numpy as np
import torch
import copy
import torch.nn as nn
from torch import Tensor
from typing import Optional, Any, Union, Callable
from d2.modeling import BACKBONE_REGISTRY

from .mdm_module import PositionalEncoding, TimestepEmbedder
from .mdm import MDMHandMANOCO
from .pcd.spconvunet import SpUNetEncoder, SpUNetDecoder
from .pcd.ptv2 import PointTransformerV2

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
            nn.Linear(4*latent_dim, 2*latent_dim),
            nn.SiLU(),
            nn.Linear(2*latent_dim, 2*latent_dim)
        )

        self.temporal_2_spatial = nn.Sequential(
            nn.Linear(2*latent_dim, 2*latent_dim),
            nn.SiLU(),
            nn.Linear(2*latent_dim, 2*latent_dim)
        )

        self.hand_embedding = nn.Embedding(2, latent_dim)
        self.joint_embedding = nn.Embedding(21, latent_dim)

        self.pt_transformer = PointTransformerV2(84)
        # self.pcd_encoder = SpUNetEncoder(in_channels=84)
        # self.pcd_decoder = SpUNetDecoder(out_channels=84)

    def forward(self, 
                src: Tensor,
                pcd,
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
        hand_emb = self.hand_embedding.weight.view(1, 1, 2, 1, -1)
        joint_emb = self.joint_embedding.weight.view(1, 1, 1, 21, -1)
        num_encoder_layer = len(self.spatial_layers) // 2
        output = src
        assert nhands == 2 and nj == 21

        # Object Info
        pooled_pcd, pt_skips = self.pt_transformer.forward_encoder(pcd)
        pooled_pcd = pooled_pcd.view(bsz, nf, -1)

        # Spatial for all the joints
        # [bsz, 2, 21, nf, hidden_dim]
        #          ↑↑
        # Operate on joint dimension
        output = output + diff_cond["timestep"]
        output = output + hand_emb + joint_emb + (diff_cond["obj_shape_cond"].view(bsz, 1, 1, 1, -1) if "obj_shape_cond" in diff_cond else 0)
        output = output.view(bsz*nf*2, 21, -1)

        for spatial_layer in self.spatial_layers[:num_encoder_layer]:
            output = spatial_layer(output, src_mask=mask)
        out_skip = output.view(bsz, nf, 2, 21, -1)

        # Temporal for wrist only
        # [bsz, 2, 1, nf, hidden_dim]
        #             ↑↑
        # Operate on motion dimension
        output = out_skip[..., 0, :].reshape(bsz, nf, -1)
        output = torch.cat([output, pooled_pcd], dim=-1)
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
        pcd_feats = self.pt_transformer.forward_decoder(motion_output.view(bsz*nf*2, -1), pt_skips)
    
        if self.norm is not None:
            output = self.norm(output)
        return output, pcd_feats

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

    def dict_2_tensor(self, data):
        motion_tensor = torch.cat([
            data["pos_left_g"], data["pos_left_j"],
            data["pos_right_g"], data["pos_right_j"],
            data["rot_left_g"], data["rot_left_j"],
            data["rot_right_g"], data["rot_right_j"],
            data["local_vel_left"],
            data["local_vel_right"]
        ], dim=-1)
        bsz = motion_tensor.shape[0]
        self.motion_tensor_shape = motion_tensor.shape
        self.cnt_motion_feat = 1
        for x in self.motion_tensor_shape[1:]:
            self.cnt_motion_feat = self.cnt_motion_feat * x
        pcd_tensor = data['pcd_feat']
        self.pcd_tensor_shape = pcd_tensor.shape
        return torch.cat([motion_tensor.view(bsz, -1), pcd_tensor.view(bsz, -1)], dim=-1)
    
    def get_loss_mask(self, data, condition):
        # bsz, nf, j, 2 * hidden_dim
        pcd_tensor = data['pcd_feat']
        mask_pcd = torch.ones_like(pcd_tensor)
        for i, offset_i in enumerate(condition['pcd_offset']):
            mask_pcd[:, :, offset_i:] = 0
        return {
            "pcd_feat": mask_pcd
        }

    def tensor_2_dict(self, output_tensor):
        motion_tensor = output_tensor[:, :self.cnt_motion_feat].view(*self.motion_tensor_shape)
        pcd_tensor = output_tensor[:, self.cnt_motion_feat:].view(*self.pcd_tensor_shape)

        data = {
            "pos_left_g": motion_tensor[..., :3],
            "pos_left_j": motion_tensor[..., 3:63],
            "pos_right_g": motion_tensor[..., 63:66],
            "pos_right_j": motion_tensor[..., 66:126],
            "rot_left_g": motion_tensor[..., 126:132],
            "rot_left_j": motion_tensor[..., 132:222],
            "rot_right_g": motion_tensor[..., 222:228],
            "rot_right_j": motion_tensor[..., 228:318],
            "local_vel_left": motion_tensor[..., 318:381],
            "local_vel_right": motion_tensor[..., 381:444],
            'pcd_feat': pcd_tensor
        }
        return data

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

    def process_pcd_input(self, pcd_tensor, condition):
        bsz, nf, max_n, dim = pcd_tensor.shape
        pcd_tensor = pcd_tensor.reshape(bsz, nf, max_n, 2, -1).transpose(2, 3)
        coord = condition['pcd_coord'].unsqueeze(2).repeat(1, 1, 2, 1, 1)
        pcd_tensors = []
        coords = []
        offsets = []
        for i, offset_i in enumerate(condition['pcd_offset']):
            offsets.append(torch.ones(nf*2, device=pcd_tensor.device, dtype=torch.int)*offset_i)
            pcd_tensors.append(pcd_tensor[i,:,:,:offset_i].reshape(nf*2*offset_i,-1))
            coords.append(coord[i,:,:,:offset_i].reshape(nf*2*offset_i,-1))
        
        offset = torch.cat(offsets, dim=0)
        offset = torch.cumsum(offset, dim=0)

        return {
            "feat": torch.cat(pcd_tensors, dim=0),
            "offset": offset,
            "coord": torch.cat(coords, dim=0).to(torch.float32)
        }

    def process_pcd_output(self, pcd_feat, input_pcd_tensor, pcd_offset):
        bsz, nf, max_n, dim = input_pcd_tensor.shape
        padded_output = torch.zeros(bsz, nf, 2, max_n, dim//2, device=pcd_feat.device)
        start = 0
        for i, offset_i in enumerate(pcd_offset):
            end = start + nf*2*offset_i
            padded_output[i, :, :, :offset_i] = pcd_feat[start:end, :].reshape(nf, 2, offset_i, dim//2)
            start = end
        return padded_output.transpose(2, 3).reshape(bsz, nf, max_n, dim)

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
        motion_tensor = x[:, :self.cnt_motion_feat].reshape(*self.motion_tensor_shape)
        pcd_tensor = x[:, self.cnt_motion_feat:].reshape(*self.pcd_tensor_shape)

        xseq = self.process_input(motion_tensor) # bsz, 2, 21, nf, dim 
        pcd = self.process_pcd_input(pcd_tensor, condition)
        bsz, nhand, njoint, nf, hidden_dim = xseq.shape
        xseq = xseq.permute(0, 3, 1, 2, 4).contiguous()# bsz, nf, 2, 21, dim 

        diff_cond = {
            "timestep": self.embed_timestep(timesteps).view(bsz, 1, 1, 1, hidden_dim)  
        }
        # TODO 
        cond = condition
        if len(cond.keys()) > 0:
            if self.condition_dim > 0:
                cond_tensor = []
                for k, v in cond.items():
                    if k == "obj_type" or 'pcd' in k:
                        continue
                    assert v.ndim == 3 and v.shape[1] == nf, (k, v.shape)
                    cond_tensor.append(v)
                cond_tensor = torch.cat(cond_tensor, dim=-1).to(x)
                cond_tensor = self.condition_embed(cond_tensor) # bs, nf, dim
                diff_cond["obj_motion_cond"] = cond_tensor
            if 'obj_type' in cond:
                obj_tensor = self.obj_embedding(cond['obj_type']).view(bsz, 1, -1) # 1, bs, dim
                diff_cond["obj_shape_cond"] = obj_tensor


        out, pcd_feat = self.st_encoder(xseq, pcd, diff_cond)
        out = self.output(out)
        out_pcd = self.process_pcd_output(pcd_feat, pcd_tensor, condition["pcd_offset"])

        out_flatten = torch.cat([out.view(bsz, -1), out_pcd.view(bsz, -1)], dim=-1)
        # TODO add output
        return out_flatten


@BACKBONE_REGISTRY.register()
def pcd_structured_decoder(cfg, input_shape):
    return STHandformer2(input_dim=444, condition_dim=10, latent_dim=256, ff_size=512, num_layers=8,
                         target_keys=["pos_left_g", "pos_left_j",
                                      "pos_right_g", "pos_right_j",
                                      "rot_left_g", "rot_left_j",
                                      "rot_right_g", "rot_right_j",
                                      "local_vel_left", "local_vel_right", "pcd_feat"],
                         cond_keys=["relative_obj_positions", "obj_rot_6D", "obj_arti", 'pcd_coord', 'pcd_offset'])