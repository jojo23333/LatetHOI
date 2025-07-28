import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from .mdm_module import *

from d2.modeling import BACKBONE_REGISTRY

class EmptyModule(nn.Module):
    def forward(self, *args, **kwargs):
        return 0

class HOTransformerLayer(nn.Module):
    def __init__(self, latent_dim, num_heads, ff_size, dropout, activation) -> None:
        super().__init__()
        self_att_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                    nhead=num_heads,
                                                    dim_feedforward=ff_size,
                                                    dropout=dropout,
                                                    activation=activation)
        cross_att_layer = nn.TransformerDecoderLayer(d_model=latent_dim,
                                                    nhead=num_heads,
                                                    dim_feedforward=ff_size,
                                                    dropout=dropout,
                                                    activation=activation)
        self.lhand_sa = copy.deepcopy(self_att_layer)
        self.rhand_sa = copy.deepcopy(self_att_layer)
        self.lhand_ca = copy.deepcopy(cross_att_layer)
        self.rhand_ca = copy.deepcopy(cross_att_layer)
        
    def forward(self, lhand_seq_in, rhand_seq_in, obj_seq):
        lhand_seq = lhand_seq_in + self.lhand_ca(lhand_seq_in, rhand_seq_in+obj_seq)
        rhand_seq = rhand_seq_in + self.rhand_ca(rhand_seq_in, lhand_seq_in+obj_seq)
        lhand_seq = self.lhand_sa(lhand_seq)
        rhand_seq = self.rhand_sa(rhand_seq)
        return lhand_seq, rhand_seq, obj_seq


class HOTransformer(nn.Module):
    def __init__(self, hand_dim=51, obj_dim=7, condition_dim=0, latent_dim=512, 
                 ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 activation="gelu", target_keys=["rot_r", "pose_r", "trans_r", "rot_l", "pose_l", "trans_l", "arti_obj", "rot_obj", "trans_obj"], 
                 cond_keys=[], **kargs):
        super().__init__()
        self.hand_dim = hand_dim
        self.obj_dim = obj_dim
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.target_keys = target_keys
        self.cond_keys = cond_keys

        self.activation = activation

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.input_process = InputProcess(self.latent_dim, hand_dim, obj_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        self.motion_encoder = nn.ModuleList([
            HOTransformerLayer(
                self.latent_dim,
                self.num_heads,
                self.ff_size,
                self.dropout,
                self.activation
            )
            for i in range(num_layers)
        ])
        
        self.embed_timestep = TimestepEmbedder(self.latent_dim)
        if 'rot_obj' in target_keys:
            self.output_process = OutputProcess(self.latent_dim, hand_dim, obj_dim)
        else:
            self.output_process = OutputProcess(self.latent_dim, hand_dim, obj_dim=0)
        if condition_dim > 0:
            self.condition_embed = nn.Sequential(
                nn.Linear(condition_dim, latent_dim),
                nn.SiLU(),
                nn.Linear(latent_dim, latent_dim),
            )
        self.type_embedding = torch.nn.Embedding(3, self.latent_dim)
    def get_gen_target(self, data):
        target = {k:v for k,v in data.items() if k in self.target_keys}
        cond = {k:v for k,v in data.items() if k in self.cond_keys}
        return target, cond

    def dict_2_tensor(self, data):
        motion_r = torch.cat([data['rot_r'], data['pose_r'], data['trans_r']], dim=-1)
        motion_l = torch.cat([data['rot_l'], data['pose_l'], data['trans_l']], dim=-1)
        motion_obj = torch.cat([data['arti_obj'].unsqueeze(-1), data['rot_obj'], data['trans_obj']], dim=-1)
        motion_tensor = torch.cat([motion_l, motion_r, motion_obj], dim=-1)
        # import ipdb; ipdb.set_trace()
        return motion_tensor
    
    def tensor_2_dict(self, output_tensor):
        motion_l = output_tensor[..., :51]
        motion_r = output_tensor[..., 51:102]
        motion_obj = output_tensor[..., 102:]
        data = {
            "rot_r": motion_r[:, :, :3],
            "pose_r": motion_r[:, :, 3:48],
            "trans_r": motion_r[:, :, 48:51],
            "rot_l": motion_l[:, :, :3],
            "pose_l": motion_l[:, :, 3:48],
            "trans_l": motion_l[:, :, 48:51],
            "arti_obj": motion_obj[:, :, 0],
            "rot_obj": motion_obj[:, :, 1:4],
            "trans_obj": motion_obj[:, :, 4:7],
        }
        return data

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def forward(self, x, timesteps, condition=None, y=None, **kwargs):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, nframes, ndim = x.shape
        
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        if emb.shape[1] == 1:
            emb = emb.repeat(1, bs, 1)

        lhand_seq, rhand_seq, obj_seq = self.input_process(x)

        lhand_seq = torch.cat((emb, lhand_seq), axis=0)  # [seqlen+1, bs, d]
        rhand_seq = torch.cat((emb, rhand_seq), axis=0)
        obj_seq = torch.cat((emb, obj_seq), axis=0)

        lhand_seq = self.sequence_pos_encoder(lhand_seq) + self.type_embedding.weight[0].view(1, 1, -1)
        rhand_seq = self.sequence_pos_encoder(rhand_seq) + self.type_embedding.weight[1].view(1, 1, -1)
        obj_seq = self.sequence_pos_encoder(obj_seq) + self.type_embedding.weight[2].view(1, 1, -1)

        if condition is not None:
            if condition.ndim == 2:
                raise NotImplementedError
            elif condition.ndim == 3:
                condition = self.condition_embed(condition).transpose(0, 1)
                x = x + condition

        # # adding the timestep embed
        # xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        for i in range(self.num_layers):
            lhand_seq, rhand_seq, obj_seq = self.motion_encoder[i](lhand_seq, rhand_seq, obj_seq)

        output = self.output_process(lhand_seq[1:], rhand_seq[1:], obj_seq[1:])  # [bs, njoints, nfeats, nframes]
        return output


    def _apply(self, fn):
        super()._apply(fn)
        # self.rot2xyz.smpl_model._apply(fn)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        # self.rot2xyz.smpl_model.train(*args, **kwargs)


class HOTransformer6D(HOTransformer):
    def dict_2_tensor(self, data):
        motion_r = torch.cat([data['rot_r'], data['pose_r'], data['trans_r']], dim=-1)
        motion_l = torch.cat([data['rot_l'], data['pose_l'], data['trans_l']], dim=-1)
        motion_obj = torch.cat([data['arti_obj'].unsqueeze(-1), data['rot_obj'], data['trans_obj']], dim=-1)
        motion_tensor = torch.cat([motion_l, motion_r, motion_obj], dim=-1)
        return motion_tensor
    
    def tensor_2_dict(self, output_tensor):
        motion_l = output_tensor[..., :99]
        motion_r = output_tensor[..., 99:198]
        motion_obj = output_tensor[..., 198:208]
        data = {
            "rot_r": motion_r[:, :, :6],
            "pose_r": motion_r[:, :, 6:96],
            "trans_r": motion_r[:, :, 96:99],
            "rot_l": motion_l[:, :, :6],
            "pose_l": motion_l[:, :, 6:96],
            "trans_l": motion_l[:, :, 96:99],
            "arti_obj": motion_obj[:, :, 0],
            "rot_obj": motion_obj[:, :, 1:7],
            "trans_obj": motion_obj[:, :, 7:10],
        }
        return data

@BACKBONE_REGISTRY.register()
def ho_transformer(cfg, input_shape=None):
    if cfg.MODEL.ROT_REP == '6d':
        return HOTransformer6D(hand_dim=99, obj_dim=10)
    else:
        return HOTransformer()

class HOTransformerCondO(HOTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Hack to drop unused layer
        self.motion_encoder[-1].obj_ca = EmptyModule()

    def dict_2_tensor(self, data):
        motion_r = torch.cat([data['rot_r'], data['pose_r'], data['trans_r']], dim=-1)
        motion_l = torch.cat([data['rot_l'], data['pose_l'], data['trans_l']], dim=-1)
        motion_tensor = torch.cat([motion_l, motion_r], dim=-1)
        # import ipdb; ipdb.set_trace()
        return motion_tensor
    
    def tensor_2_dict(self, output_tensor):
        motion_l = output_tensor[..., :99]
        motion_r = output_tensor[..., 99:198]
        data = {
            "rot_r": motion_r[:, :, :6],
            "pose_r": motion_r[:, :, 6:96],
            "trans_r": motion_r[:, :, 96:99],
            "rot_l": motion_l[:, :, :6],
            "pose_l": motion_l[:, :, 6:96],
            "trans_l": motion_l[:, :, 96:99]
        }
        return data
    
    def forward(self, x, timesteps, condition=None, y=None, **kwargs):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, nframes, ndim = x.shape

        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        if emb.shape[1] == 1:
            emb = emb.repeat(1, bs, 1)

        obj_seq = torch.cat([condition['arti_obj'], condition['rot_obj'], condition['trans_obj']], dim=-1)

        x = torch.cat([x, obj_seq], dim=-1)
        lhand_seq, rhand_seq, obj_seq = self.input_process(x)

        lhand_seq = torch.cat((emb, lhand_seq), axis=0)  # [seqlen+1, bs, d]
        rhand_seq = torch.cat((emb, rhand_seq), axis=0)
        obj_seq = torch.cat((emb, obj_seq), axis=0)

        lhand_seq = self.sequence_pos_encoder(lhand_seq) + self.type_embedding.weight[0].view(1, 1, -1)
        rhand_seq = self.sequence_pos_encoder(rhand_seq) + self.type_embedding.weight[1].view(1, 1, -1)
        obj_seq = self.sequence_pos_encoder(obj_seq) + self.type_embedding.weight[2].view(1, 1, -1)

        # # adding the timestep embed
        # xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        for i in range(self.num_layers):
            lhand_seq, rhand_seq, obj_seq = self.motion_encoder[i](lhand_seq, rhand_seq, obj_seq)

        output = self.output_process(lhand_seq[1:], rhand_seq[1:], obj_seq[1:])  # [bs, njoints, nfeats, nframes]
        return output


@BACKBONE_REGISTRY.register()
def ho_transformer_co(cfg, input_shape):
    return HOTransformerCondO(
        target_keys=["rot_r", "pose_r", "trans_r", "rot_l", "pose_l", "trans_l"],
        cond_keys=["arti_obj", "rot_obj", "trans_obj"],
        hand_dim=99,
        obj_dim=10)


class KPHOTransformerCondO(HOTransformerCondO):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Hack to drop unused layer
        # self.output_process = OutputHeads(self.latent_dim, self.target_keys)

    def dict_2_tensor(self, data):
        bs, nf, _ = data['joint_r'].shape
        motion_r = torch.cat([data['rot_r'], data['pose_r'], data['trans_r'], data['joint_r'].view(bs, nf, -1), data['vol_r'].view(bs, nf, -1)], dim=-1)
        motion_l = torch.cat([data['rot_l'], data['pose_l'], data['trans_l'], data['joint_l'].view(bs, nf, -1), data['vol_l'].view(bs, nf, -1)], dim=-1)
        motion_tensor = torch.cat([motion_l, motion_r], dim=-1)
        return motion_tensor
    
    def tensor_2_dict(self, output_tensor):
        bs, nf, _ = output_tensor.shape
        motion_l = output_tensor[..., :225]
        motion_r = output_tensor[..., 225:450]
        data = {
            "rot_r": motion_r[:, :, :6],
            "pose_r": motion_r[:, :, 6:96],
            "trans_r": motion_r[:, :, 96:99],
            'joint_r': motion_r[:, :, 99:162],
            'vol_r': motion_r[:, :, 162:225],
            "rot_l": motion_l[:, :, :6],
            "pose_l": motion_l[:, :, 6:96],
            "trans_l": motion_l[:, :, 96:99],
            'joint_l': motion_l[:, :, 99:162],
            'vol_l': motion_l[:, :, 162:225],
        }
        return data

@BACKBONE_REGISTRY.register()
def kpho_transformer_co(cfg, input_shape):
    return KPHOTransformerCondO(
        target_keys=["rot_r", "pose_r", "trans_r", "rot_l", "pose_l", "trans_l", "joint_r", "joint_l", "vol_r", "vol_l"],
        cond_keys=["arti_obj", "rot_obj", "trans_obj"],
        hand_dim=225,
        obj_dim=10,
        latent_dim=512,
        ff_size=1024)


class KPTransformerCondO(HOTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Hack to drop unused layer
        self.motion_encoder[-1].obj_ca = EmptyModule()

    def dict_2_tensor(self, data):
        bs, nf, nj, _ = data['joint_r'].shape
        joint_r = data['joint_r'].view(bs, nf, -1)
        joint_l = data['joint_l'].view(bs, nf, -1)
        wrist_r = data['wrist_r'].view(bs, nf, -1)
        wrist_l = data['wrist_l'].view(bs, nf, -1)
        return torch.cat([joint_l, wrist_l, joint_r, wrist_r], dim=-1)
    
    def tensor_2_dict(self, output_tensor):
        bs, nf, _ = output_tensor.shape
        return {
            'joint_l': output_tensor[..., 0:60].view(bs, nf, -1, 3),
            'wrist_l': output_tensor[..., 60:63].view(bs, nf, -1, 3),
            'joint_r': output_tensor[..., 63:123].view(bs, nf, -1, 3),
            'wrist_r': output_tensor[..., 123:126].view(bs, nf, -1, 3)
        }
    
    def forward(self, x, timesteps, condition=None, y=None, **kwargs):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, nframes, ndim = x.shape

        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        if emb.shape[1] == 1:
            emb = emb.repeat(1, bs, 1)

        obj_seq = torch.cat([condition['arti_obj'].unsqueeze(-1), condition['rot_obj'], condition['trans_obj']], dim=-1)

        x = torch.cat([x, obj_seq], dim=-1)
        lhand_seq, rhand_seq, obj_seq = self.input_process(x)

        lhand_seq = torch.cat((emb, lhand_seq), axis=0)  # [seqlen+1, bs, d]
        rhand_seq = torch.cat((emb, rhand_seq), axis=0)
        obj_seq = torch.cat((emb, obj_seq), axis=0)

        lhand_seq = self.sequence_pos_encoder(lhand_seq) + self.type_embedding.weight[0].view(1, 1, -1)
        rhand_seq = self.sequence_pos_encoder(rhand_seq) + self.type_embedding.weight[1].view(1, 1, -1)
        obj_seq = self.sequence_pos_encoder(obj_seq) + self.type_embedding.weight[2].view(1, 1, -1)

        # # adding the timestep embed
        # xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        for i in range(self.num_layers):
            lhand_seq, rhand_seq, obj_seq = self.motion_encoder[i](lhand_seq, rhand_seq, obj_seq)

        output = self.output_process(lhand_seq[1:], rhand_seq[1:], obj_seq[1:])  # [bs, njoints, nfeats, nframes]
        return output

@BACKBONE_REGISTRY.register()
def kp_transformer_co(cfg, input_shape):
    return KPTransformerCondO(
        target_keys=["joint_r", "joint_l", "wrist_r", "wrist_l"],
        cond_keys=["arti_obj", "rot_obj", "trans_obj"],
        hand_dim=63,
        obj_dim=10)
