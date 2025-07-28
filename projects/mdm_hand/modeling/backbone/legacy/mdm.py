import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rotation2xyz import Rotation2xyz

from d2.modeling import BACKBONE_REGISTRY
from .mdm_module import PositionalEncoding, TimestepEmbedder

# data input for hands default to x,y,z
class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, nframes, input_dim = x.shape
        x = self.poseEmbedding(x)
        x = x.transpose(0, 1) # [seqlen, bs, d]
        return x

# data input for hands default to x,y,z
class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        output = self.poseFinal(output)  # [seqlen, bs, 150]
        output = output.transpose(0, 1)
        return output

class MDMHandMANO(nn.Module):
    def __init__(self, input_dim=58, condition_dim=0, latent_dim=512, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", arch='trans_enc', emb_trans_dec=False, target_keys=["combined_feat"], cond_keys=[], **kargs):
        super().__init__()

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation

        self.input_feats = input_dim

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.arch = arch
        self.input_process = InputProcess(self.input_feats, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec

        print("TRANS_ENC init")
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)
        self.embed_timestep = TimestepEmbedder(self.latent_dim)
        self.output_process = OutputProcess(input_dim, self.latent_dim)
        self.condition_dim = condition_dim
        if condition_dim > 0:
            self.condition_embed = nn.Sequential(
                nn.Linear(condition_dim, latent_dim),
                nn.SiLU(),
                nn.Linear(latent_dim, latent_dim),
            )
        if 'obj_type' in cond_keys:
            self.obj_embedding = nn.Embedding(15, self.latent_dim)
        self.target_keys = target_keys
        self.cond_keys = cond_keys

    def get_gen_target(self, data):
        target = {k:v for k,v in data.items() if k in self.target_keys}
        cond = {k:v for k,v in data.items() if k in self.cond_keys}
        return target, cond

    def get_loss_mask(self, data, cond):
        return {}

    def dict_2_tensor(self, data):
        return data["combined_feat"]
    
    def tensor_2_dict(self, output_tensor):
        data = {
            "combined_feat": output_tensor
        }
        return data

    def forward(self, x, timesteps, condition, y=None, **kwargs):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, nframes, ndim = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        if emb.shape[1] == 1:
            emb = emb.repeat(1, bs, 1)

        x = self.input_process(x)# seq_len, bs, d

        if len(condition.keys()) > 0:
            if self.condition_dim > 0:
                cond_tensor = condition['cond_obj_10d'].to(x)
                cond_tensor = self.condition_embed(cond_tensor).transpose(0, 1) # bs, nf, dim
                x = x + cond_tensor
            if 'obj_type' in condition:
                obj_tensor = self.obj_embedding[condition['obj_type']].view(1, bs, -1) # 1, bs, dim
                x = x + obj_tensor

        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output

    def _apply(self, fn):
        super()._apply(fn)
        # self.rot2xyz.smpl_model._apply(fn)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        # self.rot2xyz.smpl_model.train(*args, **kwargs)


class MDMHandMANOCO(MDMHandMANO):
    def dict_2_tensor(self, data, *args, **kwargs):
        tensor = torch.cat([
            data["pos_left_g"], data["pos_left_j"],
            data["pos_right_g"], data["pos_right_j"],
            data["rot_left_g"], data["rot_left_j"],
            data["rot_right_g"], data["rot_right_j"],
            data["local_vel_left"],
            data["local_vel_right"]
        ], dim=-1)
        return tensor
    
    def tensor_2_dict(self, output_tensor, *args, **kwargs):
        data = {
            "pos_left_g": output_tensor[..., :3],
            "pos_left_j": output_tensor[..., 3:63],
            "pos_right_g": output_tensor[..., 63:66],
            "pos_right_j": output_tensor[..., 66:126],
            "rot_left_g": output_tensor[..., 126:132],
            "rot_left_j": output_tensor[..., 132:222],
            "rot_right_g": output_tensor[..., 222:228],
            "rot_right_j": output_tensor[..., 228:318],
            "local_vel_left": output_tensor[..., 318:381],
            "local_vel_right": output_tensor[..., 381:444],
        }
        return data

@BACKBONE_REGISTRY.register()
def mdm(cfg, input_shape):
    return MDMHandMANO(input_dim=460, latent_dim=512, ff_size=1024)

@BACKBONE_REGISTRY.register()
def mdm_co(cfg, input_shape):
    return MDMHandMANOCO(input_dim=444, condition_dim=10, latent_dim=512, ff_size=1024,
                         target_keys=["pos_left_g", "pos_left_j",
                                      "pos_right_g", "pos_right_j",
                                      "rot_left_g", "rot_left_j",
                                      "rot_right_g", "rot_right_j",
                                      "local_vel_left", "local_vel_right"],
                         cond_keys=["cond_obj_10d"])

@BACKBONE_REGISTRY.register()
def mdm_co_all(cfg, input_shape):
    return MDMHandMANOCO(input_dim=444, condition_dim=10, latent_dim=512, ff_size=1024,
                         target_keys=["pos_left_g", "pos_left_j",
                                      "pos_right_g", "pos_right_j",
                                      "rot_left_g", "rot_left_j",
                                      "rot_right_g", "rot_right_j",
                                      "local_vel_left", "local_vel_right"],
                         cond_keys=["relative_obj_positions", "obj_rot_6D", "obj_arti", "obj_type"])