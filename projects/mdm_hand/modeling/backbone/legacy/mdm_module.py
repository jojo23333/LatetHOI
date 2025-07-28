

import numpy as np
import torch
import copy
import torch.nn as nn

class EmptyModule(nn.Module):
    def forward(self, *args, **kwargs):
        return 0

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # do not use buffer, it cause problem in ddp
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

# Timestep embedding used in the DDPM++ and ADM architectures.
class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, max_positions=10000, endpoint=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_positions = max_positions
        self.endpoint = endpoint

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.latent_dim//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.latent_dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return self.time_embed(x).unsqueeze(0)

class InputProcess(nn.Module):
    def __init__(self, latent_dim, hand_dim=51, obj_dim=7):
        super().__init__()
        self.latent_dim = latent_dim
        self.hand_dim = hand_dim
        self.obj_dim = obj_dim
        self.lhand = nn.Sequential(
            nn.Linear(hand_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.rhand = nn.Sequential(
            nn.Linear(hand_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.obj = nn.Sequential(
            nn.Linear(obj_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

    def forward(self, x):
        x = x.transpose(0, 1) # [nframes, bs, d]
        lhand_seq = x[:, :, :self.hand_dim]
        rhand_seq = x[:, :, self.hand_dim:2*self.hand_dim]
        obj_seq = x[:, :, 2*self.hand_dim:]
        lhand_seq = self.lhand(lhand_seq)
        rhand_seq = self.rhand(rhand_seq)
        obj_seq = self.obj(obj_seq)
        return lhand_seq, rhand_seq, obj_seq


class OutputProcess(nn.Module):
    def __init__(self, latent_dim, hand_dim=51, obj_dim=7):
        super().__init__()
        self.latent_dim = latent_dim
        self.hand_dim = hand_dim
        self.obj_dim = obj_dim
        self.lhand = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, hand_dim),
        )
        self.rhand = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, hand_dim),
        )
        if self.obj_dim > 0:
            self.obj = nn.Linear(self.latent_dim, obj_dim)

    def forward(self, lhand_seq, rhand_seq, obj_seq):
        lhand = self.lhand(lhand_seq)
        rhand = self.rhand(rhand_seq)
        if self.obj_dim > 0:
            obj = self.obj(obj_seq)
            output = torch.cat([lhand, rhand, obj], dim=-1)
        else:
            output = torch.cat([lhand, rhand], dim=-1)

        output = output.transpose(0, 1)
        return output

class OutputProcess(nn.Module):
    def __init__(self, latent_dim, hand_dim=51, obj_dim=7):
        super().__init__()
        self.latent_dim = latent_dim
        self.hand_dim = hand_dim
        self.obj_dim = obj_dim
        self.lhand = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, hand_dim),
        )
        self.rhand = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, hand_dim),
        )
        if self.obj_dim > 0:
            self.obj = nn.Linear(self.latent_dim, obj_dim)

    def forward(self, lhand_seq, rhand_seq, obj_seq):
        lhand = self.lhand(lhand_seq)
        rhand = self.rhand(rhand_seq)
        if self.obj_dim > 0:
            obj = self.obj(obj_seq)
            output = torch.cat([lhand, rhand, obj], dim=-1)
        else:
            output = torch.cat([lhand, rhand], dim=-1)

        output = output.transpose(0, 1)
        return output

class OutputHeads(nn.Module):
    def __init__(self, latent_dim, target_keys):
        super().__init__()
        self.latent_dim = latent_dim
        self.rhand = SingleHandOut(latent_dim, target_keys)
        self.lhand = SingleHandOut(latent_dim, target_keys)

    def forward(self, lhand_seq, rhand_seq, obj_seq):
        lhand = self.lhand(lhand_seq)
        rhand = self.rhand(rhand_seq)
        
        output = torch.cat([lhand, rhand], dim=-1)
        output = output.transpose(0, 1)
        return output


class SingleHandOut(nn.Module):
    def __init__(self, latent_dim, target_keys=["rot_r", "pose_r", "trans_r", "joint_r", "wrist_r", 
                                                "rot_l", "pose_l", "trans_l", "joint_l", "wrist_l"]):
        super().__init__()
        self.latent_dim = latent_dim
        self.trans = 'trans_r' in target_keys
        self.rot = 'rot_r' in target_keys
        self.pose = 'pose_r' in target_keys
        self.kp = 'joint_r' in target_keys
        if self.trans:
            self.trans_out = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.SiLU(),
                nn.Linear(self.latent_dim, 3),
            )
        if self.rot:
            self.rot_out = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.SiLU(),
                nn.Linear(self.latent_dim, 6),
            )
        if self.pose:
            self.pose_out = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.SiLU(),
                nn.Linear(self.latent_dim, 6*15),
            )
        if self.kp:
            self.kpj_out = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.SiLU(),
                nn.Linear(self.latent_dim, 3*20),
            )
            self.kpw_out = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.SiLU(),
                nn.Linear(self.latent_dim, 3),
            )

    def forward(self, x):
        out = []
        if self.trans:
            out.append(self.trans_out(x))
        if self.rot:
            out.append(self.rot_out(x))
        if self.pose:
            out.append(self.pose_out(x))
        if self.kp:
            out.append(self.kpj_out(x))
            out.append(self.kpw_out(x))
        out = torch.cat(out, dim=-1)
        return out