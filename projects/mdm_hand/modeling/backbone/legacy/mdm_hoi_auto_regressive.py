from .mdm_hoi import HOTransformer, EmptyModule
import torch
import torch.nn as nn

from d2.modeling import BACKBONE_REGISTRY

class HOTransformerAutoRegress(HOTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Hack to drop unused layer
        del self.input_process
        self.motion_encoder[-1].obj_ca = EmptyModule()
        
        self.map_l_noisy = nn.Sequential(
            nn.Linear(self.hand_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.map_l_cond = nn.Sequential(
            nn.Linear(self.hand_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.map_r_noisy = nn.Sequential(
            nn.Linear(self.hand_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.map_r_cond = nn.Sequential(
            nn.Linear(self.hand_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.map_obj_cond = nn.Sequential(
            nn.Linear(self.obj_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

    def get_gen_target(self, data):
        target = {k:v[:,-1:,:] for k,v in data.items() if k in self.target_keys}
        cond = {k:v for k,v in data.items() if k in self.cond_keys}
        return target, cond

    def dict_2_tensor(self, data):
        motion_r = torch.cat([data['rot_r'][:, -1:, :], data['pose_r'][:, -1:, :], data['trans_r'][:, -1:, :]], dim=-1)
        motion_l = torch.cat([data['rot_l'][:, -1:, :], data['pose_l'][:, -1:, :], data['trans_l'][:, -1:, :]], dim=-1)
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
        cond = condition

        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        if emb.shape[1] == 1:
            emb = emb.repeat(1, bs, 1)

        l_noisy =  self.map_l_noisy(x[...,:99])
        l_cond = torch.cat([cond['rot_l'][:, :-1, :], cond['pose_l'][:, :-1, :], cond['trans_l'][:, :-1, :]], dim=-1)
        l_cond = self.map_l_cond(l_cond)
        r_noisy =  self.map_r_noisy(x[...,99:])
        r_cond = torch.cat([cond['rot_r'][:, :-1, :], cond['pose_r'][:, :-1, :], cond['trans_r'][:, :-1, :]], dim=-1)
        r_cond = self.map_r_cond(r_cond)
        obj_cond = torch.cat([cond['arti_obj'].unsqueeze(-1), cond['rot_obj'], cond['trans_obj']], dim=-1)
        obj_cond = self.map_obj_cond(obj_cond)

        lhand_seq = torch.cat((l_cond, l_noisy), dim=1).transpose(0, 1)
        rhand_seq = torch.cat((r_cond, r_noisy), dim=1).transpose(0, 1)
        obj_seq = obj_cond.transpose(0, 1)

        lhand_seq = torch.cat((emb, lhand_seq), dim=0)  # [seqlen+1, bs, d]
        rhand_seq = torch.cat((emb, rhand_seq), dim=0)
        obj_seq = torch.cat((emb, obj_seq), dim=0)

        lhand_seq = self.sequence_pos_encoder(lhand_seq) + self.type_embedding.weight[0].view(1, 1, -1)
        rhand_seq = self.sequence_pos_encoder(rhand_seq) + self.type_embedding.weight[1].view(1, 1, -1)
        obj_seq = self.sequence_pos_encoder(obj_seq) + self.type_embedding.weight[2].view(1, 1, -1)

        # # adding the timestep embed
        # xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        for i in range(self.num_layers):
            lhand_seq, rhand_seq, obj_seq = self.motion_encoder[i](lhand_seq, rhand_seq, obj_seq)

        output = self.output_process(lhand_seq[-1:], rhand_seq[-1:], obj_seq[-1:])  # [bs, njoints, nfeats, nframes]
        return output
    
@BACKBONE_REGISTRY.register()
def ho_transformer_co_ar(cfg, input_shape):
    return HOTransformerAutoRegress(
        target_keys=["rot_r", "pose_r", "rot_l", "pose_l", "trans_l", "trans_r"],
        cond_keys=["rot_r", "pose_r", "rot_l", "pose_l", "arti_obj", "rot_obj", "trans_obj", "trans_l", "trans_r"],
        hand_dim=99,
        obj_dim=10,
        latent_dim=384,
        ff_size=768,
        num_layers=8
    )
