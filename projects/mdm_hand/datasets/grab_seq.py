from typing import Any
import numpy as np
from torch.utils import data
import os
import copy
import logging

from d2.data.build import DATASET_REGISTRY
import glob
import numpy as np
import torch
import trimesh
from bps_torch.bps import bps_torch

SPLIT_MODE = "object" # 
logger = logging.getLogger("d2")
class GrabMotion(data.Dataset):
    def __init__(self, root, split='train', max_frame=256):
        self.root = root
        self.split = split
        d = np.load(os.path.join(root, 'mean_std.npy'), allow_pickle=True).item()
        self.mean = d['mean']
        self.std = d['std']
        self.seqs = self.get_seqs(root, split=split)
        self.max_frame=max_frame

        self.data = [None] * len(self.seqs)
        self.bps = self.compute_canon_bps()
        self.compute_clip_embedding()

    def compute_clip_embedding(self):
        import clip
        device = 'cuda'
        clip_model, _ = clip.load("ViT-B/32")
        clip_model = clip_model.to(device)
        self.txt2clip = {}
        action_txt = []
        for seq_name in self.seqs:
            name = os.path.basename(seq_name).split('.')[0]
            obj, action = name.split('_')[0:2]
            txt = f'{action} the {obj}'
            clip_embedding = clip.tokenize([txt]).to(device)
            action_txt.append(txt)
            if txt not in self.txt2clip:
                self.txt2clip[txt] = clip_model.encode_text(clip_embedding).squeeze().to(torch.float32).cpu().detach()
        logger.info(f"{len(self.txt2clip)} different text cond for {self.split} set")
        self.action_txt = action_txt
        
        # if split is test, then for each action txt only get 4 sample
        # TODO change this back
        if self.split == 'test':
            txt2seqs = {}
            for seq_name, txt in zip(self.seqs, action_txt):
                if txt not in txt2seqs:
                    txt2seqs[txt] = [seq_name]
                else:
                    txt2seqs[txt].append(seq_name)
            new_seqs = []
            new_action_txt = []
            for txt, seqs in txt2seqs.items():
                new_seqs.extend(seqs[:4])
                new_action_txt.extend([txt]*4)
            self.seqs = new_seqs
            self.action_txt = new_action_txt
            logger.info(f"All test action txts: {self.action_txt}")
        return

    def compute_canon_bps(self):
        np.random.seed(100)
        BPS_BASE = torch.from_numpy(np.load('config/bps.npz')['basis']).to(torch.float32)
        bps = bps_torch(custom_basis = BPS_BASE)
        mesh_path = './data/grab/tools/object_meshes/contact_meshes'
        meta = {}
        for obj_mesh in os.listdir(mesh_path):
            if obj_mesh.endswith(".ply"):
                print("Preprocessing: ", obj_mesh)
                obj_name = obj_mesh.split('.')[0]
                verts_obj = trimesh.load(os.path.join(mesh_path, obj_mesh)).vertices

                if verts_obj.shape[0] > 1024:
                    verts_sample_id = np.random.choice(verts_obj.shape[0], 1024, replace=False)
                else:
                    verts_sample_id = np.arange(verts_obj.shape[0])
                verts_sampled = torch.from_numpy(verts_obj[verts_sample_id])
                bps_obj = bps.encode(verts_sampled, feature_type='dists')['dists'].to(torch.float32).cpu().detach()
                meta[obj_name] = bps_obj[0]
        return meta

    def get_seqs(self, root, split='train'):
        # IMOS SPLIT
        if SPLIT_MODE == 'subject':
            if split == 'train':
                seqs = glob.glob(os.path.join(root, 'train/s*/**.npy')) +\
                    glob.glob(os.path.join(root, 'val/s*/**.npy'))+\
                    glob.glob(os.path.join(root, 'test/s*/**.npy'))
                seqs = [seq for seq in seqs if 's10' not in seq]
            else:
                seqs = glob.glob(os.path.join(root, 'train/s*/**.npy')) +\
                    glob.glob(os.path.join(root, 'val/s*/**.npy'))+\
                    glob.glob(os.path.join(root, 'test/s*/**.npy'))
                seqs = [seq for seq in seqs if 's10' in seq]
                # seqs = seqs[::6]
        else:
            seqs = glob.glob(os.path.join(root, split, 's*/**.npy'))
        return seqs

    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, index) -> Any:
        seq = self.data[index]
        if seq is not None:
            return seq
        else:
            seq = np.load(self.seqs[index], allow_pickle=True).item()
            seq['feat'] = (seq['feat'] - self.mean.reshape(1, -1)) / self.std.reshape(1, -1)
            nf = seq['feat'].shape[0]
            if nf >= self.max_frame:
                seq['feat'] = seq['feat'][:self.max_frame, :]
            else:
                seq['feat'] = np.pad(seq['feat'], ((0, self.max_frame-nf), (0, 0)), 'constant', constant_values=0)
                seq['feat'][nf:, :] = seq['feat'][nf-1:nf, :]

            if seq['text_features'].ndim == 2:
                seq['text_features'] = seq['text_features'][0, :]
            seq['text_features'] = seq['text_features'].astype(np.float32)
            action_name = self.action_txt[index]
            text_features = self.txt2clip[action_name]
            data =  {
                'feat': seq['feat'],
                "text_features": text_features,
                "bps_object": self.bps[seq['object']],
                "obj_name": seq['object'],
                'nf': nf,
                "action_name": action_name,
            }
            self.data[index] = data
            return data
    
    def unnormalize(self, x):
        if isinstance(x, torch.Tensor):
            std = torch.from_numpy(self.std).to(x.device)
            mean = torch.from_numpy(self.mean).to(x.device)
        else:
            std = self.std
            mean = self.mean
        return x * std + mean


class GrabLatentMotion(GrabMotion):
    def __init__(self, root, split='split', max_frame=256):
        self.root = root
        self.split = split

        d = np.load(os.path.join(root, 'mean_std.npy'), allow_pickle=True).item()
        self.mean = d['ldm_mean'].reshape(1, -1)
        self.std = d['ldm_std'].reshape(1, -1)
        self.seqs = self.get_seqs(root, split=split)
        self.max_frame=max_frame
        self.latent = [None] * len(self.seqs) 
        self.cache = {}
        self.object = {}
        self.bps = self.compute_canon_bps()
        self.compute_clip_embedding()

    def pad_with_last(self, feat, max_frame, dim=2):
        nf = feat.shape[0]
        if nf >= max_frame:
            feat = feat[:max_frame, ...]
        else:
            if dim == 2:
                feat = np.pad(feat, ((0, max_frame-nf), (0, 0)), 'constant', constant_values=0)
            elif dim == 3 :
                feat = np.pad(feat, ((0, max_frame-nf), (0, 0), (0, 0)), 'constant', constant_values=0)
            feat[nf:, ...] = feat[nf-1, ...]
        return feat

    def update(self, idx, latent):
        nf = latent.shape[0]
        latent = self.pad_with_last(latent, self.max_frame, dim=2)
        self.latent[idx] = latent

    def get_raw(self, i):
        seq_name = self.seqs[i]
        return np.load(seq_name, allow_pickle=True).item()

    def __getitem__(self, index) -> Any:
        if index not in self.cache:
            seq_name = self.seqs[index]
            seq = np.load(seq_name, allow_pickle=True).item()
            feat = (seq['feat_ldm'] - self.mean) / self.std.reshape(1, -1)

            latent = self.latent[index]
            feat = self.pad_with_last(feat, self.max_frame, dim=2)
            verts_obj = seq['obj_verts']
            seq['rhand']['verts_rhand'] = self.pad_with_last(seq['rhand']['verts_rhand'], self.max_frame, dim=3)
            seq['rhand']['verts_object'] = self.pad_with_last(seq['rhand']['verts_object'], self.max_frame, dim=3)
            seq['lhand']['verts_rhand'] = self.pad_with_last(seq['lhand']['verts_rhand'], self.max_frame, dim=3)
            seq['lhand']['verts_object'] = self.pad_with_last(seq['lhand']['verts_object'], self.max_frame, dim=3)
            seq['rhand']['trans_obj'] = self.pad_with_last(seq['rhand']['trans_obj'], self.max_frame, dim=2)
            seq['lhand']['trans_obj'] = self.pad_with_last(seq['lhand']['trans_obj'], self.max_frame, dim=2)

            assert latent is not None
            # TODO here, replace verts_sample with furthest point sample
            if verts_obj.shape[0] > 1024:
                verts_sample_id = np.random.choice(verts_obj.shape[0], 1024, replace=False)
            else:
                verts_sample_id = np.arange(verts_obj.shape[0])

            # if seq['bps_object'].ndim == 2:
            #     seq['bps_object'] = seq['bps_object'][0, :]
                
            action_name = self.action_txt[index]
            text_features = self.txt2clip[action_name]
            # if seq['text_features'].ndim == 2:
            #     seq['text_features'] = seq['text_features'][0, :]
            data = {
                'feat': feat,
                'latent': latent, 
                'obj_verts': verts_obj[verts_sample_id],
                'rhand': {
                    'verts_rhand': seq['rhand']['verts_rhand'] + seq['rhand']['trans_obj'].reshape(-1, 1, 3),
                    'verts_object': seq['rhand']['verts_object']+ seq['rhand']['trans_obj'].reshape(-1, 1, 3),
                },
                'lhand': {
                    'verts_rhand': seq['lhand']['verts_rhand'] + seq['lhand']['trans_obj'].reshape(-1, 1, 3),
                    'verts_object': seq['lhand']['verts_object'] + seq['lhand']['trans_obj'].reshape(-1, 1, 3),
                },
                "text_features": text_features,
                "bps_object": self.bps[seq['object']],
                'std_feat': self.std,
                'mean_feat': self.mean,
                "obj_name": seq['object'],
                "action_name": action_name,
            }
            # self.cache[index] = data
            self.object[data['obj_name']] = seq['obj_verts']
            return data
        else:
            data = self.cache[index]
            # verts_obj = self.object[data['obj_name']]
            # if verts_obj.shape[0] > 1024:
            #     verts_sample_id = np.random.choice(verts_obj.shape[0], 1024, replace=False)
            # else:
            #     verts_sample_id = np.arange(verts_obj.shape[0])
            # data['obj_verts'] = verts_obj[verts_sample_id]
            return data

class GrabMotionMLD(GrabLatentMotion):
    def __init__(self, root, split='train', max_frame=256):
        self.root = root
        self.split = split
        d = np.load(os.path.join(root, 'mean_std.npy'), allow_pickle=True).item()
        self.mean = d['mean']
        self.std = d['std']
        self.seqs = self.get_seqs(root, split=split)
        self.max_frame=max_frame

        self.data = [None] * len(self.seqs)
        self.latent = [None] * len(self.seqs) 
        self.bps = self.compute_canon_bps()
        self.compute_clip_embedding()


    def update(self, idx, latent):
        self.latent[idx] = latent[0]

    def get_raw(self, index):
        seq_name = self.seqs[index]
        
        seq = np.load(self.seqs[index], allow_pickle=True).item()
        seq['feat'] = (seq['feat'] - self.mean.reshape(1, -1)) / self.std.reshape(1, -1)
        nf = seq['feat'].shape[0]
        if nf >= self.max_frame:
            seq['feat'] = seq['feat'][:self.max_frame, :]
        else:
            seq['feat'] = np.pad(seq['feat'], ((0, self.max_frame-nf), (0, 0)), 'constant', constant_values=0)
            seq['feat'][nf:, :] = seq['feat'][nf-1:nf, :]

        return {
            'feat': seq['feat'],
            'nf': nf,
        }
    
    def __getitem__(self, index) -> Any:
        seq = self.data[index]
        if seq is not None:
            return seq
        else:
            seq_name = self.seqs[index]
            seq = np.load(seq_name, allow_pickle=True).item()

            latent = self.latent[index]
            assert latent is not None
            action_name = self.action_txt[index]
            text_features = self.txt2clip[action_name]
            # if seq['text_features'].ndim == 2:
            #     seq['text_features'] = seq['text_features'][0, :]
            data = {
                'latent': latent, 
                "text_features": text_features,
                "bps_object": self.bps[seq['object']],
                'std_feat': self.std,
                'mean_feat': self.mean,
                "obj_name": seq['object'],
                "action_name": action_name,
            }
            return data
    
    def unnormalize(self, x):
        if isinstance(x, torch.Tensor):
            std = torch.from_numpy(self.std).to(x.device)
            mean = torch.from_numpy(self.mean).to(x.device)
        else:
            std = self.std
            mean = self.mean
        return x * std + mean

@DATASET_REGISTRY.register()
def grab_motion_mld(cfg, split):
    dataset = GrabMotionMLD("data/grab/grab_seq_20fps", split=split, max_frame=160)
    return dataset


@DATASET_REGISTRY.register()
def grab_motion(cfg, split):
    dataset = GrabMotion("data/grab/grab_seq_20fps", split=split, max_frame=160)
    return dataset

@DATASET_REGISTRY.register()
def grab_latent_motion(cfg, split):
    dataset = GrabLatentMotion("data/grab/grab_seq_20fps", split=split, max_frame=160)
    return dataset

if __name__ == '__main__':
    dataset = GrabMotion("data/grab/grab_seq", split='train')
    print(dataset[0])
    # dataset = GrabLatentMotion("data/grab/grab_latnet", split='train')
    # print(dataset[0])