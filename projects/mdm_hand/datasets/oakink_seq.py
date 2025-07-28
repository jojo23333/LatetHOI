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

# obj_to_action = {
#     'binoculars': ['lift', 'pass', 'see', 'offhand'],
#     'bowl': ['drink', 'pass', 'lift', 'offhand'],
#     'camera': ['takepicture', 'browse', 'pass', 'lift', 'offhand'],
#     'can': ['pass', 'lift', 'inspect', 'offhand'],
#     'drill': ['pass', 'lift', 'inspect'],
#     'eyeglasses': ['lift', 'clean', 'wear', 'pass', 'offhand'],
#     'flashlight': ['on', 'pass', 'lift', 'offhand'],
#     'fryingpan': ['cook', 'pass', 'lift', 'offhand'],
#     'gamecontroller': ['pass', 'play', 'lift', 'offhand'],
#     'hammers': ['use', 'pass', 'lift'],
#     'headphones': ['lift', 'pass', 'use', 'offhand'],
#     'knife': ['chop', 'lift', 'peel', 'pass'],
#     'lightbulb': ['pass', 'screw'],
#     'mouse': ['use', 'pass', 'lift', 'offhand'],
#     'mug': ['drink', 'lift', 'pass', 'offhand', 'toast'],
#     'phone': ['call', 'pass', 'lift', 'offhand'],
#     'teapot': ['pass', 'pour', 'lift'],
#     'toothbrush': ['pass', 'brush', 'lift'],
#     'wineglass': ['drink', 'pass', 'toast', 'lift', 'offhand'],
#     'waterbottle': ['open', 'shake', 'drink', 'pass', 'pour', 'lift', 'offhand'],
#     'screwdriver': ['pass', 'lift'],
#     'donut': ['eat', 'pass', 'lift']
# }

obj_to_action = {
    'binoculars': ['see', 'offhand'],
    'bowl': ['drink', 'pass'],
    'camera': ['takepicture', 'browse'],
    'can': ['inspect', 'offhand'],
    'drill': ['pass', 'lift'],
    'eyeglasses': ['clean', 'wear'],
    'flashlight': ['on', 'offhand'],
    'fryingpan': ['cook', 'pass'],
    'gamecontroller': ['play', 'lift'],
    'hammers': ['use', 'pass'],
    'headphones': ['use', 'offhand'],
    'knife': ['chop', 'peel'],
    'lightbulb': ['pass', 'screw'],
    'mouse': ['use', 'offhand'],
    'mug': ['drink', 'toast'],
    'phone': ['call', 'pass'],
    'teapot': ['pour', 'lift'],
    'toothbrush': ['brush', 'lift'],
    'wineglass': ['drink', 'toast'],
    'waterbottle': ['open', 'shake', 'drink', 'pour'],
    'screwdriver': ['pass', 'lift'],
    'donut': ['eat', 'lift']
}

SPLIT_MODE = "object" # 
logger = logging.getLogger("d2")
class OakInkMotionLatent(data.Dataset):
    def __init__(self, root, split='train', max_frame=160):
        self.root = root
        self.split = split
        d = np.load('data/grab/grab_seq_20fps/mean_std.npy', allow_pickle=True).item()
        self.mean = d['ldm_mean'].reshape(1, -1)
        self.std = d['ldm_std'].reshape(1, -1)
        # Neurips Submit split 1: oakink 100 object set
        self.seqs = glob.glob('data/oakink/oakink_aligned/*.ply')
        # Neurips Submit split 2: oakink colored object set
        # self.seqs = glob.glob('data/oakink/selected_texture_uv/**/*.obj')
        self.max_frame=max_frame

        self.data = [None] * len(self.seqs)
        self.object = {}
        self.bps = self.compute_canon_bps()
        self.compute_clip_embedding()
        print("Seq Count:", self.__len__())
        
    def compute_canon_bps(self):
        np.random.seed(100)
        BPS_BASE = torch.from_numpy(np.load('config/bps.npz')['basis']).to(torch.float32)
        bps = bps_torch(custom_basis = BPS_BASE)
        meta = {}
        for obj_mesh in self.seqs:
            print("Preprocessing: ", obj_mesh)
            obj_full_name = os.path.basename(obj_mesh).split('.')[0]
            mesh = trimesh.load(obj_mesh)
            verts_obj, _ = trimesh.sample.sample_surface_even(mesh, 4096)
            self.object[obj_full_name] = verts_obj

            if verts_obj.shape[0] > 2048:
                verts_sample_id = np.random.choice(verts_obj.shape[0], 2048, replace=False)
            else:
                print(f"Warning: has only {verts_obj.shape[0]} points for ", obj_full_name)
                verts_sample_id = np.arange(verts_obj.shape[0])
            verts_sampled = torch.from_numpy(verts_obj[verts_sample_id])
            bps_obj = bps.encode(verts_sampled, feature_type='dists')['dists'].to(torch.float32).cpu().detach()
            meta[obj_full_name] = bps_obj[0]
        return meta
    
    def compute_clip_embedding(self):
        import clip
        device = 'cuda'
        clip_model, _ = clip.load("ViT-B/32")
        clip_model = clip_model.to(device)
        self.txt2clip = {}
        action_txt = []
        seqs = []
        for obj_mesh in self.seqs:
            obj_name = os.path.basename(obj_mesh).split('_')[0]
            actions = obj_to_action[obj_name]
            for action in actions:
                seqs.append(obj_mesh)
                txt = f'{action} the {obj_name}'
                clip_embedding = clip.tokenize([txt]).to(device)
                action_txt.append(txt)
                if txt not in self.txt2clip:
                    self.txt2clip[txt] = clip_model.encode_text(clip_embedding).squeeze().to(torch.float32).cpu().detach()
        logger.info(f"{len(self.txt2clip)} different text cond for {self.split} set")
        logger.info(str(action_txt))
        self.action_txt = action_txt#*4
        self.seqs = seqs#*4
        return
    
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, index) -> Any:
        obj_mesh = self.seqs[index]
        obj_name = os.path.basename(obj_mesh).split('_')[0]
        obj_full_name = os.path.basename(obj_mesh).split('.')[0]
        
        verts_obj = self.object[obj_full_name]
        if verts_obj.shape[0] > 1024:
            verts_sample_id = np.random.choice(verts_obj.shape[0], 1024, replace=False)
        else:
            verts_sample_id = np.arange(verts_obj.shape[0])
        
        action_name = self.action_txt[index]
        text_features = self.txt2clip[action_name]
        data = {
            'feat': torch.zeros((self.max_frame, 15), dtype=torch.float32),
            'latent': torch.zeros((self.max_frame, 64), dtype=torch.float32), 
            'obj_verts': verts_obj[verts_sample_id],
            "text_features": text_features,
            "bps_object": self.bps[obj_full_name],
            'std_feat': self.std,
            'mean_feat': self.mean,
            "obj_name": obj_full_name,
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
    
class OakInkMotionLatentMLD(OakInkMotionLatent):
    
    def __init__(self, root, split='train', max_frame=160):
        self.root = root
        self.split = split
        d = np.load('data/grab/grab_seq_20fps/mean_std.npy', allow_pickle=True).item()
        self.mean = d['mean']
        self.std = d['std']
        # Neurips Submit split 1: oakink 100 object set
        self.seqs = glob.glob('data/oakink/oakink_aligned/*.ply')
        # Neurips Submit split 2: oakink colored object set
        # self.seqs = glob.glob('data/oakink/selected_texture_uv/**/*.obj')
        self.max_frame=max_frame

        self.data = [None] * len(self.seqs)
        self.object = {}
        self.bps = self.compute_canon_bps()
        self.compute_clip_embedding()
        print("Seq Count:", self.__len__())
        
    def __getitem__(self, index) -> Any:
        obj_mesh = self.seqs[index]
        obj_name = os.path.basename(obj_mesh).split('_')[0]
        obj_full_name = os.path.basename(obj_mesh).split('.')[0]
        
        verts_obj = self.object[obj_full_name]
        if verts_obj.shape[0] > 1024:
            verts_sample_id = np.random.choice(verts_obj.shape[0], 1024, replace=False)
        else:
            verts_sample_id = np.arange(verts_obj.shape[0])
        
        action_name = self.action_txt[index]
        text_features = self.txt2clip[action_name]
        data = {
            'feat': torch.zeros((self.max_frame, 15), dtype=torch.float32),
            'latent': torch.zeros((1, 512), dtype=torch.float32), 
            'obj_verts': verts_obj[verts_sample_id],
            "text_features": text_features,
            "bps_object": self.bps[obj_full_name],
            'std_feat': self.std,
            'mean_feat': self.mean,
            "obj_name": obj_full_name,
            "action_name": action_name,
        }
        return data

class OakInkMotion(OakInkMotionLatent):
    def __init__(self, root, split='train', max_frame=160):
        self.root = root
        self.split = split
        d = np.load('data/grab/grab_seq_20fps/mean_std.npy', allow_pickle=True).item()
        self.mean = d['mean'].reshape(1, -1)
        self.std = d['std'].reshape(1, -1)
        self.seqs = glob.glob('data/oakink/oakink_aligned/*.ply')
        self.max_frame=max_frame

        self.data = [None] * len(self.seqs)
        self.object = {}
        self.bps = self.compute_canon_bps()
        self.compute_clip_embedding()
        print("Seq Count:", self.__len__())

    def __getitem__(self, index) -> Any:
        obj_mesh = self.seqs[index]
        obj_name = os.path.basename(obj_mesh).split('_')[0]
        obj_full_name = os.path.basename(obj_mesh).split('.')[0]
        
        verts_obj = self.object[obj_full_name]
        if verts_obj.shape[0] > 1024:
            verts_sample_id = np.random.choice(verts_obj.shape[0], 1024, replace=False)
        else:
            verts_sample_id = np.arange(verts_obj.shape[0])
        
        action_name = self.action_txt[index]
        text_features = self.txt2clip[action_name]
        data = {
            'feat': torch.zeros((self.max_frame, 75), dtype=torch.float32),
            # 'latent': torch.zeros((self.max_frame, 64), dtype=torch.float32), 
            'obj_verts': verts_obj[verts_sample_id],
            "text_features": text_features,
            "bps_object": self.bps[obj_full_name],
            'std_feat': self.std,
            'mean_feat': self.mean,
            "obj_name": obj_full_name,
            "action_name": action_name,
        }
        return data

@DATASET_REGISTRY.register()
def oakink_fake_motion_mld(cfg, split):
    dataset = OakInkMotionLatentMLD("data/oakink/", split=split, max_frame=160)
    return dataset

@DATASET_REGISTRY.register()
def oakink_fake_motion_latent(cfg, split):
    dataset = OakInkMotionLatent("data/oakink/", split=split, max_frame=160)
    return dataset

@DATASET_REGISTRY.register()
def oakink_fake_motion(cfg, split):
    dataset = OakInkMotion("data/oakink/", split=split, max_frame=160)
    return dataset