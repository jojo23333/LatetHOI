# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#

import numpy as np
import torch
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
import os
import tqdm
import time
import glob
import numpy as np
import torch
from d2.data.build import DATASET_REGISTRY

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()

class LoadData(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 ds_name='train',
                 dtype=torch.float32,
                 rot_aug = True):

        super().__init__()

        self.ds_path = os.path.join(dataset_dir, ds_name)
        self.rot_aug = rot_aug if ds_name == 'train' else False
        print(f"Dsname: {ds_name}, Rot Aug: {self.rot_aug}")
        frame_names = self.get_seqs(dataset_dir, split=ds_name)
        print(frame_names[:10])
        print(frame_names[-10:])

        self.is_rhand = np.asarray([os.path.basename(fname).startswith('r_') for fname in frame_names])
        self.frame_names = np.asarray(frame_names)
        # self.frame_sbjs = np.asarray([name.split('/')[-3] for name in self.frame_names])
        # self.frame_objs = np.asarray([name.split('/')[-2].split('_')[0] for name in self.frame_names])
        self.rhand_offset = torch.tensor([0.0957, 0.0064, 0.0062])

    
    def get_seqs(self, root, split='train'):
        # all_seqs = glob.glob(os.path.join(root, 'train/s*/**/*.npy')) +\
        #     glob.glob(os.path.join(root, 'val/s*/**/*.npy'))+\
        #     glob.glob(os.path.join(root, 'test/s*/**/*.npy'))
        # if split == 'train':
        #     seqs = [seq for seq in all_seqs if 's10' not in seq]
        # else:
        #     seqs = [seq for seq in all_seqs if 's10' in seq]
        seqs = glob.glob(os.path.join(root, split, 's*/**/*.npy'))
        if split == 'train':
            seqs_oakink = list(glob.glob(os.path.join(root, split, 'oakink-train/*.npy'))) * 5
            print(f"OAKINKx5: {len(seqs_oakink)}")
            print(f"GRABNET: {len(seqs)}")
            return seqs + seqs_oakink
        else:
            return seqs

    def analysis(self, frames):
        left_contact = 0
        right_contact = 0
        for seq in tqdm.tqdm(frames):
            data = np.load(seq, allow_pickle=True).item()
            if data['has_contact']:
                if data['is_right']:
                    right_contact += 1
                else:
                    left_contact += 1
        print(f'Left: {left_contact}, Right: {right_contact} Out of {len(frames)}')

    def _np2torch(self,ds_path):
        data = np.load(ds_path, allow_pickle=True).item()
        data_torch = {k:torch.tensor(data[k]) for k in data.keys()}
        return data_torch

    def load_disk(self,idx):
        assert isinstance(idx, int)
        return self._np2torch(self.frame_names[idx])

    def __len__(self):
        return len(self.frame_names)

    def rot_augmentation(self, data_out):
        from smplx.lbs import batch_rodrigues
        orient = torch.FloatTensor(1, 3).uniform_(-torch.pi, torch.pi)
        orient[:, :-1] = 0
        rot_mats = batch_rodrigues(orient.view(-1, 3)).view(3, 3).to(torch.float32)
        data_out['verts_object'] = torch.matmul(data_out['verts_object'], rot_mats).to(torch.float32)
        data_out['verts_rhand'] = torch.matmul(data_out['verts_rhand'], rot_mats).to(torch.float32)
        data_out['trans_rhand'] = torch.matmul(data_out['trans_rhand'].view(1, 3)+self.rhand_offset.view(1, 3), rot_mats).to(torch.float32).squeeze() - self.rhand_offset
        data_out['global_orient_rhand_rotmat'] = torch.matmul(rot_mats.transpose(0, 1), data_out['global_orient_rhand_rotmat'].view(3, 3)).to(torch.float32)
        
        # TODO adding this for up and down rotation augmentation
        orient = torch.FloatTensor(1, 3).uniform_(-torch.pi/4, torch.pi/4)
        orient[:, 1:] = 0
        rot_mats = batch_rodrigues(orient.view(-1, 3)).view(3, 3).to(torch.float32)
        data_out['verts_object'] = torch.matmul(data_out['verts_object'], rot_mats).to(torch.float32)
        data_out['verts_rhand'] = torch.matmul(data_out['verts_rhand'], rot_mats).to(torch.float32)
        data_out['trans_rhand'] = torch.matmul(data_out['trans_rhand'].view(1, 3)+self.rhand_offset.view(1, 3), rot_mats).to(torch.float32).squeeze() - self.rhand_offset
        data_out['global_orient_rhand_rotmat'] = torch.matmul(rot_mats.transpose(0, 1), data_out['global_orient_rhand_rotmat'].view(3, 3)).to(torch.float32)

        return data_out

    def __getitem__(self, idx):
        import copy
        data_out = self.load_disk(idx)
        for k in data_out.keys():
            if k not in ['has_contact', 'is_right']:
                data_out[k] = data_out[k].to(torch.float32)
            else:
                data_out[k] = data_out[k].to(torch.int)

        data_out['trans_rhand'] = data_out['trans_rhand'].squeeze()
        data_out['frame_name'] = self.frame_names[idx]
        data_out.pop('root_orient_obj_rotmat')
        # data_out
        # if isinstance(idx, int):
        #     # is_right = data_out['is_right']
        #     data_out['verts_rhand'] = (data_out['verts_rhand'] - data_out['trans_obj'].reshape(1,3)).to(torch.float32)
        #     data_out['verts_object'] = (data_out['verts_object'] - data_out['trans_obj'].reshape(1,3)).to(torch.float32)
            
        #     data_out['frame_name'] = self.frame_names[idx]
        #     data_out['frame_id'] = int(os.path.basename(self.frame_names[idx]).split('.')[0].split('_')[-1])
        # else:
        #     data_out['verts_rhand'] = (data_out['verts_rhand'] - data_out['trans_obj'].reshape(-1,1,3)).to(torch.float32)
        #     data_out['verts_object'] = (data_out['verts_object'] - data_out['trans_obj'].reshape(-1,1,3)).to(torch.float32)
        if self.rot_aug:
            data_out = self.rot_augmentation(data_out)
            
        if data_out['verts_object'].shape[0] > 1024:
            verts_sample_id = np.random.choice(data_out['verts_object'].shape[0], 1024, replace=False)
            assert verts_sample_id.shape[0] == 1024
            verts_sample_id = torch.from_numpy(verts_sample_id)
            data_out['verts_object'] = data_out['verts_object'][verts_sample_id]
            
        if data_out['verts_object'].shape[0] < 1024:
            verts_sample_id = np.random.choice(data_out['verts_object'].shape[0], 1024, replace=True)
            assert verts_sample_id.shape[0] == 1024
            verts_sample_id = torch.from_numpy(verts_sample_id)
            data_out['verts_object'] = data_out['verts_object'][verts_sample_id]
        
        # for k, v in data_out.items():
        #     print(k, v.shape)
        return data_out

    def get_seq_ids(self):
        for seq_name in self.seqs:
            yield seq_name, self.seqs[seq_name]
    
    def sanity_check(self):
        # randomly visualize 10 frame for sanity check
        import mano, trimesh, smplx
        from psbody.mesh.colors import name_to_rgb
        # rhm_train = mano.load(model_path='./data/body_models/mano',
        #                         model_type='mano',
        #                         num_pca_comps=45,
        #                         batch_size=1,
        #                         flat_hand_mean=True)
        rhm_train = smplx.create(model_path='./data/body_models',
                                model_type='mano',
                                is_rhand=True,
                                # v_template=rh_vtemp,
                                # num_pca_comps=45,
                                use_pca=False,
                                flat_hand_mean=True,
                                batch_size=1)
        np.random.seed(2024)
        # torch.manual_seed(2024)
        idx = np.random.choice(range(self.__len__()), 10, replace=False)
        cnt = 0
        for i in idx:
            cnt = cnt + 1
            if cnt == 20:
                import ipdb; ipdb.set_trace()
            data = self.__getitem__(int(i))
            print(data['frame_name'])
            if not data['has_contact']:
                continue
            frame_name = data['frame_name']
            obj_points = data['verts_object'].numpy()
            hand_params = parms_decode(data)
            out1 = rhm_train(**hand_params)
            mesh1 = trimesh.Trimesh(vertices=out1.vertices[0].detach(), faces=rhm_train.faces, face_colors=name_to_rgb['green'])
            hand_params.pop('transl')
            out = rhm_train(**hand_params)
            print(out.joints[0, 0])
            mesh2 = trimesh.Trimesh(vertices=out.vertices[0].detach(), faces=rhm_train.faces, face_colors=name_to_rgb['pink'])
            #
            obj = trimesh.PointCloud(vertices=obj_points, colors=name_to_rgb['gray'])
            mesh3 = trimesh.Trimesh(vertices=data['verts_rhand'], faces=rhm_train.faces, face_colors=name_to_rgb['cyan'])
            obj.export(file_obj='.exps/debug/'+frame_name.replace('/', '_')+'_obj.ply', file_type='ply')
            mesh1.export(file_obj='.exps/debug/'+frame_name.replace('/', '_')+'_param.ply', file_type='ply')
            mesh2.export(file_obj='.exps/debug/'+frame_name.replace('/', '_')+'_param_no_trans.ply', file_type='ply')
            mesh3.export(file_obj='.exps/debug/'+frame_name.replace('/', '_')+'_verts.ply', file_type='ply')

class DexycbPose(LoadData):
    def rot_augmentation(self, data_out):
        from smplx.lbs import batch_rodrigues
        orient = torch.FloatTensor(1, 3).uniform_(-torch.pi, torch.pi)
        orient[:, :-1] = 0
        rot_mats = batch_rodrigues(orient.view(-1, 3)).view(3, 3).to(torch.float32)
        data_out['verts_object'] = torch.matmul(data_out['verts_object'], rot_mats).to(torch.float32)
        data_out['verts_rhand'] = torch.matmul(data_out['verts_rhand'], rot_mats).to(torch.float32)
        data_out['trans_rhand'] = torch.matmul(data_out['trans_rhand'].view(1, 3)+self.rhand_offset.view(1, 3), rot_mats).to(torch.float32).squeeze() - self.rhand_offset
        data_out['global_orient_rhand_rotmat'] = torch.matmul(rot_mats.transpose(0, 1), data_out['global_orient_rhand_rotmat'].view(3, 3)).to(torch.float32)
        
        # TODO adding this for up and down rotation augmentation
        orient = torch.FloatTensor(1, 3).uniform_(-torch.pi, torch.pi)
        orient[:, 1:] = 0
        rot_mats = batch_rodrigues(orient.view(-1, 3)).view(3, 3).to(torch.float32)
        data_out['verts_object'] = torch.matmul(data_out['verts_object'], rot_mats).to(torch.float32)
        data_out['verts_rhand'] = torch.matmul(data_out['verts_rhand'], rot_mats).to(torch.float32)
        data_out['trans_rhand'] = torch.matmul(data_out['trans_rhand'].view(1, 3)+self.rhand_offset.view(1, 3), rot_mats).to(torch.float32).squeeze() - self.rhand_offset
        data_out['global_orient_rhand_rotmat'] = torch.matmul(rot_mats.transpose(0, 1), data_out['global_orient_rhand_rotmat'].view(3, 3)).to(torch.float32)

        return data_out
    
    def get_seqs(self, root, split='train'):
        return glob.glob(os.path.join(root, split, '**/*.npy')) 

class DexycbOakinkPose(DexycbPose):
    def get_seqs(self, root, split='train'):
        seqs = glob.glob(os.path.join(root, split, '**/*.npy'))
        if split == 'train':
            seqs_oakink = list(glob.glob('data/grab/grab_frames/train/oakink-train/*.npy'))
            print(f"OAKINK: {len(seqs_oakink)}")
            print(f"GRABNET: {len(seqs)}")
            return seqs + seqs_oakink
        else:
            return seqs


class GraspXL(LoadData):
    
    def __init__(self,
                 dataset_dir,
                 ds_name='train',
                 dtype=torch.float32,
                 rot_aug = True):
        self.dataset_dir = dataset_dir
        self.ds_path = os.path.join(dataset_dir, ds_name)
        self.rot_aug = rot_aug if ds_name == 'train' else False
        print(f"Dsname: {ds_name}, Rot Aug: {self.rot_aug}")
        frame_names = self.get_seqs(dataset_dir, split=ds_name)
        print(frame_names[:100])

        self.frame_names = np.asarray(frame_names)
        self.obj_info = {}
        # for obj in os.listdir(os.path.join(dataset_dir, 'objects')):
        #     obj_name = obj.split('.')[0]
        #     obj_verts = np.load(os.path.join(dataset_dir, 'objects', obj))
        #     self.obj_info[obj_name] = torch.from_numpy(obj_verts).to(torch.float32)

    def clean(self):
        from tqdm import tqdm
        for i in tqdm(self.frame_names):
            x = np.load(i, allow_pickle=True).item()
            x['has_contact'] = x['has_contact'].to('cpu').to(torch.int32).item()
            np.save(i, x, allow_pickle=True)

    def get_seqs(self, root, split='train'):
        all_seqs = glob.glob(os.path.join(root, '**/**/*.npy')) 
        if split == 'train':
            seqs = all_seqs[5000:]
        else:
            seqs = all_seqs[:5000]
        return seqs

    def __getitem__(self, idx):
        data = self._np2torch(self.frame_names[idx])
        obj_name = self.frame_names[idx].split('/')[-3]
        obj_verts = torch.from_numpy(
            np.load(os.path.join(self.dataset_dir, 'objects', obj_name+'.npy'))
        ).to(torch.float32)
        # obj_verts = self.obj_info[obj_name]
        
        verts_sample_id = np.random.choice(obj_verts.shape[0], 1024, replace=False)
        data['verts_obj'] = obj_verts[verts_sample_id]
        data['has_contact'] = data['has_contact'].to('cpu').to(torch.int32)
        return data

    def sanity_check(self):
        # randomly visualize 10 frame for sanity check
        import mano, trimesh, smplx
        from psbody.mesh.colors import name_to_rgb
        rhm_train = smplx.create(model_path='./data/body_models',
                                model_type='mano',
                                is_rhand=True,
                                use_pca=False,
                                flat_hand_mean=True,
                                batch_size=1)
        np.random.seed(2024)
        idx = np.random.choice(range(self.__len__()), 10, replace=False)
        for i in idx:
            data = self.__getitem__(int(i))
            if not data['has_contact']:
                continue
            # import ipdb; ipdb.set_trace()
            frame_name = data['frame_name']
            print(data['frame_name'])
            obj_points = data['verts_object'].numpy()
            hand_params = parms_decode(data)
            out = rhm_train(**hand_params)
            #
            obj = trimesh.PointCloud(vertices=obj_points, colors=name_to_rgb['gray'])
            mesh1 = trimesh.Trimesh(vertices=out.vertices[0].detach(), faces=rhm_train.faces, face_colors=name_to_rgb['green'])
            mesh2 = trimesh.Trimesh(vertices=data['verts_rhand'], faces=rhm_train.faces, face_colors=name_to_rgb['pink'])
            obj.export(file_obj='.exps/debug/'+frame_name.replace('/', '_')+'_obj.ply', file_type='ply')
            mesh1.export(file_obj='.exps/debug/'+frame_name.replace('/', '_')+'_param.ply', file_type='ply')
            mesh2.export(file_obj='.exps/debug/'+frame_name.replace('/', '_')+'_vert.ply', file_type='ply')

from modeling.vae.grabnet import rotmat2aa

def parms_decode(data):

    pose_full = torch.concatenate(
        [data['global_orient_rhand_rotmat'].view(1, 3, 3), data['fpose_rhand_rotmat']], dim=0
    )
    trans= data['trans_rhand'].view(1, 3)
    
    pose = pose_full.view([1, 1, -1, 9])
    pose = rotmat2aa(pose).view(1, -1)

    global_orient = pose[:, :3]
    hand_pose = pose[:, 3:]
    pose_full = pose_full.view([1, -1, 3, 3])

    hand_parms = {'global_orient': global_orient, 'hand_pose': hand_pose, 'transl': trans}#, 'fullpose': pose_full}

    return hand_parms

@DATASET_REGISTRY.register()
def grab_pose(cfg, split):
    dataset = LoadData("data/grab/grab_frames", ds_name=split)
    return dataset

@DATASET_REGISTRY.register()
def dexycb_pose(cfg, split):
    dataset = DexycbPose("data/dexYCB/dexycb_grab", ds_name=split)
    return dataset

@DATASET_REGISTRY.register()
def dexycb_oakink_pose(cfg, split):
    dataset = DexycbOakinkPose("data/dexYCB/dexycb_grab", ds_name=split)
    return dataset

@DATASET_REGISTRY.register()
def graspxl(cfg, split):
    dataset = GraspXL("data/graspXL/large_processed", ds_name=split)
    return dataset

if __name__=='__main__':
    # data_path = 'data/grab/grab_frames'
    # data_path = 'data/graspXL/large_processed'
    # ds = GraspXL(data_path, ds_name='train')
    # ds.clean()
    # ds = GraspXL(data_path, ds_name='test')
    # ds.clean()
    ds = DexycbPose("data/dexYCB/dexycb_grab", ds_name='train')
    ds.sanity_check()
    # data_path = 'data/graspXL/large_processed'
    # grasp = GraspXL(data_path, ds_name='train', only_params=False)
    # grasp.sanity_check()
