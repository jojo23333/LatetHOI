
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

import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import os, glob
import smplx
import argparse
import copy

from tqdm import tqdm
from tools.objectmodel import ObjectModel
from tools.cfg_parser import Config
from tools.utils import makepath, makelogger
from tools.meshviewer import Mesh
from tools.utils import parse_npz
from tools.utils import params2torch
from tools.utils import prepare_params
from tools.utils import to_cpu
from tools.utils import append2dict
from tools.utils import np2torch
from bps_torch.bps import bps_torch
from smplx.lbs import batch_rodrigues
from tqdm import tqdm
import trimesh

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INTENTS = ['lift', 'pass', 'offhand', 'use', 'all']

def solve_hand(self, cfg, vtemp, n_comps, T, h_params, is_right):
    with torch.no_grad():
        hand_model_vtemp = smplx.create(
            model_path=cfg.model_path,
            model_type='mano',
            is_rhand=is_right,
            v_template=vtemp,
            num_pca_comps=n_comps,
            flat_hand_mean=True,
            batch_size=T
        )

        h_params = params2torch(h_params)
        h_out = hand_model_vtemp(return_full_pose=True, **h_params)
        verts = to_cpu(h_out.vertices).detach()
    full_pose = h_out.full_pose.cpu().detach()
    transl = h_params['transl'].cpu().detach()
    

    ################################################
    # Verification Debug
    hand_model = smplx.create(
        model_path=cfg.model_path,
        model_type='mano',
        is_rhand=False,
        use_pca=False,
        flat_hand_mean=True,
        batch_size=T
    ).to(device)

    glob_rot = full_pose[:, :3].clone()#.requires_grad_(True)
    joint_rot = full_pose[:, 3:].clone().requires_grad_(True)
    transl = transl.clone().requires_grad_(True)
    verts = verts.to(device)
    
    optimizer = torch.optim.Adam([{'params': transl, 'lr': 0.01},
                                    {'params': joint_rot, 'lr': 0.01}])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    losses = []
    for i in range(1000):
        out = hand_model(
            global_orient=glob_rot.to(device),
            hand_pose=joint_rot.to(device),
            transl=transl.to(device)
        )
        loss = (out.vertices - verts).abs().mean()
        if loss.item() < 1e-5:
            break
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        scheduler.step()
        if loss.item() < 5e-4:
            break
        print(i, loss, optimizer.param_groups[0]['lr'])
    if loss.item() > 1e-3:
        logger.warning("Probably not converged")
    logger.info(f'Hand Solver: {i} iterations, loss: {loss.item()}')
    return glob_rot.cpu().detach().numpy(),\
        joint_rot.cpu().detach().numpy(),\
        transl.cpu().detach().numpy(),\
        out

GAP = 5
TOTOAL_SPLIT=10
class GRABDataSet(object):

    def __init__(self, cfg, logger=None, **params):

        self.cfg = cfg
        self.out_path = cfg.out_path
        self.data_path = cfg.data_path
        makepath(self.out_path)

        if logger is None:
            log_dir = os.path.join(self.out_path, 'grab_preprocessing.log')
            self.logger = makelogger(log_dir=log_dir, mode='a').info
        else:
            self.logger = logger
        self.logger('Starting data preprocessing !')

        self.obj_folders = [os.path.join(self.data_path, x) for x in os.listdir(self.data_path)]

        # process the data
        self.data_preprocessing(cfg)

    def data_preprocessing(self, cfg):
        self.subject_mesh = {}
        self.obj_info = {}
        self.sbj_info = {}
        split = cfg.split
        obj_folders = self.obj_folders[split::TOTOAL_SPLIT]
        self.logger(f'Processing Split {split} with {len(obj_folders)} objects out of {len(self.obj_folders)}')
        for obj_folder in tqdm(obj_folders):
            obj_name = os.path.basename(obj_folder)
            obj_mesh_path = os.path.join(self.data_path, obj_name, f'{obj_name}.obj')
            obj = trimesh.load_mesh(obj_mesh_path)
            obj_verts, face_idx = trimesh.sample.sample_surface(obj, 2048)
            obj_verts = obj_verts.reshape(-1, 3)
            savedir = os.path.join(self.out_path, 'objects')
            os.makedirs(savedir, exist_ok=True)
            np.save(os.path.join(savedir, f'{obj_name}.npy'), obj_verts)
        
            seqs = [x for x in os.listdir(obj_folder) if x.endswith('.npy')]
            for seq in seqs:
                seq_data = np.load(os.path.join(obj_folder, seq), allow_pickle=True).item()
                hand_params = {
                    'global_orient': torch.from_numpy(seq_data['right_hand']['rot'][::GAP]).to(device),
                    'hand_pose': torch.from_numpy(seq_data['right_hand']['pose'][::GAP]).to(device),
                    'transl': torch.from_numpy(seq_data['right_hand']['trans'][::GAP]).to(device),
                }
                r_trans = hand_params['transl'].cpu()
                T = r_trans.shape[0]
                
                rhand_model = smplx.create(
                    model_path=cfg.model_path,
                    model_type='mano',
                    is_rhand=True,
                    use_pca=False,
                    batch_size=T
                ).to(device)
                out = rhand_model(return_full_pose=True, **hand_params)
                verts_rh = out.vertices
                kp_rh = out.joints.cpu().detach()
                r_full_pose = out.full_pose.cpu().detach()
                # r_full_pose = batch_rodrigues(
                #         r_full_pose.view(-1, 3)
                # ).view(-1, 16, 3, 3)
                
                # rhand_model = smplx.create(
                #     model_path=cfg.model_path,
                #     model_type='mano',
                #     is_rhand=True,
                #     use_pca=False,
                #     batch_size=T,
                #     flat_hand_mean=True
                # ).to(device)
                # out1 = rhand_model(
                #     global_orient=out.full_pose.cpu().detach()[:, :3].to(device),
                #     hand_pose=out.full_pose.cpu().detach()[:, 3:].to(device),
                #     transl=hand_params['transl']
                # )
                # assert (out1.vertices - out.vertices < 1e-5).all()
                # import ipdb; ipdb.set_trace()
                obj_params = {
                    "global_orient": torch.from_numpy(seq_data[obj_name]['rot'][::GAP]).to(device),
                    "transl": torch.from_numpy(seq_data[obj_name]['trans'][::GAP]).to(device),
                    "rot_transpose": True
                }
                if seq_data[obj_name]['angle'].sum(0) > 0:
                    print(obj_name, seq)
                    continue

                obj_m = ObjectModel(v_template=obj_verts, batch_size=T).to(device)
                verts_obj = to_cpu(obj_m(**obj_params).vertices).cpu().detach()
                trans_obj = obj_params['transl'].cpu().detach()
                rot_obj = obj_params['global_orient'].cpu()
                # verts_no_trans = to_cpu(obj_m(global_orient=rot_obj).vertices)
                dis = (verts_rh.view(T, -1, 1, 3) - verts_obj.view(T, 1, -1, 3).to(verts_rh)).norm(dim=-1)
                # p9 of https://arxiv.org/pdf/2008.11200
                rhand_contact = dis.view(T, -1).min(dim=-1)[0] < 0.0045
                
                verts_rh = verts_rh.cpu().detach()
                verts_obj = verts_obj.cpu().detach()
                r_full_pose = r_full_pose.cpu().detach()
                # from psbody.mesh.colors import name_to_rgb
                # for i in range(0, T, 6):
                #     path = os.path.join('../data/graspXL/debug', obj_name, seq)
                #     os.makedirs(path, exist_ok=True)

                #     mesh = trimesh.Trimesh(vertices=verts_rh[i].cpu() - trans_obj[i], faces=rhand_model.faces, face_colors=name_to_rgb['cyan'])
                #     mesh.export(f'{path}/{i}_rhand.ply')

                #     mesh = trimesh.points.PointCloud(vertices=verts_obj[i] - trans_obj[i], vertex_colors=name_to_rgb['green'])
                #     mesh.export(f'{path}/{i}_obj.ply')

                seq_name = seq.split('.')[0]
                for i in range(0, T):
                    export_path = os.path.join(self.out_path, obj_name, seq_name, f"r_{i*GAP}.npy")
                    dir_name = os.path.dirname(export_path)
                    os.makedirs(dir_name, exist_ok=True)
                    data = {
                        'trans_obj': trans_obj[i].to(torch.float32).numpy(),
                        'rot_obj': rot_obj[i].to(torch.float32).numpy(),
                        # 'root_orient_obj_rotmat': rot_obj[i].to(torch.float32).numpy(),
                        # 'verts_object': verts_obj[i].to(torch.float32).numpy(),
                        'full_pose': r_full_pose[i].to(torch.float32).numpy(),
                        # 'global_orient_rhand_rotmat': r_full_pose[i,0].to(torch.float32).numpy(),
                        # 'fpose_rhand_rotmat': r_full_pose[i,1:].to(torch.float32).numpy(),
                        'trans_rhand': r_trans[i].to(torch.float32).numpy(),
                        # 'verts_rhand': verts_rh[i].to(torch.float32).numpy(),
                        # 'kp_rhand': kp_rh[i].to(torch.float32).numpy(),
                        "has_contact": rhand_contact[i].item(),
                        "is_right": 1,
                    }
                    # data['verts_rhand'] = data['verts_rhand'] - data['trans_obj'].reshape(1, 3)
                    # data['verts_object'] = data['verts_object'] - data['trans_obj'].reshape(1, 3)
                    # data['kp_rhand'] = data['kp_rhand'] - data['trans_obj'].reshape(1, 3)
                    # data['trans_rhand'] = data['trans_rhand'] - data['trans_obj']
                    np.save(
                        export_path,
                        data,
                        allow_pickle=True
                    )



if __name__ == '__main__':


    instructions = ''' 
    Please do the following steps before starting the GRAB dataset processing:
    1. Download GRAB dataset from the website https://grab.is.tue.mpg.de/ 
    2. Set the grab_path, out_path to the correct folder
    3. Change the configuration file for your desired data, like:
    
        a) if you only need the frames with contact,
        b) if you need body, hand, or object vertices to be computed,
        c) which data splits
            and etc
        
        WARNING: saving vertices requires a high-capacity RAM memory.
        
    4. In case you need body or hand vertices make sure to set the model_path
        to the models downloaded from smplx website 
    
    This code will process the data and save the pt files in the out_path folder.
    You can use the dataloader.py file to load and use the data.    

        '''

    parser = argparse.ArgumentParser(description='GRAB-vertices')

    parser.add_argument('--grab-path', required=False, type=str,
                        help='The path to the downloaded grab data')
    parser.add_argument('--out-path', default=None, type=str,
                        help='The path to the folder to save the processed data')
    parser.add_argument('--model-path', required=False, type=str,
                        help='The path to the folder containing smplx models')
    parser.add_argument("--split", type=int, default=0)

    args = parser.parse_args()

    data_path = "../data/graspXL/large" #args.grab_path
    out_path = "../data/graspXL/large_processed"
    model_path = "../data/body_models" #args.model_path

    cfg = {

        'intent':'all', # from 'all', 'use' , 'pass', 'lift' , 'offhand'
        'only_contact':True, # if True, returns only frames with contact
        'save_body_verts': False, # if True, will compute and save the body vertices
        'save_lhand_verts': True, # if True, will compute and save the body vertices
        'save_rhand_verts': True, # if True, will compute and save the body vertices
        'save_object_verts': True,

        'save_contact': True, # if True, will add the contact info to the saved data

        #IO path
        'data_path': data_path,
        'out_path': out_path,

        # number of vertices samples for each object
        'n_verts_sample': 1024,

        # body and hand model path
        'model_path': model_path,
        "split": args.split
    }

    cwd = os.getcwd()
    default_cfg_path = os.path.join(cwd, '../configs/grab_preprocessing_cfg.yaml')
    cfg = Config(default_cfg_path=default_cfg_path, **cfg)
    cfg.write_cfg(write_path=cfg.out_path+'/grab_preprocessing_cfg.yaml')

    log_dir = os.path.join(cfg.out_path, 'grab_processing.log')
    logger = makelogger(log_dir=log_dir, mode='a').info

    logger(instructions)

    GRABDataSet(cfg, logger)

