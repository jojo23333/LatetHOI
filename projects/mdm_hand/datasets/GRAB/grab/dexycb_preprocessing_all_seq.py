
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
# For dexycb to work, one need to comment out Line 1583 1584 of smplx body_models

import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import os, glob
import smplx
import argparse
import clip

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
from scipy.spatial.transform import Rotation as Rot
# from bps_torch.bps import bps_torch
import yaml
from smplx.lbs import batch_rodrigues
import copy

_YCB_CLASSES = {
     1: '002_master_chef_can',
     2: '003_cracker_box',
     3: '004_sugar_box',
     4: '005_tomato_soup_can',
     5: '006_mustard_bottle',
     6: '007_tuna_fish_can',
     7: '008_pudding_box',
     8: '009_gelatin_box',
     9: '010_potted_meat_can',
    10: '011_banana',
    11: '019_pitcher_base',
    12: '021_bleach_cleanser',
    13: '024_bowl',
    14: '025_mug',
    15: '035_power_drill',
    16: '036_wood_block',
    17: '037_scissors',
    18: '040_large_marker',
    19: '051_large_clamp',
    20: '052_extra_large_clamp',
    21: '061_foam_brick',
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INTENTS = ['lift', 'pass', 'offhand', 'use', 'all']
MODEL_DIR = "../../data/body_models"

mano_layers = {
    "right": smplx.create(
                model_path=MODEL_DIR,
                model_type='mano',
                is_rhand=True,
                num_pca_comps=45,
                flat_hand_mean=False),
    "left": smplx.create(
                model_path=MODEL_DIR,
                model_type='mano',
                is_rhand=False,
                num_pca_comps=45,
                flat_hand_mean=False),
}

mano_layers_flat = {
    "right": smplx.create(
                model_path=MODEL_DIR,
                model_type='mano',
                is_rhand=True,
                use_pca=False,
                flat_hand_mean=True),
    "left": smplx.create(
                model_path=MODEL_DIR,
                model_type='mano',
                is_rhand=False,
                use_pca=False,
                flat_hand_mean=True),
}

def solve_hand(hand_model, verts, full_pose):
    model = copy.deepcopy(hand_model).to(device)
    glob_rot = full_pose[:, :3].clone().requires_grad_(True)
    joint_rot = full_pose[:, 3:48].clone().requires_grad_(True)
    transl = full_pose[:, 48:].clone().requires_grad_(True)
    betas = torch.zeros((transl.shape[0], 10)).to(transl).to(device)
    verts = verts.to(device)

    optimizer = torch.optim.Adam([{'params': transl, 'lr': 0.01},
                                    {'params': joint_rot, 'lr': 0.01},
                                     {'params': glob_rot, 'lr': 0.01}])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    losses = []
    for i in range(400):
        out = model(
            betas=betas,
            global_orient=glob_rot.to(device),
            hand_pose=joint_rot.to(device),
            transl=transl.to(device)
        )
        loss = ((out.vertices - verts)**2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        scheduler.step()
        if loss.sqrt().item() < 1.5e-3:
            break
        # print(i, loss, optimizer.param_groups[0]['lr'])
    if loss.item() > 1e-3:
        print("Probably not converged")
    print(f'Hand Solver: {i} iterations, loss: {loss.item()}')
    full_pose = torch.cat([glob_rot, joint_rot], dim=-1)
    return full_pose.cpu().detach(), transl.cpu().detach()


def get_hand_grasp(pose_file, mano_side, obj_id, betas, obj_v_canon):
    data = np.load(pose_file)
    pose_ycb = data['pose_y'][:, obj_id, :]
    # Process Object Transformation
    q = pose_ycb[:, :4]
    obj_rot = Rot.from_quat(q).as_matrix().astype(np.float32)
    obj_trans = pose_ycb[:, 4:].astype(np.float32)

    # Rule out invalid frames
    hand_pose = data['pose_m'][:,0,:]
    valid_frame = np.where(hand_pose.sum(axis=1) != 0)[0]
    hand_pose = hand_pose[valid_frame]
    obj_rot = obj_rot[valid_frame]
    obj_trans = obj_trans[valid_frame]
    
    betas = torch.tensor(betas).unsqueeze(0).repeat(hand_pose.shape[0], 1)
    print("Side: ", mano_side)
    
    global_orient=torch.from_numpy(hand_pose[:, :3])
    transl=torch.from_numpy(hand_pose[:, 48:])
    hand_pose=torch.from_numpy(hand_pose[:, 3:48])
    out = mano_layers[mano_side](betas=betas,
                                 global_orient=global_orient,
                                 hand_pose=hand_pose,
                                 transl=transl,
                                 return_full_pose=True)

    obj_v = torch.matmul(
        torch.from_numpy(obj_v_canon).reshape(1, -1, 3),
        torch.from_numpy(obj_rot).reshape(-1, 3, 3).transpose(1, 2)
    ) + torch.from_numpy(obj_trans).reshape(-1, 1, 3)
    hand_v = out.vertices.detach().cpu()
    hand_j = out.joints.detach().cpu()
    obj_rot = torch.from_numpy(obj_rot).reshape(-1, 3, 3).transpose(1, 2)

    full_pose = out.full_pose.detach()
    full_pose = torch.cat([full_pose, transl], dim=1)
    
    out = mano_layers_flat[mano_side](betas=betas,
                                 global_orient=full_pose[:, :3],
                                 hand_pose=full_pose[:, 3:48],
                                 transl=full_pose[:, 48:])
    hand_v_1 = out.vertices.detach().cpu()
    hand_j_1 = out.joints.detach().cpu()
    assert (hand_v_1- hand_v < 1e-6).all()
    assert (hand_j_1 - hand_j < 1e-6).all()

    full_pose, transl = solve_hand(mano_layers_flat[mano_side], hand_v, full_pose)
    full_pose = torch.cat([full_pose, transl], dim=1)
    #########################################################################################################################
    # debug part
    # from utils.rotation_conversions import rotation_6d_to_matrix, axis_angle_to_rotation_6d, rotation_6d_to_axis_angle
    # from modeling.vae.helpers import DistMatrixSolver, RotMatSolver, HandSolver
    # from scipy.spatial.transform import Rotation as R
    # from smplx.lbs import batch_rodrigues
    # global_orient_6d = axis_angle_to_rotation_6d(full_pose[:, :3])
    # hand_pose_6d = axis_angle_to_rotation_6d(full_pose[:, 3:48].reshape(-1, 15, 3))
    # hand_pose_matrix = rotation_6d_to_matrix(hand_pose_6d).view(-1, 15, 3, 3)
    # hand_pose_matrix2 = batch_rodrigues(full_pose[:, 3:48].reshape(-1, 3)).view(-1, 15, 3, 3)
    
    # hand_pose = axis_angle_to_rotation_6d(full_pose[:, 3:48].reshape(-1, 3)).view(-1, 15, 6)
    # solver = HandSolver(betas, mano_side=='right')
    # import ipdb; ipdb.set_trace()
    # solver.solve(hand_pose, out.joints)
    #########################################################################################################################
    
    #########################################################################################################################
    # import trimesh
    # for i in range(0, 50):
    #     path = os.path.join('../../data/dexYCB/debug', f'i')
    #     os.makedirs(path, exist_ok=True)
    #     mesh = trimesh.Trimesh(vertices=hand_v[i] - obj_trans[i].reshape(1, 3), faces=mano_layers_flat[mano_side].faces)
    #     mesh.export(f'{path}/{i}_rhand_r.ply')
        
    #     mesh = trimesh.points.PointCloud(vertices=obj_v[i].numpy() - obj_trans[i].reshape(1, 3))
    #     mesh.export(f'{path}/{i}_obj.ply')
    #########################################################################################################################
    
    
    dis_matrix = (hand_v.view(-1, 778, 1, 3) - obj_v.view(-1, 1, 1024, 3).to(hand_v)).norm(dim=-1)
    has_contact = dis_matrix.view(-1, 778*1024).min(dim=-1)[0] < 0.0045

    if mano_side == 'left':
        m_verts_lh = hand_v * torch.tensor([[[-1, 1, 1]]])
        ml_full_pose = batch_rodrigues(
            full_pose[:, :48].reshape(-1, 3).cpu() * torch.tensor([[1, -1, -1]])
        ).view(-1, 16, 3, 3)
        ml_trans = full_pose[:, 48:] * torch.tensor([[-1, 1, 1]])

        verts_obj_mirror = obj_v * torch.tensor([[[-1, 1, 1]]])  
        trans_obj_mirror = torch.from_numpy(obj_trans) * torch.tensor([[-1, 1, 1]], dtype=torch.float32)
        obj_rot[:, 0, :] = obj_rot[:, 0, :] * -1
        obj_rot[:, :, 0] = obj_rot[:, :, 0] * -1
        
        ########################
        # import trimesh
        # obj_v_canon = torch.from_numpy(obj_v_canon).reshape(1, -1, 3) * torch.tensor([[[-1, 1, 1]]])  
        # obj_v = torch.matmul(
        #     obj_v_canon,
        #     obj_rot.reshape(-1, 3, 3)
        # ) + trans_obj_mirror.reshape(-1, 1, 3)
        # out = mano_layers_flat['right'](
        #     betas=torch.zeros((transl.shape[0], 10)),
        #     global_orient=(full_pose[:, :3] * torch.tensor([[1, -1, -1]])),
        #     hand_pose=(full_pose[:, 3:48].reshape(-1, 3) * torch.tensor([[1, -1, -1]])).reshape(-1, 45),
        #     transl=ml_trans
        # )
        # hand_v = out.vertices.reshape(-1, 778, 3).cpu()
        
        # from psbody.mesh.colors import name_to_rgb
        # for i in range(0, 50):
        #     path = os.path.join('../../data/dexYCB/debug', f'{i}')
        #     os.makedirs(path, exist_ok=True)
        #     mesh = trimesh.Trimesh(vertices=m_verts_lh[i] - trans_obj_mirror[i].reshape(1, 3), faces=mano_layers_flat["right"].faces, face_colors=name_to_rgb['cyan'])
        #     mesh.export(f'{path}/{i}_rhand_ml.ply')
            
        #     mesh = trimesh.Trimesh(vertices=hand_v[i] - trans_obj_mirror[i].reshape(1, 3), faces=mano_layers_flat["right"].faces, face_colors=name_to_rgb['pink'])
        #     mesh.export(f'{path}/{i}_rhand_l.ply')
            
        #     mesh = trimesh.points.PointCloud(vertices=obj_v[i] - trans_obj_mirror[i].reshape(1, 3))
        #     mesh.export(f'{path}/{i}_obj.ply')

        #     mesh = trimesh.points.PointCloud(vertices=verts_obj_mirror[i] - trans_obj_mirror[i].reshape(1, 3))
        #     mesh.export(f'{path}/{i}_mobj.ply')
        # import ipdb; ipdb.set_trace()
        ########################
        
        return {
            'trans_obj': trans_obj_mirror.to(torch.float32).numpy(),
            'root_orient_obj_rotmat': obj_rot.to(torch.float32).numpy(),
            'verts_object': verts_obj_mirror.to(torch.float32).numpy(),
            'global_orient_rhand_rotmat': ml_full_pose[:, 0].to(torch.float32).numpy(),
            'fpose_rhand_rotmat': ml_full_pose[:, 1:].to(torch.float32).numpy(),
            'trans_rhand': ml_trans.to(torch.float32).numpy(),
            'verts_rhand': m_verts_lh.to(torch.float32).numpy(),
            "has_contact": has_contact.cpu().to(torch.int32).numpy(),
            "is_right": 1 if mano_side == "right" else 0
        }
    else:
        global_orient_rhand_rotmat = batch_rodrigues(full_pose[:, :3]).reshape(-1, 3, 3)
        fpose_rhand_rotmat = batch_rodrigues(full_pose[:, 3:48].reshape(-1, 3)).reshape(-1, 15, 3, 3)

        return {
            'trans_obj': obj_trans.astype(np.float32),
            'root_orient_obj_rotmat': obj_rot.to(torch.float32).numpy(),
            'verts_object': obj_v.to(torch.float32).numpy(),
            'global_orient_rhand_rotmat': global_orient_rhand_rotmat.to(torch.float32).numpy(),
            'fpose_rhand_rotmat': fpose_rhand_rotmat.to(torch.float32).numpy(),
            'trans_rhand': full_pose[:, 48:].to(torch.float32).numpy(),
            'verts_rhand': hand_v.to(torch.float32).numpy(),
            "has_contact": has_contact.cpu().to(torch.int32).numpy(),
            "is_right": 1 if mano_side == "right" else 0
        }

class GRABDataSet(object):

    def __init__(self, cfg, logger=None, **params):

        self.cfg = cfg
        self.data_path = cfg.data_path
        self.out_path = cfg.out_path
        makepath(self.out_path)

        if logger is None:
            log_dir = os.path.join(self.out_path, 'grab_preprocessing.log')
            self.logger = makelogger(log_dir=log_dir, mode='a').info
        else:
            self.logger = logger
        self.logger('Starting data preprocessing !')

        assert cfg.intent in INTENTS

        self.intent = cfg.intent
        self.logger('intent:%s --> processing %s sequences!' % (self.intent, self.intent))
        self.splits = cfg.splits
            
        self.all_seqs = glob.glob(os.path.join(cfg.data_path, '**/**/pose.npz'))
        
        ## to be filled 
        self.selected_seqs = []
        self.obj_based_seqs = {}
        self.sbj_based_seqs = {}
        self.split_seqs = {'test': [],
                           'val': [],
                           'train': []
                           }

        # group, mask, and sort sequences based on objects, subjects, and intents
        self.process_sequences()

        self.logger('Total sequences: %d' % len(self.all_seqs))
        self.logger('Selected sequences: %d' % len(self.selected_seqs))
        self.logger('Number of sequences in each data split : train: %d , test: %d , val: %d'
                         %(len(self.split_seqs['train']), len(self.split_seqs['test']), len(self.split_seqs['val'])))
        self.logger('Number of objects in each data split : train: %d , test: %d , val: %d'
                         % (len(self.splits['train']), len(self.splits['test']), len(self.splits['val'])))

        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)

        # process the data
        self.data_preprocessing(cfg)

    def get_feat(self, data):
        obj_trans = data['trans_obj']
        origin_trans = obj_trans[0:1, :]
        
        rhand_rot = data['global_orient_rhand_rotmat'][:, :2, :].reshape(-1, 6)
        rhand_pose = data['fpose_rhand_rotmat'][:, :, :2, :].reshape(-1, 15*6)
        rhand_trans = data['trans_rhand'] - obj_trans
        rhand_feat = np.concatenate([rhand_rot, rhand_pose, rhand_trans], axis=-1)
        
        obj_rot = data['root_orient_obj_rotmat'][:, :2, :].reshape(-1, 6)
        obj_trans = obj_trans - origin_trans
        obj_feat = np.concatenate([obj_rot, obj_trans], axis=-1)

        comb_feat = np.concatenate([obj_feat, rhand_feat], axis=-1)
        ldm_feat = np.concatenate([obj_feat, rhand_trans], axis=-1)
        return comb_feat, ldm_feat

    def data_preprocessing(self, cfg):
        self.subject_mesh = {}
        self.obj_info = {}
        self.sbj_info = {}
        all_feat = []
        kw = cfg.filter
        EXPORT_SEQ = self.cfg.export_seq
    
        for split in self.split_seqs.keys():

            self.logger('Processing data for %s split.' % (split))

            frame_names = []

            for sequence in tqdm(self.split_seqs[split]):
                if not kw in sequence:
                    continue

                meta_file = sequence.replace('pose.npz', 'meta.yml')
                with open(meta_file, 'r') as f:
                    meta = yaml.load(f, Loader=yaml.FullLoader)
                mano_calib_file = os.path.join(self.data_path, "calibration", "mano_{}".format(meta['mano_calib'][0]), "mano.yml")
                
                with open(mano_calib_file, 'r') as f:
                    mano_calib = yaml.load(f, Loader=yaml.FullLoader)
                mano_betas = mano_calib['betas']
                obj_id = meta['ycb_ids'][meta['ycb_grasp_ind']]
                obj_name = _YCB_CLASSES[obj_id]

                obj_info = self.load_obj_verts(obj_name)
                seq_data = get_hand_grasp(sequence, meta['mano_sides'][0], meta['ycb_grasp_ind'], mano_betas, obj_info["verts_sample"])
                T = seq_data['trans_obj'].shape[0]

                obj_word = ' '.join(obj_name.split('_')[1:])
                clip_word = f"pick up the {obj_word}"
                clip_embedding = clip.tokenize([clip_word]).to(device)
                text_features = self.clip_model.encode_text(clip_embedding)
                
                #TODO add comb feat & ldm feat
                comb_feat, ldm_feat = self.get_feat(seq_data)
                
                if EXPORT_SEQ:
                    #############################################################################################################
                    # Export sequences
                    seq_name = os.path.basename(os.path.dirname(sequence))
                    # for i in range(T):
                    export_path = os.path.join(self.out_path, split, seq_name + f'_{meta["mano_sides"][0]}.npy')
                    os.makedirs(os.path.dirname(export_path), exist_ok=True)
                    data = {
                            'feat': comb_feat,
                            'feat_ldm': ldm_feat,
                            'obj_verts': obj_info['verts'],
                            "text_features": text_features.cpu().detach().numpy(),
                            "object": obj_name,
                            'rhand': seq_data,
                    }
                    data['rhand']['verts_rhand'] = data['rhand']['verts_rhand'] - data['rhand']['trans_obj'].reshape(-1, 1, 3)
                    data['rhand']['verts_object'] = data['rhand']['verts_object'] - data['rhand']['trans_obj'].reshape(-1, 1, 3)
                    data['rhand']['trans_rhand'] = data['rhand']['trans_rhand'] - data['rhand']['trans_obj']
                    np.save(export_path, data, allow_pickle=True)
                    all_feat.append((torch.from_numpy(comb_feat), torch.from_numpy(ldm_feat)))
                    ##############################################################################################################
                else:
                    #############################################################################################################
                    # Export Frames
                    seq_name = os.path.basename(os.path.dirname(sequence))
                    seq_data['verts_rhand'] = seq_data['verts_rhand'] - seq_data['trans_obj'].reshape(-1, 1, 3)
                    seq_data['verts_object'] = seq_data['verts_object'] - seq_data['trans_obj'].reshape(-1, 1, 3)
                    seq_data['trans_rhand'] = seq_data['trans_rhand'] - seq_data['trans_obj']
                    for i in range(T):
                        export_path = os.path.join(self.out_path, split, seq_name, f'{meta["mano_sides"][0]}_{i}.npy')
                        os.makedirs(os.path.dirname(export_path), exist_ok=True)
                        data = {    
                            'trans_obj': seq_data['trans_obj'][i],
                            'root_orient_obj_rotmat': seq_data['root_orient_obj_rotmat'][i],
                            'verts_object': seq_data['verts_object'][i],
                            'global_orient_rhand_rotmat': seq_data['global_orient_rhand_rotmat'][i],
                            'fpose_rhand_rotmat': seq_data['fpose_rhand_rotmat'][i],
                            'trans_rhand': seq_data['trans_rhand'][i],
                            'verts_rhand': seq_data['verts_rhand'][i],
                            "has_contact": seq_data['has_contact'][i],
                            "is_right": seq_data['is_right']
                        }
                        np.save(export_path, data, allow_pickle=True)
                    #############################################################################################################

            self.logger('Processing for %s split finished' % split)
            self.logger('Total number of frames for %s split is:%d' % (split, len(frame_names)))

        if EXPORT_SEQ:
            all_tensor = torch.cat([x[0] for x in all_feat], dim=0)
            all_mean = all_tensor.mean(dim=0)
            all_std = all_tensor.std(dim=0)

            ldm_tensor = torch.cat([x[1] for x in all_feat], dim=0)
            ldm_mean = ldm_tensor.mean(dim=0)
            ldm_std = ldm_tensor.std(dim=0)

            np.save(
                os.path.join(self.out_path, 'mean_std.npy'),
                {"mean": all_mean.numpy(), "std": all_std.numpy(),
                "ldm_mean": ldm_mean.numpy(), "ldm_std": ldm_std.numpy()},
                allow_pickle=True
            )

    def process_sequences(self):

        for sequence in self.all_seqs:
            subject_id = sequence.split('/')[-2]
            action_name = os.path.basename(sequence)
            object_name = action_name.split('_')[0]
    
            meta_file = sequence.replace('pose.npz', 'meta.yml')
            with open(meta_file, 'r') as f:
                meta = yaml.load(f, Loader=yaml.FullLoader)
            obj_id = meta['ycb_ids'][meta['ycb_grasp_ind']]
            object_name = _YCB_CLASSES[obj_id]

            # filter data based on the motion intent
            if self.intent == 'all':
                pass
            elif self.intent == 'use' and any(intnt in action_name for intnt in INTENTS[:3]):
                continue
            elif self.intent not in action_name:
                continue

            # group motion sequences based on objects
            if object_name not in self.obj_based_seqs:
                self.obj_based_seqs[object_name] = [sequence]
            else:
                self.obj_based_seqs[object_name].append(sequence)

            # group motion sequences based on subjects
            if subject_id not in self.sbj_based_seqs:
                self.sbj_based_seqs[subject_id] = [sequence]
            else:
                self.sbj_based_seqs[subject_id].append(sequence)

            # split train, val, and test sequences
            self.selected_seqs.append(sequence)
            if object_name in self.splits['test']:
                self.split_seqs['test'].append(sequence)
            elif object_name in self.splits['val']:
                self.split_seqs['val'].append(sequence)
            else:
                self.split_seqs['train'].append(sequence)
                if object_name not in self.splits['train']:
                    self.splits['train'].append(object_name)


    def filter_contact_frames(self, seq_data):
        frame_mask = (seq_data['contact']['object']>0).any(axis=1)
        start = 0
        end = frame_mask.shape[0]-1
        while start < frame_mask.shape[0]:
            if frame_mask[start]:
                break
            start = start + 1
        while end > 0:
            if frame_mask[end]:
                break
            end = end -1
        # TODO MODIFY Here to Control Whether Include Approach Phase
        # start = max(0, start-60)
        print(f"Contact {start} to {end} / {frame_mask.shape[0]} frames")
        frame_mask[start:end] = 1
        return frame_mask, start, end

    def load_obj_verts(self, obj_name, n_verts_sample=1024):
        import trimesh
        from pytorch3d.ops import sample_farthest_points
        mesh_f = os.path.join(self.data_path, 'models', obj_name, 'textured_simple.obj') 
        # mesh_path = os.path.join(self.grab_path, '..',seq_data.object.object_mesh)
        if obj_name not in self.obj_info:
            # obj_name = os.path.basename(mesh_f).split('.stl')[0]
            mesh = trimesh.load(mesh_f, force='mesh')
            pcd = np.array(mesh.vertices)
            normals = np.array(mesh.vertex_normals)
            faces = np.array(mesh.faces)
            sampled_pcd, idx = sample_farthest_points(torch.from_numpy(pcd).view(1, -1, 3), K=n_verts_sample)
            sampled_pcd = sampled_pcd[0].numpy()
            idx = idx[0].numpy()
            full_obj_name = os.path.basename(os.path.dirname(mesh_f))

            _, sampled_faces = decimate_mesh(pcd, faces, idx)

            self.obj_info[obj_name] = {'verts': pcd,
                                       'faces': faces,
                                       "normals": normals,
                                       'verts_sample_id': idx,
                                       'verts_sample': sampled_pcd,
                                       "faces_sample": sampled_faces,
                                       'obj_mesh_file': mesh_f}

        return self.obj_info[obj_name]

    def load_sbj_verts(self, sbj_id, seq_data):

        mesh_path = os.path.join(self.grab_path, '..',seq_data.body.vtemp)
        if sbj_id in self.sbj_info:
            sbj_vtemp = self.sbj_info[sbj_id]
        else:
            sbj_vtemp = np.array(Mesh(filename=mesh_path).vertices)
            self.sbj_info[sbj_id] = sbj_vtemp
        return sbj_vtemp

def decimate_mesh(v, f, target_v_idx):
    target_v = v[target_v_idx]
    distance = np.sum((target_v.reshape(1, -1, 3) - v.reshape(-1, 1, 3))**2, axis=-1)
    merge_to_idx = np.argmin(distance, axis=1)
    new_f = merge_to_idx[f].astype(np.int64)
    MAX_NUM_VERTEX = 1000000
    hash_fn = lambda x: x[:, 0] * MAX_NUM_VERTEX**2 + x[:, 1] * MAX_NUM_VERTEX + x[:, 2]
    dehash_fn = lambda x: (x // MAX_NUM_VERTEX**2, (x % MAX_NUM_VERTEX**2) // MAX_NUM_VERTEX, x % MAX_NUM_VERTEX)
    dedup_idx = set(hash_fn(new_f).tolist())
    dedup_idx = np.array(list(dedup_idx))
    new_f = np.stack(dehash_fn(dedup_idx), axis=-1)
    new_f = new_f[new_f[:, 0] != new_f[:, 1]]
    new_f = new_f[new_f[:, 1] != new_f[:, 2]]
    new_f = new_f[new_f[:, 2] != new_f[:, 0]]
    return v[target_v_idx], new_f


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
    parser.add_argument("--filter", type=str, default="")
    parser.add_argument('--seq', action='store_true')

    args = parser.parse_args()

    data_path = "../../data/dexYCB/raw" #args.grab_path
    out_path = "../../data/dexYCB"
    model_path = "../../data/body_models" #args.model_path
    process_id = 'dexycb_seq' if args.seq else 'dexycb_grab'# choose an appropriate ID for the processed data

    # grab_path = 'PATH_TO_DOWNLOADED_GRAB_DATA/grab'
    # out_path = 'PATH_TO_THE LOCATION_TO_SAVE_DATA'
    # process_id = 'GRAB_V00' # choose an appropriate ID for the processed data
    # model_path = 'PATH_TO_DOWNLOADED_MODELS_FROM_SMPLX_WEBSITE/'


    # split the dataset based on the objects
    # grab_splits = { 'test': ['mug', 'wineglass', 'camera', 'binoculars', 'fryingpan', 'toothpaste'],
    #                 'val': ['apple', 'toothbrush', 'elephant', 'hand'],
    #                 'train': []}
    grab_splits = { 'test': ['002_master_chef_can', '011_banana', '025_mug', '061_foam_brick'],
                    'val': [],
                    'train': []}

    cfg = {

        'intent':'all', # from 'all', 'use' , 'pass', 'lift' , 'offhand'
        'only_contact':True, # if True, returns only frames with contact
        'save_body_verts': False, # if True, will compute and save the body vertices
        'save_lhand_verts': True, # if True, will compute and save the body vertices
        'save_rhand_verts': True, # if True, will compute and save the body vertices
        'save_object_verts': True,

        'save_contact': True, # if True, will add the contact info to the saved data

        # splits
        'splits':grab_splits,

        #IO path
        'data_path': data_path,
        'out_path': os.path.join(out_path, process_id),

        # number of vertices samples for each object
        'n_verts_sample': 1024,

        # body and hand model path
        'model_path':model_path,
        "filter": args.filter,
        "export_seq": args.seq,
    }

    cwd = os.getcwd()
    default_cfg_path = os.path.join(cwd, '../configs/grab_preprocessing_cfg.yaml')
    cfg = Config(default_cfg_path=None, **cfg)
    cfg.write_cfg(write_path=cfg.out_path+'/grab_preprocessing_cfg.yaml')

    log_dir = os.path.join(cfg.out_path, 'grab_processing.log')
    logger = makelogger(log_dir=log_dir, mode='a').info

    logger(instructions)

    GRABDataSet(cfg, logger)

