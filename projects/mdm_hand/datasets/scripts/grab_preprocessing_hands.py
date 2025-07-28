
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
import os
sys.path.append('.')
sys.path.append('..')
sys.path.append(os.path.dirname(__file__))
import numpy as np
import torch
import glob
import smplx
import argparse

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
from tools.quaternion import *
# from tools.skeleton import Skeleton
from tools.paramUtil import *

from scipy.spatial.transform import Rotation as R
import scipy.ndimage.filters as filters


INTENTS = ['lift', 'pass', 'offhand', 'use', 'all']

class GRABDataSet(object):

    def __init__(self, cfg, logger=None, **params):

        self.cfg = cfg
        self.grab_path = cfg.grab_path
        self.out_path = cfg.out_path
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

        if cfg.splits is None:
            grab_splits = { 'test': ['s10'],
                    'val': ['s1'],
                    'train': []}
        else:
            assert isinstance(cfg.splits, dict)
            self.splits = cfg.splits

        self.all_seqs = glob.glob(self.grab_path + '/*/*.npz')
        ## to be filled
        self.selected_seqs = []
        self.obj_based_seqs = {}
        self.sbj_based_seqs = {}
        self.split_seqs = {'test': [],
                           'val': [],
                           'train': []
                           }

        self.process_sequences_grab()
        # group, mask, and sort sequences based on objects, subjects, and intents
        self.process_sequences()

        self.logger('Total sequences: %d' % len(self.all_seqs))
        self.logger('Selected sequences: %d' % len(self.selected_seqs))
        self.logger('Number of sequences in each data split : train: %d , test: %d , val: %d'
                         %(len(self.split_seqs['train']), len(self.split_seqs['test']), len(self.split_seqs['val'])))
        self.logger('Number of objects in each data split : train: %d , test: %d , val: %d'
                         % (len(self.splits['train']), len(self.splits['test']), len(self.splits['val'])))

        # process the data
        self.data_preprocessing(cfg)

    def get_forward_root_rot(self,positions):
        l_hip, r_hip, sdr_r, sdr_l = [2, 1, 17, 16]
        across1 = positions[:, r_hip] - positions[:, l_hip]
        across2 = positions[:, sdr_r] - positions[:, sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across**2).sum(axis=-1))[:, np.newaxis]

        forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        forward = filters.gaussian_filter1d(forward, 20, axis=0, mode='nearest')
        forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]

        '''Get Root Rotation'''
        target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
        root_quat = qbetween_np(forward, target)
        root_quat[0] =  np.array([1.0, 0.0, 0.0, 0.0])

        return root_quat

    def recover_root_rot_pos(self,data):
        rot_vel = data[..., 0]
        r_rot_ang = torch.zeros_like(rot_vel).to(self.device)
        '''Get Y-axis rotation from rotation velocity'''
        r_rot_ang[..., 1:] = rot_vel[..., :-1]
        r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

        r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(self.device)

        r_rot_quat[..., 0] = torch.tensor(np.cos(r_rot_ang.cpu().numpy())) #torch.cos(r_rot_ang.unsqueeze(0))[0]
        r_rot_quat[..., 2] = torch.tensor(np.sin(r_rot_ang.cpu().numpy())) #torch.sin(r_rot_ang.unsqueeze(0))[0]

        r_pos = torch.zeros(data.shape[:-1] + (3,)).to(self.device)
        r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
        '''Add Y-axis rotation to root position'''
        r_pos = qrot(qinv(r_rot_quat), r_pos)

        r_pos = torch.cumsum(r_pos, dim=-2)

        r_pos[..., 1] = data[..., 3]
        return r_rot_quat, r_pos

    def recover_from_ric(self,data, joints_num):
        data = torch.tensor(data, dtype=torch.float32).to(self.device)
        r_rot_quat, r_pos = self.recover_root_rot_pos(data)
        positions = data[..., 4:(joints_num - 1) * 3 + 4]
        positions = positions.view(positions.shape[:-1] + (-1, 3))

        '''Add Y-axis rotation to local joints'''
        positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

        '''Add root XZ to joints'''
        positions[..., 0] += r_pos[..., 0:1]
        positions[..., 2] += r_pos[..., 2:3]

        '''Concate root and joints'''
        positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

        return positions, r_rot_quat

    def get_obj_velocity(self, obj_poses):
        obj_rot = R.from_rotvec(obj_poses[:,3:]).as_quat()
        '''Obj Linear Velocity'''
        obj_lvelocity = (obj_poses[1:, :3] - obj_poses[:-1, :3]).copy()
        obj_rot_lvel = qrot_np(obj_rot[1:], obj_lvelocity)
        '''Obj Angular Velocity'''
        obj_rot_qvel = R.from_quat(qmul_np(obj_rot[1:], qinv_np(obj_rot[:-1]))).as_rotvec()
        return obj_rot_lvel, obj_rot_qvel

    def get_rifke(self, positions, root_rot, obj_verts):
        '''Local pose'''
        position_obj = obj_verts.copy()
        positions -= position_obj[:, 0:1]

        '''All positions are invariant to the object orientation'''
        #positions = qrot_np(np.repeat(root_rot[:, None], positions.shape[1], axis=1), positions)
        positions = np.matmul(root_rot.swapaxes(1,2),positions.swapaxes(1,2)).swapaxes(1,2)

        obj_verts[:,1:] = obj_verts[:,1:]-position_obj[:, 0:1]

        #positions_object = qrot_np(np.repeat(root_rot[:, None], obj_verts.shape[1], axis=1), obj_verts)
        return positions, obj_verts

    def rotation_6d_to_matrix(self,d6):
        import torch.nn.functional as F
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)

    def matrix_to_rotation_6d(self,matrix):
        """
        Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
        by dropping the last row. Note that 6D representation is not unique.
        Args:
            matrix: batch of rotation matrices of size (*, 3, 3)

        Returns:
            6D rotation representation, of size (*, 6)

        [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks.
        IEEE Conference on Computer Vision and Pattern Recognition, 2019.
        Retrieved from http://arxiv.org/abs/1812.07035
        """
        batch_dim = matrix.size()[:-2]
        return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

    def get_cont6d_params(self, poses, positions, obj_poses, glob_rot):
        '''Quaternion to continuous 6D'''
        #n_raw_offsets = torch.from_numpy(t2m_raw_offsets_full)
        #kinematic_chain = t2m_kinematic_chain_full
        #skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        #face_joint_indx = [2, 1, 17, 16]
        #quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)
        #cont_6d_params = quaternion_to_cont6d_np(quat_params)

        pose_shape = poses.shape
        pose_mats = R.from_rotvec(poses.reshape(-1,3)).as_matrix()
        cont_6d_params = self.matrix_to_rotation_6d(torch.tensor(pose_mats)).numpy().reshape(pose_shape[0],-1,6)
        #recon = R.from_matrix(self.rotation_6d_to_matrix(torch.tensor(cont_6d_params)).numpy().reshape(-1,3,3)).as_rotvec().reshape(pose_shape[0],-1,3)
        # (seq_len, 4)
        #poses_quat = quaternion_scipy_to_mdm(R.from_rotvec(obj_poses[:,3:]).as_quat().reshape(obj_poses.shape[0],4))
        obj_rot = R.from_rotvec(obj_poses[:,3:]).as_matrix() #poses_quat.copy() #quat_params[:, 0].copy()
        obj_rot_quat = quaternion_scipy_to_mdm(R.from_rotvec(obj_poses[:,3:]).as_quat())

        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (obj_poses[1:, :3] - obj_poses[:-1, :3]).copy()
        #     print(r_rot.shape, velocity.shape)

        obj_lin_velocity = qrot_np(obj_rot_quat[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        obj_ang_velocity = qmul_np(obj_rot_quat[1:], qinv_np(obj_rot_quat[:-1]))

        #l_wrist_quat = quaternion_scipy_to_mdm(R.from_matrix(glob_rots[0][:,20,:3,:3]).as_quat())

        #r_wrist_quat = quaternion_scipy_to_mdm(R.from_matrix(glob_rots[0][:,21,:3,:3]).as_quat())
        #self.matrix_to_rotation_6d(torch.tensor(R.from_quat(quaternion_mdm_to_scipy(qmul_np(qinv_np(obj_rot[:1]), l_wrist_quat))).as_matrix()))
        #cont_6d_params[:,0] = self.matrix_to_rotation_6d(torch.tensor(glob_rots[:,0,:3,:3]))#self.matrix_to_rotation_6d(torch.tensor(np.matmul(obj_rot.swapaxes(1,2),glob_rots[:,0,:3,:3])))

        cont_6d_params[:,0] = self.matrix_to_rotation_6d(torch.tensor(np.matmul(obj_rot.swapaxes(1,2),pose_mats.reshape(pose_shape[0],-1,3,3)[:,0])))
        cont_6d_params[:,16] = self.matrix_to_rotation_6d(torch.tensor(np.matmul(obj_rot.swapaxes(1,2),pose_mats.reshape(pose_shape[0],-1,3,3)[:,16])))

        #R.from_quat(qmul_np(obj_rot[:1], qinv_np())).as_matrix()
        #r_wrist_rel_rot = qmul_np((qinv_np(obj_rot[:1]),r_wrist_quat ))
        #l_mat = np.matmul(R.from_quat(quaternion_mdm_to_scipy(obj_rot[:1])).as_matrix().swapaxes(1,2),glob_rots[0][:,20,:3,:3])
        # (seq_len, joints_num, 4)


        return cont_6d_params, obj_ang_velocity, obj_lin_velocity, obj_rot_quat, None, None, obj_rot

    def generate_feature_vector(self, positions,rotations, obj_poses, obj_verts_sampled, glob_rots, parents=None, position_mode = 'root_relative', only_hands=False):
        #grab_to_amass_rotmat = np.tile(np.array([[1,0,0], [0, 0, 1], [0,-1,0]]), (positions.shape[0],positions.shape[1],1,1))
        #positions = np.matmul(grab_to_amass_rotmat, np.expand_dims(positions,-1))[...,0]

        #obj_verts_sampled = np.matmul(grab_to_amass_rotmat[0,0], np.expand_dims(obj_verts_sampled,-1))[...,0]
        #rotations[:,:3] = np.matmul(grab_to_amass_rotmat[:,0], np.expand_dims(rotations[:,:3],-1))[...,0]
        #obj_poses[:,:3] = np.matmul(grab_to_amass_rotmat[:,0], np.expand_dims(obj_poses[:,:3],-1))[...,0]

        #obj_poses[:,3:] = np.matmul(grab_to_amass_rotmat[:,0],np.expand_dims(obj_poses[:,3:],axis=-1))[...,0]
        #obj_poses[:,3:] = R.from_matrix(np.matmul(grab_to_amass_rotmat[:,0],R.from_rotvec(obj_poses[:,3:]).as_matrix())).as_rotvec()
        '''Obj at origin'''
        import ipdb; ipdb.set_trace()
        obj_pos_init =  obj_poses[:1,:3]
        positions = positions - obj_pos_init

        obj_poses[:,:3] = obj_poses[:,:3] - obj_pos_init

        positions_b = positions.copy()

        obj_verts = np.tile(obj_verts_sampled, (obj_poses.shape[0], 1,1))
        obj_verts[:,0] = np.zeros_like(obj_verts[:,0])
        obj_verts = np.matmul(obj_verts,R.from_rotvec(obj_poses[:,3:]).as_matrix())
        obj_verts += obj_poses[:,np.newaxis,:3]
        #obj_verts[:,0] = obj_poses[:,:3]
        global_object_positions = obj_verts.copy()
        global_positions = positions.copy()
        root_quat = self.get_forward_root_rot(positions)
        #root_quat = qfix(root_quat)
        #rotations[:,:3] = R.from_quat(root_quat).as_rotvec()
        cont_6d_params, obj_ang_velocity, obj_lin_velocity, object_root_rot, lwrist_rel_or, rwrist_rel_or, obj_rot_mat = self.get_cont6d_params(rotations,positions, obj_poses, glob_rots)

        positions, positions_object = self.get_rifke(positions, obj_rot_mat, obj_verts.copy())
        ''' Body Root Information '''


        '''Joint Position Representation'''
        ### Generate root relative key joint positions (root can either be object or body root) ###
        ### Variant 1: object relative body root position, all other joints root relative
        ### Variant 2: object relative body root position, body joints root relative, hand joints wrist relative
        ### Variant 3: all joints object relative
        relative_positions_left = positions[:-1, :21].reshape(len(positions)-1, -1)#np.concatenate((positions[:-1, 20:21], positions[:-1, 22:42]),axis=1).reshape(len(positions)-1, -1) #leave out wrist
        relative_positions_right = positions[:-1, 21:].reshape(len(positions)-1, -1)#np.concatenate((positions[:-1, 21:22],positions[:-1, 42:]),axis=1).reshape(len(positions)-1, -1) #leave out wrist
        relative_obj_positions = positions_object[:-1].reshape(len(positions)-1, -1)

        '''Joint Rotation Representation'''
        ### Either use HumanML3D representation or parent relative representation
        relative_rotations_left = cont_6d_params[:-1, :16].reshape(len(cont_6d_params)-1, -1) #np.concatenate((lwrist_rel_or[:-1].unsqueeze(1),cont_6d_params[:-1, 1:16]),axis=1).reshape(len(cont_6d_params)-1, -1) #leave out wrist
        relative_rotations_right = cont_6d_params[:-1, 16:].reshape(len(cont_6d_params)-1, -1) #np.concatenate((rwrist_rel_or[:-1].unsqueeze(1),cont_6d_params[:-1, 17:]),axis=1).reshape(len(cont_6d_params)-1, -1) #leave out wrist

        '''Get Joint Velocity Representation'''
        ### Either use HumanML3D representation or parent relative representation
        local_vel = np.matmul(np.repeat(obj_rot_mat.swapaxes(1,2)[:-1, None], global_positions.shape[1], axis=1),
                            np.expand_dims(global_positions[1:] - global_positions[:-1], axis=-1))[...,0]

        local_vel_left = local_vel[:,:21].reshape(len(local_vel), -1)#np.concatenate((local_vel[:,20:21], local_vel[:,22:42]),axis=1).reshape(len(local_vel), -1)
        local_vel_right = local_vel[:,21:].reshape(len(local_vel), -1)#np.concatenate((local_vel[:,21:22], local_vel[:,42:]),axis=1).reshape(len(local_vel), -1)

        '''Obj root-relative position'''
        obj_distance = obj_poses[:,:3] - positions[:, 0, :3]

        relative_obj_positions = global_object_positions
        relative_obj_positions[:,1:] = relative_obj_positions[:,1:] - global_object_positions[:,:1]#obj_distance

        relative_obj_positions[:,0] = positions_object[:,0]

        relative_obj_positions = relative_obj_positions[:-1].reshape(len(positions)-1, -1)
        ''' Obj initial pose relative position '''
        obj_init_distance = obj_poses[:,:3] - obj_poses[0,:3]

        '''Object rotation 6d'''
        ### Let's take it relative to the initial frame
        obj_rot_quat = R.from_rotvec(obj_poses[:, 3:6]).as_quat()
        #obj_rot_aa = R.from_quat(qmul_np(np.repeat(obj_rot_quat[0, None], obj_rot_quat.shape[0], axis=0), qinv_np(obj_rot_quat))).as_rotvec()
        obj_rot_6D = self.matrix_to_rotation_6d(torch.tensor(R.from_quat(obj_rot_quat).as_matrix())).numpy()

        '''Object velocity'''
        ### Let's take it relative to the last frame
        obj_lvel, obj_qvel = self.get_obj_velocity(obj_poses)
        local_obj_vel = qrot_np(np.repeat(object_root_rot[:-1, None], global_object_positions.shape[1], axis=1),
                            global_object_positions[1:] - global_object_positions[:-1])
        local_obj_vel = local_obj_vel.reshape(len(local_obj_vel), -1)

        '''Contact information'''
        ### TO ADD

        feature_vector = np.concatenate([relative_positions_left, relative_positions_right,#global_positions[:-1,22:].                            reshape(-1,90),
                                         relative_rotations_left, relative_rotations_right,
                                         local_vel_left, local_vel_right,
                                         #relative_obj_positions, local_obj_vel], axis=-1)
                                         relative_obj_positions[:,:3], obj_rot_6D[:-1], #relative_obj_positions, local_obj_vel, #obj_rot_6D[:-1]
                                         obj_lvel, obj_qvel], axis=-1)

        ### ONLY MANO ###
        # feature_vector = np.concatenate([relative_positions_left[:,:3], relative_positions_right[:,:3],
        #                                 relative_rotations_left, relative_rotations_right,
        #                                 local_vel_left[:,:3], local_vel_right[:,:3],
        #                                 relative_obj_positions[:,:3], obj_rot_6D[:-1],
        #                                 obj_lvel, obj_qvel], axis=-1)

        #feature_vector = np.concatenate([root_data, relative_positions[:-1], relative_rotations[:-1], local_vel, feet_l, feet_r, relative_obj_positions[:-1], local_obj_vel], axis=-1)
        # feature_vec_subsampled = []
        # for fId in range(0, feature_vector.shape[0], 6):
        #     feature_vec_subsampled.append(feature_vector[fId:fId+1]) # controls the global root orientation
        #new_pos, r_quat_rec = self.recover_from_ric(feature_vector,52)#.cpu().numpy().reshape(root_data.shape[0],-1)

        #print(global_positions[:-1]- new_pos.cpu().numpy().reshape(-1,52,3))


        # feature_vector = np.concatenate([root_data, new_pos, relative_rotations[:-1], local_vel, feet_l, feet_r, obj_distance[:-1],obj_init_distance[:-1], obj_poses[:-1], obj_lvel, obj_qvel], axis=-1)
        return feature_vector, global_positions

    def data_preprocessing(self,cfg):

        # stime = datetime.now().replace(microsecond=0)
        # shutil.copy2(sys.argv[0],
        #              os.path.join(self.out_path,
        #                           os.path.basename(sys.argv[0]).replace('.py','_%s.py' % datetime.strftime(stime,'%Y%m%d_%H%M'))))

        self.subject_mesh = {}
        self.obj_info = {}
        self.sbj_info = {}

        # for split in self.split_seqs.keys():
        #     self.logger('Processing data for %s split.' % (split))

        frame_names = []



        self.seq_num = 0
        print(self.grab_paths)

        for sequence in tqdm(self.grab_paths):
            body_data = {
            'global_orient': [],'body_pose': [],'transl': [],
            'right_hand_pose': [],'left_hand_pose': [],
            'jaw_pose': [],'leye_pose': [],'reye_pose': [],
            'expression': [],'fullpose': [],
            'contact':[], 'verts' :[], 'joints': [], 'glob_rots': []
            }
            object_data ={'verts': [], 'global_orient': [], 'transl': [], 'contact': [], 'joints': [], 'global_orient': []}
            lhand_data = {'verts': [], 'global_orient': [], 'hand_pose': [], 'transl': [], 'fullpose': [], 'joints': []}
            rhand_data = {'verts': [], 'global_orient': [], 'hand_pose': [], 'transl': [], 'fullpose': [], 'joints': []}
            feature_vec = {'feature_vector': []}
            print(self.seq_num)
            try:
                seq_data = parse_npz(sequence)
            except:
                print("Error loading file", sequence)
                self.seq_num += 1
                continue

            obj_name = seq_data.obj_name
            sbj_id   = seq_data.sbj_id
            n_comps  = seq_data.n_comps
            gender   = seq_data.gender

            frame_mask = self.filter_contact_frames(seq_data)

            # total selectd frames
            T = frame_mask.sum()

            if T < 1:
                continue # if no frame is selected continue to the next sequence

            sbj_params = prepare_params(seq_data.body.params, frame_mask)
            rh_params  = prepare_params(seq_data.rhand.params, frame_mask)
            lh_params  = prepare_params(seq_data.lhand.params, frame_mask)
            obj_params = prepare_params(seq_data.object.params, frame_mask)

            append2dict(body_data, sbj_params)
            append2dict(rhand_data, rh_params)
            append2dict(lhand_data, lh_params)
            append2dict(object_data, obj_params)

            sbj_vtemp = self.load_sbj_verts(sbj_id, seq_data)

            if cfg.save_body_verts:

                sbj_m = smplx.create(model_path=cfg.model_path,
                                        model_type='smplx',
                                        gender=gender,
                                        num_pca_comps=n_comps,
                                        v_template=sbj_vtemp,
                                        batch_size=T)

                sbj_parms = params2torch(sbj_params)

                verts_sbj = sbj_m(**sbj_parms)#joints[:,:52])

                #body_data['verts'].append(to_cpu(verts_sbj.vertices))
                body_joints=to_cpu(verts_sbj.joints)
                #body_rots = to_cpu(verts_sbj.glob_rots)
                #body_data['joints'].append(np.concatenate((body_joints[:,:22],body_joints[:,25:55]),axis=1))
                body_data['joints'].append(np.concatenate((body_joints[:,:22],body_joints[:,25:40], body_joints[:,66:71], body_joints[:,40:55], body_joints[:,71:76]),axis=1))
                #body_data['glob_rots'].append(np.concatenate((body_rots[:,:22],body_rots[:,25:]),axis=1))#, body_joints[:,66:71], body_joints[:,40:55], body_joints[:,71:76]),axis=1))
            if cfg.save_lhand_verts:
                lh_mesh = os.path.join(grab_path, '..', seq_data.lhand.vtemp)
                lh_vtemp = np.array(Mesh(filename=lh_mesh).vertices)

                lh_m = smplx.create(model_path=cfg.model_path,
                                    model_type='mano',
                                    is_rhand=False,
                                    v_template=lh_vtemp,
                                    num_pca_comps=n_comps,
                                    flat_hand_mean=True,
                                    batch_size=T)

                lh_parms = params2torch(lh_params)
                #lh_m(hand_pose=torch.zeros((1040,24)), transl=torch.zeros((1040,3)),global_orient=torch.zeros((1040,3)))
                verts_lh = lh_m(**lh_parms)
                #lhand_data['verts'].append(to_cpu(verts_lh.vertices))
                lhand_data['joints'].append(to_cpu(verts_lh.joints))

            if cfg.save_rhand_verts:
                rh_mesh = os.path.join(grab_path, '..', seq_data.rhand.vtemp)
                rh_vtemp = np.array(Mesh(filename=rh_mesh).vertices)

                rh_m = smplx.create(model_path=cfg.model_path,
                                    model_type='mano',
                                    is_rhand=True,
                                    v_template=rh_vtemp,
                                    num_pca_comps=n_comps,
                                    flat_hand_mean=True,
                                    batch_size=T)
                rh_parms = params2torch(rh_params)
                verts_rh = rh_m(**rh_parms)
                #rhand_data['verts'].append(to_cpu(verts_rh.vertices))
                rhand_data['joints'].append(to_cpu(verts_rh.joints))

            ### for objects
            obj_info = self.load_obj_verts(obj_name, seq_data, cfg.n_verts_sample, cfg.bounding_box)

            if cfg.save_object_verts:

                obj_m = ObjectModel(v_template=obj_info['verts_sample'],
                                    batch_size=T)
                obj_parms = params2torch(obj_params)
                verts_obj, rot_mats = obj_m(**obj_parms)
                object_data['verts'].append(to_cpu(verts_obj.vertices))

            if cfg.save_contact:
                body_data['contact'].append(seq_data.contact.body[frame_mask])
                object_data['contact'].append(seq_data.contact.object[frame_mask][:,obj_info['verts_sample_id']])

            frame_names.extend(['%s_%s' % (sequence.split('.')[0], fId) for fId in np.arange(T)])

            #full_body_pose = np.concatenate((np.array(body_data['fullpose'])[...,:66],np.array(body_data['fullpose'])[...,75:]),axis=-1)
            full_body_pose = np.concatenate((lh_params['global_orient'], lh_params['fullpose'],
                                            rh_params['global_orient'], rh_params['fullpose']),axis=-1)

            obj_pose = np.concatenate((np.array(object_data['transl']),np.array(object_data['global_orient'])),axis=-1)
            body_pose = []
            body_positions = []
            obj_poses = []
            glob_rots = []

            for fId in range(0, obj_pose.shape[1], 6):
                #body_positions.append(np.array(body_data['joints'])[0][fId:fId+1]) # controls the global root orientation
                body_positions.append(np.concatenate((np.array(lhand_data['joints'])[0][fId:fId+1],np.array(rhand_data['joints'])[0][fId:fId+1]),axis=1))
                body_pose.append(full_body_pose[fId:fId+1])
                obj_poses.append(obj_pose[0][fId:fId+1])
                #glob_rots.append(np.array(body_data['glob_rots'])[0][fId:fId+1])
            assert self.obj_info[obj_name]['verts_sample'].shape[0] >= 6
            import ipdb; ipdb.set_trace()
            feature_vector, glob_pos = self.generate_feature_vector(np.array(body_positions)[:,0],
                                        np.array(body_pose)[:,0],
                                        np.array(obj_poses)[:,0],self.obj_info[obj_name]['verts_sample'][:12],
                                        None) #np.array(glob_rots)[:,0])

            np.save(os.path.join(self.out_path,str(self.seq_num).zfill(6)+'.npy'), feature_vector)
            self.seq_num += 1

            #np.save(os.path.join(self.out_path, 'obj_info.npy'), self.obj_info)
            #object_data['joints'].append(glob_pos)
            # data = np2torch(object_data)
            # data_name = 'object_data'
            # outfname = makepath(os.path.join(self.out_path, '%s.pt' % data_name), isfile=True)
            # torch.save(data, outfname)

        #self.logger('Processing for %s split finished' % split)
        #self.logger('Total number of frames for %s split is:%d' % (split, len(frame_names)))

        # feature_vec['feature_vector'].append(feature_vector)
        # out_data = [body_data, rhand_data, lhand_data, object_data, feature_vec]
        # out_data_name = ['body_data', 'rhand_data', 'lhand_data', 'object_data', 'feature_vector']


        # for idx, data in enumerate(out_data):
        #     data = np2torch(data)
        #     data_name = out_data_name[idx]
        #     outfname = makepath(os.path.join(self.out_path, split, '%s.pt' % data_name), isfile=True)
        #     torch.save(data, outfname)

        # np.savez(os.path.join(self.out_path, split, 'frame_names.npz'), frame_names=frame_names)

        #np.save(os.path.join(self.out_path, 'obj_info_anchored.npy'), self.obj_info)
        # np.save(os.path.join(self.out_path, 'sbj_info.npy'), self.sbj_info)

    def process_sequences_grab(self):
        paths = []
        folders = []
        dataset_names = []
        for root, dirs, files in os.walk('./data/GRAB/grab_sch'):
            folders.append(root)
            for name in files:
                dataset_name = root.split('/')[2]
                if dataset_name not in dataset_names:
                    dataset_names.append(dataset_name)

                if 'male' not in name and 'female' not in name:
                    name = name.replace('_stageii','')
                    name = name.replace('pick_all','lift')
                    file_path = os.path.join(self.grab_path, root.split('/')[-1], name)
                    #if os.path.exists(file_path):
                    paths.append(file_path)


        self.grab_paths = paths


    def process_sequences(self):
        for sequence in self.all_seqs[:1]:
            subject_id = sequence.split('/')[-2]
            action_name = os.path.basename(sequence)
            object_name = action_name.split('_')[0]

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
            if subject_id in self.splits['test']:
                self.split_seqs['test'].append(sequence)
            elif subject_id in self.splits['val']:
                self.split_seqs['val'].append(sequence)
            else:
                self.split_seqs['train'].append(sequence)
                if subject_id not in self.splits['train']:
                    self.splits['train'].append(object_name)

    def filter_contact_frames(self, seq_data):
        if self.cfg.only_contact:
            frame_mask = (seq_data['contact']['object']>0).any(axis=1)
        else:
            frame_mask = (seq_data['contact']['object']>-1).any(axis=1)
        return frame_mask

    def load_obj_verts(self, obj_name, seq_data, n_verts_sample=512, bounding_box=False):

        mesh_path = os.path.join(self.grab_path, '..',seq_data.object.object_mesh)

        if obj_name not in self.obj_info:
            np.random.seed(100)
            print('bounding ',bounding_box)
            obj_mesh = Mesh(filename=mesh_path, bounding_box=bounding_box)
            verts_obj = np.array(obj_mesh.vertices)
            faces_obj = np.array(obj_mesh.faces)

            verts_sample_id = np.arange(obj_mesh.sampled_points.shape[0]) if bounding_box else obj_mesh.sampled_points[1]
            # if verts_obj.shape[0] > n_verts_sample:
            #     verts_sample_id = np.random.choice(verts_obj.shape[0], n_verts_sample, replace=False)
            # else:
            #     verts_sample_id = np.arange(verts_obj.shape[0])
            verts_sampled = obj_mesh.sampled_points if bounding_box else verts_obj[verts_sample_id]

            self.obj_info[obj_name] = {'verts': verts_obj,
                                       'faces': faces_obj,
                                       'verts_sample_id': verts_sample_id,
                                       'verts_sample': verts_sampled,
                                       'obj_mesh_file': mesh_path}

        return self.obj_info[obj_name]

    def load_sbj_verts(self, sbj_id, seq_data):

        mesh_path = os.path.join(self.grab_path, '..',seq_data.body.vtemp)
        if sbj_id in self.sbj_info:
            sbj_vtemp = self.sbj_info[sbj_id]
        else:
            sbj_vtemp = np.array(Mesh(filename=mesh_path).vertices)
            self.sbj_info[sbj_id] = sbj_vtemp
        return sbj_vtemp


if __name__ == '__main__':

    from IPython import get_ipython
    from ipdb.__main__ import _get_debugger_cls
    import ipdb
    debugger = _get_debugger_cls()
    shell = get_ipython()
    shell.debugger_history_file = '.pdbhistory'

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

    parser.add_argument('--grab-path', required=True, type=str,
                        help='The path to the downloaded grab data')
    parser.add_argument('--out-path', default=None, type=str,
                        help='The path to the folder to save the processed data')
    parser.add_argument('--model-path', required=True, type=str,
                        help='The path to the folder containing smplx models')

    args = parser.parse_args()

    grab_path = args.grab_path
    out_path = args.out_path
    model_path = args.model_path
    process_id = 'GRAB_V00' # choose an appropriate ID for the processed data


    # grab_path = 'PATH_TO_DOWNLOADED_GRAB_DATA/grab'
    # out_path = 'PATH_TO_THE LOCATION_TO_SAVE_DATA'
    # process_id = 'GRAB_V00' # choose an appropriate ID for the processed data
    # model_path = 'PATH_TO_DOWNLOADED_MODELS_FROM_SMPLX_WEBSITE/'


    # split the dataset based on the objects
    grab_splits = { 'test': ['s10'],
                    'val': ['s1'],
                    'train': []}


    cfg = {
        'intent':'all', # from 'all', 'use' , 'pass', 'lift' , 'offhand'
        'only_contact':False, # if True, returns only frames with contact
        'save_body_verts': True, # if True, will compute and save the body vertices
        'save_lhand_verts': True, # if True, will compute and save the body vertices
        'save_rhand_verts': True, # if True, will compute and save the body vertices
        'save_object_verts': False,

        'save_contact': False, # if True, will add the contact info to the saved data

        'bounding_box': True, # if True, will compute and save the bounding box instead of random surface points

        # splits
        'splits':grab_splits,

        #IO path
        'grab_path': grab_path,
        'out_path': os.path.join(out_path, process_id),

        # number of vertices samples for each object
        'n_verts_sample': 1024,

        # body and hand model path
        'model_path':model_path,
    }

    cwd = os.getcwd()
    default_cfg_path = os.path.join(cwd, '../configs/grab_preprocessing_cfg.yaml')
    cfg = Config(default_cfg_path=default_cfg_path, **cfg)
    cfg.write_cfg(write_path=cfg.out_path+'/grab_preprocessing_cfg.yaml')

    log_dir = os.path.join(cfg.out_path, 'grab_processing.log')
    logger = makelogger(log_dir=log_dir, mode='a').info

    logger(instructions)

    GRABDataSet(cfg, logger)
