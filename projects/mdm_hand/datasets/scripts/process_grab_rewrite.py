# modified from src/arctic/split.py in the ARCTIC CODE

import json
import os
import sys
import os.path as op

sys.path.append('.')
sys.path.append('..')
sys.path.append(os.path.dirname(__file__))
import argparse
import copy
import torch
import glob
import numpy as np
import smplx
import shutil
from loguru import logger
from tqdm import tqdm
from utils.rotation_conversions import axis_angle_to_rotation_6d, rotation_6d_to_axis_angle, axis_angle_to_quaternion
from utils.visualize.object_tensors import GrabObjectTensors
import open3d as o3d
import h5py

from tools.quaternion import *
from tools.paramUtil import *
from tools.pcd_helper import voxelize_batch
from tools.meshviewer import Mesh
from scipy.spatial.transform import Rotation as R


MODEL_DIR = './data/body_models'
lhand_cnt = 0
rhand_cnt = 0

def parse_npz(npz, allow_pickle=True):
    npz = np.load(npz, allow_pickle=allow_pickle)
    npz = {k: npz[k].item() for k in npz.files}
    return DotDict(npz)

def params2torch(params, dtype = torch.float32):
    return {k: torch.from_numpy(v).type(dtype) for k, v in params.items()}

def prepare_params(params, frame_mask, dtype = np.float32):
    return {k: v[frame_mask].astype(dtype) for k, v in params.items()}

def DotDict(in_dict):
    out_dict = copy.copy(in_dict)
    for k,v in out_dict.items():
       if isinstance(v,dict):
           out_dict[k] = DotDict(v)
    return dotdict(out_dict)

def get_frame_mask(seq_data, only_contact=False):
    if not only_contact:
        return np.ones(seq_data['contact']['object'].shape[0], dtype=bool)
    else:
        frame_mask = (seq_data['contact']['object']>0).any(axis=1)
        i = 0
        j = frame_mask.shape[0]-1
        while i<frame_mask.shape[0]:
            if frame_mask[i]:
                break
            i = i+1
        while j>0:
            if frame_mask[j]:
                break
            j = j -1
        print(f"Contact {i} to {j} / {frame_mask.shape[0]} frames")
        frame_mask[i:j] = 1
        return frame_mask

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_cont6d_params(poses, positions, obj_poses, obj_init_rot):
    '''Quaternion to continuous 6D'''

    pose_shape = poses.shape
    pose_mats = R.from_rotvec(poses.reshape(-1,3)).as_matrix()
    cont_6d_params = matrix_to_rotation_6d(torch.tensor(pose_mats)).numpy().reshape(pose_shape[0],-1,6)

    obj_rot = R.from_rotvec(obj_poses[:,3:]).as_matrix() #poses_quat.copy() #quat_params[:, 0].copy()

    cont_6d_params[:,0] = matrix_to_rotation_6d(torch.tensor(np.matmul(obj_rot.swapaxes(1,2),pose_mats.reshape(pose_shape[0],-1,3,3)[:,0])))
    cont_6d_params[:,16] = matrix_to_rotation_6d(torch.tensor(np.matmul(obj_rot.swapaxes(1,2),pose_mats.reshape(pose_shape[0],-1,3,3)[:,16])))

    return cont_6d_params, obj_rot


def get_rifke(positions, root_rot, obj_verts):
    '''Local pose'''
    position_obj = obj_verts.copy()
    positions -= position_obj[:, 0:1]

    # TODO: uncommment below to disable trans caonicalization!!!!
    # positions = qrot_np(np.repeat(root_rot[:, None], positions.shape[1], axis=1), positions)
    positions = np.matmul(root_rot.swapaxes(1,2),positions.swapaxes(1,2)).swapaxes(1,2)

    obj_verts[:,1:] = obj_verts[:,1:]-position_obj[:, 0:1]

    #positions_object = qrot_np(np.repeat(root_rot[:, None], obj_verts.shape[1], axis=1), obj_verts)
    return positions, obj_verts

def generate_feature_vector(positions, rotations, obj_poses, obj_init_rot):
    '''
        positions: [nf, 42, 3]
        rotations: [nf, 32, 3]
        obj_poses: [nf, 6]
        
    '''
    positions = copy.deepcopy(positions)
    rotations = copy.deepcopy(rotations)
    obj_poses = copy.deepcopy(obj_poses)
    if obj_init_rot is not None:
        obj_init_rot = copy.deepcopy(obj_init_rot)

        obj_poses[:, 3:] = R.from_matrix(np.matmul(
            np.repeat(obj_init_rot[None, :], obj_poses.shape[0], axis=0).swapaxes(1, 2),
            R.from_rotvec(obj_poses[:, 3:]).as_matrix()
        )).as_rotvec()

    obj_pos_init =  obj_poses[:1,:3]
    positions = positions - obj_pos_init[:,np.newaxis,:]
    global_positions = positions.copy()

    obj_poses[:,:3] = obj_poses[:,:3] - obj_pos_init
    obj_verts = obj_poses[:,np.newaxis,:3].copy()

    cont_6d_params, obj_rot_mat = get_cont6d_params(rotations, positions, obj_poses, obj_init_rot)
    positions, positions_object = get_rifke(positions, obj_rot_mat, obj_verts.copy())

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
    relative_rotations_left = cont_6d_params[:-1, :16].reshape(len(cont_6d_params)-1, -1) 
    relative_rotations_right = cont_6d_params[:-1, 16:].reshape(len(cont_6d_params)-1, -1) 

    '''Get Joint Velocity Representation'''
    ### Either use HumanML3D representation or parent relative representation
    local_vel = np.matmul(np.repeat(obj_rot_mat.swapaxes(1,2)[:-1, None], global_positions.shape[1], axis=1),
                        np.expand_dims(global_positions[1:] - global_positions[:-1], axis=-1))[...,0]

    local_vel_left = local_vel[:,:21].reshape(len(local_vel), -1)#np.concatenate((local_vel[:,20:21], local_vel[:,22:42]),axis=1).reshape(len(local_vel), -1)
    local_vel_right = local_vel[:,21:].reshape(len(local_vel), -1)#np.concatenate((local_vel[:,21:22], local_vel[:,42:]),axis=1).reshape(len(local_vel), -1)

    '''Object rotation 6d'''
    ### Let's take it relative to the initial frame
    obj_rot_quat = R.from_rotvec(obj_poses[:, 3:6]).as_quat()
    obj_rot_6D = matrix_to_rotation_6d(torch.tensor(R.from_quat(obj_rot_quat).as_matrix())).numpy()

    '''Contact information'''
    feature_vector = np.concatenate([relative_positions_left, relative_positions_right,
                                     relative_rotations_left, relative_rotations_right,
                                     local_vel_left, local_vel_right,
                                     ], axis=-1)
    cond = np.concatenate([relative_obj_positions[:,:3], obj_rot_6D[:-1]], axis=-1)
    return feature_vector, cond


def get_selected_seqs(grab_path, filter=None):
    paths = []
    for file in glob.glob(op.join(grab_path, '**/*.npz')):
        paths.append(file.split(grab_path)[-1].split(".npz")[0])
    if filter is not None:
        paths = [x for x in paths if filter in x]
    return paths


import trimesh
from pytorch3d.ops import sample_farthest_points
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

def preprocess_mesh():
    mesh_d = {}
    for ply in tqdm(glob.glob('data/grab/tools/object_meshes/contact_meshes/*.ply')):
        obj_name = os.path.basename(ply).split('.ply')[0]
        mesh = trimesh.load(ply)
        
        pcd = np.array(mesh.vertices)
        normals = np.array(mesh.vertex_normals)
        faces = np.array(mesh.faces)
        sampled_pcd, idx = sample_farthest_points(torch.from_numpy(pcd).view(1, -1, 3), K=2048)
        sampled_pcd = sampled_pcd[0].numpy()
        idx = idx[0].numpy()
        _, sampled_faces = decimate_mesh(pcd, faces, idx)
        mesh_d[obj_name] = pcd, normals, sampled_pcd, normals[idx], idx, faces, sampled_faces
        # save ply mesh
        mesh = trimesh.Trimesh(vertices=sampled_pcd, faces=sampled_faces)
        mesh.export(f'.exps/debug/{obj_name}.ply')
    return mesh_d

import matplotlib.pyplot as plt
def save_point_cloud(coord, values, file_path="pc.ply"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    coord = np.asarray(coord)
    values = np.asarray(values)
    # Normalize the values between 0 and 1
    values = 1 - (values - np.min(values)) / (np.max(values) - np.min(values))
    # Map the normalized values to a color using the "turbo" colormap
    colors = plt.get_cmap("turbo")(values)[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file_path, pcd)
    if logger is not None:
        logger.info(f"Save Point Cloud to: {file_path}")

def get_contact_maps(hand_v, hand_f, obj, keypoints=None, obj_poses=None):
    num_frames = hand_v.shape[0]
    obj, obj_2048 = obj
    obj_v = np.asarray(obj.vertices)
    obj_f = np.asanyarray(obj.triangles)
    obj_v_2048 = np.asarray(obj_2048.points)

    if obj_poses is not None:
        obj_rot = R.from_rotvec(obj_poses[:,3:]).as_matrix() #poses_quat.copy() #quat_params[:, 0].copy()
        obj_trans = obj_poses[:,:3]
        obj_v = np.matmul(
            obj_rot.reshape(-1, 3, 3),
            obj_v.reshape(1, -1, 3).swapaxes(1, 2)
        ).swapaxes(1, 2) + obj_trans.reshape(-1, 1, 3)
        obj_v_2048 = np.matmul(
            obj_rot.reshape(-1, 3, 3),
            obj_v_2048.reshape(1, -1, 3).swapaxes(1, 2)
        ).swapaxes(1, 2) + obj_trans.reshape(-1, 1, 3)

    kp_sd = []
    obj_sd = []
    for i in tqdm(range(num_frames)):
        obj_mesh = trimesh.Trimesh(obj_v[i], obj_f)
        hand_mesh = trimesh.Trimesh(hand_v[i], hand_f)
        obj_sd.append(trimesh.proximity.signed_distance(hand_mesh, obj_v_2048[i]))
        if keypoints is not None:
            kp_sd.append(trimesh.proximity.signed_distance(obj_mesh, keypoints[i]))
        # import ipdb; ipdb.set_trace()
        # save_point_cloud(
        #     np.concatenate([hand_v[i], obj_v_2048[i]], axis=0),
        #     np.concatenate([np.zeros(hand_v[i].shape[0]), obj_sd[-1]], axis=0),
        #     f"./.exps/debug/{i}.ply"
        # )# 
            save_point_cloud(
                np.concatenate([keypoints[i], obj_v_2048[i]], axis=0),
                np.concatenate([kp_sd[-1], obj_sd[-1]], axis=0),
                f"./.exps/debug/{i}.ply"
            )

    obj_sd = np.stack(obj_sd, axis=0)
    if keypoints is not None:
        kp_sd = np.stack(kp_sd, axis=0)
        return kp_sd, obj_sd
    else:
        return obj_sd

def save_hdf5(d, path):
    f = h5py.File(path, 'w')
    for k, v in d.items():
        dset = f.create_dataset(k, data=v)
    f.close()


lh_m = smplx.create(model_path=MODEL_DIR,
                model_type='mano',
                is_rhand=False,
                num_pca_comps=24,
                flat_hand_mean=True)

rh_m = smplx.create(model_path=MODEL_DIR,
                    model_type='mano',
                    is_rhand=True,
                    num_pca_comps=24,
                    flat_hand_mean=True)

lh_m_flat = smplx.create(model_path=MODEL_DIR,
                    model_type='mano',
                    is_rhand=False,
                    use_pca=False,
                    flat_hand_mean=True)

rh_m_flat = smplx.create(model_path=MODEL_DIR,
                    model_type='mano',
                    is_rhand=True,
                    use_pca=False,
                    flat_hand_mean=True)

def read_from_raw(seq, data_path, export_path, meta_file_writer, export_seq=False):
    seq_p = op.join(data_path, f"{seq}.npz")
    export_path = op.join(export_path, f"{seq}.npy")
    # if os.path.exists(export_path):
    #     return
    data = parse_npz(seq_p)
    obj_name = data.obj_name
    sbj_id   = data.sbj_id
    n_comps  = data.n_comps
    gender   = data.gender

    # filter out non-contact frames when exporting clips
    frame_mask = get_frame_mask(data, only_contact=export_seq)
    T = frame_mask.sum()

    rh_params  = prepare_params(data.rhand.params, frame_mask)
    lh_params  = prepare_params(data.lhand.params, frame_mask)
    obj_params = prepare_params(data.object.params, frame_mask)

    lh_mesh = op.join(data_path, '..', data.lhand.vtemp)
    lh_vtemp = np.array(Mesh(filename=lh_mesh).vertices)
    beta_l = np.load(lh_mesh.replace('.ply', '_betas.npy'), allow_pickle=True)
    lh_params['betas'] = beta_l.reshape(1, 10).repeat(T, axis=0)
    lh_params = params2torch(lh_params)
    verts_lh = lh_m(return_full_pose=True, **lh_params)
    full_pose_l = torch.cat([verts_lh.full_pose.cpu().detach(), lh_params['transl']], dim=-1)#.numpy()
    
    rh_mesh = os.path.join(data_path, '..', data.rhand.vtemp)
    rh_vtemp = np.array(Mesh(filename=rh_mesh).vertices)

    beta_r = np.load(rh_mesh.replace('.ply', '_betas.npy'), allow_pickle=True)
    rh_params['betas'] = beta_r.reshape(1, 10).repeat(T, axis=0)
    rh_params = params2torch(rh_params)
    verts_rh = rh_m(return_full_pose=True, **rh_params)
    full_pose_r = torch.cat([verts_rh.full_pose.cpu().detach(), rh_params['transl']], dim=-1)#.numpy()

    out = rh_m_flat(betas=rh_params['betas'],
                    global_orient=full_pose_r[:, :3],
                    hand_pose=full_pose_r[:, 3:48],
                    transl=full_pose_r[:, 48:])
    assert (out.vertices - verts_rh.vertices < 1e-6).all()
    assert (out.joints - verts_rh.joints < 1e-6).all()
    
    out = lh_m_flat(betas=lh_params['betas'],
                    global_orient=full_pose_l[:, :3],
                    hand_pose=full_pose_l[:, 3:48],
                    transl=full_pose_l[:, 48:])
    assert (out.vertices - verts_lh.vertices < 1e-6).all()
    assert (out.joints - verts_lh.joints < 1e-6).all()
    
    full_pose_l = full_pose_l.numpy()
    full_pose_r = full_pose_r.numpy()
    
    # NOTE global orient seems to be reversely defined as in arctic
    # obj_pose = np.concatenate([obj_params["transl"], obj_params["global_orient"]], axis=-1)

    # lhand_contact = data.contact['body'][:, lhand_smplx_ids]
    # rhand_contact = data.contact['body'][:, rhand_smplx_ids]
    # object_contact = data.contact['object']
    # kp_sd, obj_sd_r = get_contact_maps(verts_rh.vertices.detach().numpy(), rh_m.faces, mesh_d[obj_name], joints, obj_pose)
    # obj_sd_l = get_contact_maps(verts_lh.vertices.detach().numpy(), lh_m.faces, mesh_d[obj_name], None, obj_pose)

    # get canonicalized hand keypoints
    obj_rot = R.from_rotvec(obj_params["global_orient"]).as_matrix() #poses_quat.copy() #quat_params[:, 0].copy()
    obj_trans = obj_params["transl"]
    # hand_ca_pcd_l = np.matmul(
    #     obj_rot.reshape(-1, 3, 3).swapaxes(1, 2),
    #     verts_lh.vertices.reshape(1, -1, 3).swapaxes(1, 2)
    # ).swapaxes(1, 2) + obj_trans.reshape(-1, 1, 3)

    # hand_ca_pcd_r = np.matmul(
    #     obj_rot.reshape(-1, 3, 3).swapaxes(1, 2),
    #     verts_rh.vertices.reshape(1, -1, 3).swapaxes(1, 2)
    # ).swapaxes(1, 2) + obj_trans.reshape(-1, 1, 3)

    obj_pcd, obj_normal, obj_v_2048, obj_normal_2048, obj_2048_idx, faces, sampled_faces = mesh_d[obj_name]
    
    obj_v_2048 = np.matmul(
        obj_rot.reshape(-1, 3, 3),
        obj_v_2048.reshape(1, -1, 3).swapaxes(1, 2)
    ).swapaxes(1, 2) + obj_trans.reshape(-1, 1, 3)
    
    def has_contact(joint, obj_v):
        dist_matrix = joint.reshape(-1, 1, 21, 3) - obj_v_2048.reshape(-1, 2048, 1, 3)
        dist_matrix = np.linalg.norm(dist_matrix, axis=-1)
        has_contact = (dist_matrix < 0.02).sum(axis=(1,2)) > 0
        return has_contact
    lhand_contact = has_contact(verts_lh.joints.detach().numpy(), obj_v_2048)
    rhand_contact = has_contact(verts_rh.joints.detach().numpy(), obj_v_2048)
    joints_l = verts_lh.joints.cpu().detach().numpy()
    joints_r = verts_rh.joints.cpu().detach().numpy()
    hand_pcd_l = verts_lh.vertices.detach().numpy()
    hand_pcd_r = verts_rh.vertices.detach().numpy()

    # obj_normal_motion = np.matmul(
    #     obj_rot.reshape(-1, 3, 3),
    #     obj_normal.reshape(1, -1, 3).swapaxes(1, 2)
    # )
    # for i in range(100, 400, 50):
    #     save_point_cloud(
    #         np.concatenate([hand_pcd_l[i], obj_v_2048[i]], axis=0),
    #         np.concatenate([np.zeros(hand_pcd_l[i].shape[0]), 0.5*np.ones(obj_v_2048[i].shape[0])], axis=0),
    #         f"./.exps/debug1/{i}.ply"
    #     )# 

    # data_dict = {
    #     # "keypoint": joints,
    #     # "pose": full_pose,
    #     "shape_l": beta_l,
    #     "shape_r": beta_r,
    #     "vtemp_l": lh_vtemp,
    #     "vtemp_r": rh_vtemp,
    #     "seq_name": seq,
    #     "gender": gender,
    #     "sbj_id": sbj_id,
    #     "hand_pcd_l": hand_pcd_l, 
    #     "hand_pcd_r": hand_pcd_r,
    #     # "hand_ca_pcd_l": hand_ca_pcd_l,
    #     # "hand_ca_pcd_r": hand_ca_pcd_r,
    #     "lhand_contact": lhand_contact,
    #     "rhand_contact": rhand_contact,
    #     # "object_contact": object_contact,
    #     "obj_type": obj_name,
    #     "obj_rot": obj_rot,
    #     "obj_trans": obj_trans,
    #     "obj_pcd": np.array(obj_pcd),
    #     "obj_normal": np.array(obj_normal),
    #     "obj_pcd_2048": np.array(obj_v_2048),
    #     "obj_normal_2048": np.array(obj_normal_2048),
    #     # "object_contact_2048": object_contact[:, obj_2048_idx],
    # }
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    # save_hdf5(data_dict, export_path.replace('.npy', '.h5'))
    export_dir = os.path.dirname(export_path)
    export_base = os.path.basename(export_path).replace('.npy', '')
    global lhand_cnt, rhand_cnt
    if export_seq:
        # The original data is 120 fps, downsample to 15 fps clips
        os.makedirs(export_dir, exist_ok=True)
        for i in range(8):
            export_path = os.path.join(export_dir, export_base + f"_{i}.h5")
            data = {
                'hand_pcd_r': hand_pcd_r[i::8],
                'hand_pcd_l': hand_pcd_l[i::8],
                'lhand_contact': lhand_contact[i::8],
                'rhand_contact': rhand_contact[i::8],
                'obj_trans': obj_trans[i::8],
                'obj_rot': obj_rot[i::8],
                'obj_name': obj_name
            }
            save_hdf5(
                data,
                export_path
            )
            meta_file_writer.write(f"{export_path}\n")
    else:
        os.makedirs(os.path.join(export_dir, export_base), exist_ok=True)
        for i in range (T):
            export_path = os.path.join(export_dir, export_base, f"l_{i}.h5")
            save_hdf5({
                    "hand_pcd": hand_pcd_l[i],
                    "hand_joints": joints_l[i],
                    "obj_rot": obj_rot[i],
                    "obj_trans": obj_trans[i],
                    "hand_pose": full_pose_l[i],
                    "has_contact": lhand_contact[i],
                    "is_right": 0,
                    'obj_name': obj_name,
                    "betas": beta_l
                },
                os.path.join(export_dir, export_base, f"l_{i}.h5")
            )
            lhand_cnt = lhand_cnt + lhand_contact[i]
            meta_file_writer.write(f"{export_path} {lhand_contact[i]}\n")
            export_path = os.path.join(export_dir, export_base, f"r_{i}.h5")
            save_hdf5({
                    "hand_pcd": hand_pcd_r[i],
                    "hand_joints": joints_r[i],
                    "obj_rot": obj_rot[i],
                    "obj_trans": obj_trans[i],
                    "hand_pose": full_pose_r[i],
                    "has_contact": rhand_contact[i],
                    "is_right": 1,
                    'obj_name': obj_name,
                    "betas": beta_r
                },
                os.path.join(export_dir, export_base, f"r_{i}.h5")
            )
            meta_file_writer.write(f"{export_path} {rhand_contact[i]}\n")
            rhand_cnt = rhand_cnt + rhand_contact[i]
    return #data_dict, seq

def construct_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="./data/grab/grab_raw/"
    )
    parser.add_argument(
        "--filter", type=str, default=""
    )
    parser.add_argument(
        "--seq", action='store_true'
    )
    parser.add_argument(
        "--target_path", type=str, default="./data/grab/grab_muchen/"
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    from IPython import get_ipython
    from ipdb.__main__ import _get_debugger_cls
    import ipdb
    import concurrent.futures
    from tqdm import tqdm
    debugger = _get_debugger_cls()
    shell = get_ipython()
    shell.debugger_history_file = '.pdbhistory'
    args = construct_args()

    global lhand_smplx_ids, rhand_smplx_ids
    global mesh_d
    lhand_smplx_ids = np.load('data/grab/tools/smplx_correspondence/lhand_smplx_ids.npy')
    rhand_smplx_ids = np.load('data/grab/tools/smplx_correspondence/rhand_smplx_ids.npy')
    mesh_d = preprocess_mesh()

    os.makedirs(args.target_path, exist_ok=True)
    np.save(os.path.join(args.target_path, "seqs.npy"), mesh_d)
    import ipdb; ipdb.set_trace()
    all_seqs = get_selected_seqs(args.data_path, args.filter)
    print(f"processing {all_seqs}")

    data = [(x, args.data_path) for x in all_seqs]
    f = open(os.path.join(args.target_path, "seqs.txt"), "w")
    for x in tqdm(all_seqs):
        read_from_raw(x, args.data_path, args.target_path, f, export_seq=args.seq)
    f.close()
    print("Total left hand contact: ", lhand_cnt)
    print("Total right hand contact: ", rhand_cnt)