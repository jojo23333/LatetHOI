import os
import shutil
import glob
import yaml
import json
import numpy as np
import trimesh
import torch
from tqdm import tqdm
from pytorch3d.ops import sample_farthest_points
from scipy.spatial.transform import Rotation as Rot
import smplx
import h5py
# from manopth.manolayer import ManoLayer

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
MODEL_DIR = './data/body_models'
right_cnt = 0
left_cnt = 0

# mano_layers = {
#     "right": ManoLayer(flat_hand_mean=False,
#                 ncomps=45,
#                 side="right",
#                 mano_root='./data/body_models/mano',
#                 use_pca=True
#             ),
#     "left": ManoLayer(flat_hand_mean=False,
#                 ncomps=45,
#                 side="left",
#                 mano_root='./data/body_models/mano',
#                 use_pca=True
#             ),
# }
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

import matplotlib.pyplot as plt
import open3d as o3d
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
    print(f"Save Point Cloud to: {file_path}")

def save_hdf5(d, path):
    f = h5py.File(path, 'w')
    for k, v in d.items():
        dset = f.create_dataset(k, data=v)
    f.close()

def get_hand_grasp(pose_file, mano_side, obj_id, betas, obj_v_2048):
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
    #########################################################################################################################
    # hand_v, hand_j = mano_layers[mano_side](th_pose_coeffs=torch.from_numpy(hand_pose[:, :48]),
    #                              th_betas=betas,
    #                              th_trans=torch.from_numpy(hand_pose[:, 48:]))
    
    # obj_v_2048 = np.matmul(
    #     obj_rot.reshape(-1, 3, 3),
    #     obj_v_2048.reshape(1, -1, 3).swapaxes(1, 2)
    # ).swapaxes(1, 2) + obj_trans.reshape(-1, 1, 3)
    # hand_v = hand_v.detach().numpy() / 1000
    # hand_j = hand_j.detach().numpy() / 1000
    #########################################################################################################################
    print("Side: ", mano_side)
    
    global_orient=torch.from_numpy(hand_pose[:, :3])
    transl=torch.from_numpy(hand_pose[:, 48:])
    hand_pose=torch.from_numpy(hand_pose[:, 3:48])
    out = mano_layers[mano_side](betas=betas,
                                 global_orient=global_orient,
                                 hand_pose=hand_pose,
                                 transl=transl,
                                 return_full_pose=True)

    obj_v_2048 = np.matmul(
        obj_rot.reshape(-1, 3, 3),
        obj_v_2048.reshape(1, -1, 3).swapaxes(1, 2)
    ).swapaxes(1, 2) + obj_trans.reshape(-1, 1, 3)
    hand_v = out.vertices.detach().cpu().numpy()
    hand_j = out.joints.detach().cpu().numpy()

    full_pose = out.full_pose.detach()
    full_pose = torch.cat([full_pose, transl], dim=1)
    
    out = mano_layers_flat[mano_side](betas=betas,
                                 global_orient=full_pose[:, :3],
                                 hand_pose=full_pose[:, 3:48],
                                 transl=full_pose[:, 48:])
    hand_v_1 = out.vertices.detach().cpu().numpy()
    hand_j_1 = out.joints.detach().cpu().numpy()
    assert (hand_v_1- hand_v < 1e-6).all()
    assert (hand_j_1 - hand_j < 1e-6).all()
    
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
    # for i in range(0, 10):
    #     save_point_cloud(
    #         np.concatenate([hand_v[i], obj_v_2048[i]], axis=0),
    #         np.concatenate([np.zeros(hand_v[i].shape[0]), hand_v[i,:,0]-np.min(hand_v[i,:,0])], axis=0),
    #         f"./.exps/debug/{i}.ply"
        # )
    dist_matrix = hand_j.reshape(-1, 1, 21, 3) - obj_v_2048.reshape(-1, 2048, 1, 3)
    dist_matrix = np.linalg.norm(dist_matrix, axis=-1)
    has_contact = (dist_matrix < 0.02).sum(axis=(1,2)) > 0

    global right_cnt, left_cnt
    if mano_side == "right":
        right_cnt += full_pose.shape[0]
    else:
        left_cnt += full_pose.shape[0]
    return {
        "hand_pcd": hand_v,
        "hand_joints": hand_j,
        "obj_rot": obj_rot,
        "obj_trans": obj_trans,
        "hand_pose": full_pose.numpy(),
        "has_contact": has_contact,
        "is_right": 1 if mano_side == "right" else 0,
        "betas": betas[0].numpy()
    }

def process_pose(file, data_path, export_path, meta_file_writer, export_seq=False):
    meta_file = file.replace('pose.npz', 'meta.yml')
    with open(meta_file, 'r') as f:
        meta = yaml.load(f, Loader=yaml.FullLoader)
    mano_calib_file = os.path.join(data_path, "calibration", "mano_{}".format(meta['mano_calib'][0]), "mano.yml")
    
    with open(mano_calib_file, 'r') as f:
        mano_calib = yaml.load(f, Loader=yaml.FullLoader)
    mano_betas = mano_calib['betas']
    obj_id = meta['ycb_ids'][meta['ycb_grasp_ind']]
    obj_name = _YCB_CLASSES[obj_id]
    global mesh_d
    obj_pcd, obj_normal, obj_pcd_2048, obj_normal_2048, obj_2048_idx, faces, sampled_faces = mesh_d[obj_name]
    seq_data = get_hand_grasp(file, meta['mano_sides'][0], meta['ycb_grasp_ind'], mano_betas, obj_pcd_2048)

    export_dir = file.replace(data_path, export_path).replace('/pose.npz', '')
    if export_seq:
        # The original data is 120 fps, downsample to 15 fps clips
        export_path = export_dir + '.h5'
        save_hdf5(seq_data, export_path)
        meta_file_writer.write(f"{export_path}\n")
    else:
        os.makedirs(export_dir, exist_ok=True)
        for i in range(seq_data['hand_pcd'].shape[0]):
            if seq_data['is_right']:
                export_path = os.path.join(export_dir, f"r_{i}.h5")
            else:
                export_path = os.path.join(export_dir, f"l_{i}.h5")
            save_hdf5({
                'hand_pcd': seq_data['hand_pcd'][i],
                'hand_joints': seq_data['hand_joints'][i],
                'obj_trans': seq_data['obj_trans'][i],
                'obj_rot': seq_data['obj_rot'][i],
                'hand_pose': seq_data['hand_pose'][i],
                'has_contact': seq_data['has_contact'][i],
                'obj_name': obj_name,
                'is_right': seq_data['is_right'],
                "betas": seq_data['betas']
            }, export_path)
            meta_file_writer.write(f"{export_path} {seq_data['has_contact'][i]}\n")
            # if seq_data['has_contact'][i]:
            #     import ipdb; ipdb.set_trace()


import argparse
def construct_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="./data/dexYCB/dexYCB_pose"
    )
    parser.add_argument(
        "--filter", type=str, default=""
    )
    parser.add_argument(
        "--seq", action='store_true'
    )
    parser.add_argument(
        "--target_path", type=str, default="./data/dexYCB/dexYCB_muchen"
    )
    args = parser.parse_args()
    return args

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

if __name__ == "__main__":
    args = construct_args()
    
    global mesh_d
    mesh_d = {}
    all_mesh = glob.glob(os.path.join(args.data_path, 'models/**/textured_simple.obj'))
    for mesh_f in tqdm(all_mesh):
        # obj_name = os.path.basename(mesh_f).split('.stl')[0]
        mesh = trimesh.load(mesh_f, force='mesh')
        pcd = np.array(mesh.vertices)
        normals = np.array(mesh.vertex_normals)
        faces = np.array(mesh.faces)
        sampled_pcd, idx = sample_farthest_points(torch.from_numpy(pcd).view(1, -1, 3), K=2048)
        sampled_pcd = sampled_pcd[0].numpy()
        idx = idx[0].numpy()
        full_obj_name = os.path.basename(os.path.dirname(mesh_f))

        _, sampled_faces = decimate_mesh(pcd, faces, idx)
        mesh_d[full_obj_name] = pcd, normals, sampled_pcd, normals[idx], idx, faces, sampled_faces
        # save ply mesh
        mesh = trimesh.Trimesh(vertices=sampled_pcd, faces=sampled_faces)
        mesh.export(f'.exps/debug/{full_obj_name}.ply')

    os.makedirs(args.target_path, exist_ok=True)
    np.save(os.path.join(args.target_path, "seqs.npy"), mesh_d)

    f = open(os.path.join(args.target_path, "seqs.txt"), "w")
    files = glob.glob(os.path.join(args.data_path, '**/**/pose.npz'))
    print(f"Total {len(files)} Sequences")
    for file in tqdm(files):
        process_pose(file, args.data_path, args.target_path, f, args.seq)
    f.close()
    print("Right Hand: ", right_cnt)
    print("Left Hand: ", left_cnt)