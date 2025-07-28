# modified from src/arctic/split.py in the ARCTIC CODE

import json
import os
import sys
import os.path as op

sys.path.append('.')
sys.path.append('..')
sys.path.append(os.path.dirname(__file__))
import argparse

import torch
import numpy as np
from loguru import logger
from tqdm import tqdm
from utils.rotation_conversions import axis_angle_to_rotation_6d, rotation_6d_to_axis_angle, axis_angle_to_quaternion
from smplx import MANO
from .tools.quaternion import *
from .tools.paramUtil import *
from .tools.pcd_helper import voxelize_batch
from scipy.spatial.transform import Rotation as R


MANO_MODEL_DIR = './data/ARCTIC/body_models/mano'
right_hand_mano = MANO(
    MANO_MODEL_DIR,
    create_transl=True,
    use_pca=False,
    flat_hand_mean=True,
    is_rhand=True,
)
left_hand_mano = MANO(
    MANO_MODEL_DIR,
    create_transl=True,
    use_pca=False,
    flat_hand_mean=True,
    is_rhand=False,
)#.hand_mean.view(1, 45).numpy()

def axis_angle_to_rotation_6d_np(x):
    x = torch.from_numpy(x).contiguous().float()
    return axis_angle_to_rotation_6d(x).numpy()

def rotation_6d_to_axis_angle_np(x):
    x = torch.from_numpy(x).contiguous().float()
    return rotation_6d_to_axis_angle(x).numpy()

# view 0 is the egocentric view
_VIEWS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
_SUBJECTS = [
    "s01",  # F
    "s02",  # F
    "s03",  # M
    "s04",  # M
    "s05",  # F
    "s06",  # M
    "s07",  # M
    "s08",  # F
    "s09",  # F
    "s10",  # M
]
arctic_objects = [
    "capsulemachine",
    "box",
    "ketchup",
    "laptop",
    "microwave",
    "mixer",
    "notebook",
    "espressomachine",
    "waffleiron",
    "scissors",
    "phone",
]

OBJ_INIT_ROT = {k: [] for k in arctic_objects}

def get_selected_seqs(setup, split):
    assert split in ["train", "val", "test"]

    # load seq names from json
    with open(
        op.join("./data/ARCTIC/arctic_data/data/splits_json/", f"protocol_{setup}.json"), "r"
    ) as f:
        splits = json.load(f)

    train_seqs = splits["train"]
    val_seqs = splits["val"]
    test_seqs = splits["test"]

    # sanity check no overlap seqs
    all_seqs = train_seqs + val_seqs + test_seqs
    val_test_seqs = val_seqs + test_seqs
    assert len(set(val_test_seqs)) == len(set(val_seqs)) + len(set(test_seqs))
    for seq in val_test_seqs:
        if seq not in all_seqs:
            logger.info(seq)
            assert False, f"{seq} not in all_seqs"

    train_seqs = [seq for seq in all_seqs if seq not in val_test_seqs]
    all_seqs = train_seqs + val_test_seqs
    assert len(all_seqs) == len(set(all_seqs))

    # return
    if split == "train":
        return train_seqs
    if split == "val":
        return val_seqs
    if split == "test":
        return test_seqs


def get_selected_views(setup, split):
    # return view ids to use based on setup and split
    assert split in ["train", "val", "test"]
    assert setup in [
        "p1",
        "p2",
        "p1a",
        "p2a",
    ]
    # only static views
    if setup in ["p1", "p1a"]:
        return _VIEWS[1:]

    # seen ego view
    if setup in ["p2", "p2a"]:
        return _VIEWS[:1]


def sanity_check_splits(protocol):
    # make sure no overlapping seq
    train_seqs = get_selected_seqs(protocol, "train")
    val_seqs = get_selected_seqs(protocol, "val")
    test_seqs = get_selected_seqs(protocol, "test")
    all_seqs = list(set(train_seqs + val_seqs + test_seqs))
    assert len(train_seqs) == len(set(train_seqs))
    assert len(val_seqs) == len(set(val_seqs))
    assert len(test_seqs) == len(set(test_seqs))

    train_seqs = set(train_seqs)
    val_seqs = set(val_seqs)
    test_seqs = set(test_seqs)
    assert len(set.intersection(train_seqs, val_seqs)) == 0
    assert len(set.intersection(train_seqs, test_seqs)) == 0
    assert len(set.intersection(test_seqs, val_seqs)) == 0
    assert len(all_seqs) == len(train_seqs) + len(val_seqs) + len(test_seqs)


def sanity_check_annot(seq_name, data):
    # make sure no NaN or Inf
    num_frames = data["params"]["pose_r"].shape[0]
    for pkey, side_dict in data.items():
        if isinstance(side_dict, dict):
            for key, val in side_dict.items():
                if "smplx" in key:
                    # smplx distortion can be undefined
                    continue
                assert np.isnan(val).sum() == 0, f"{seq_name}: {pkey}_{key} has NaN"
                assert np.isinf(val).sum() == 0, f"{seq_name}: {pkey}_{key} has Inf"
                assert num_frames == val.shape[0]
        else:
            if "smplx" in pkey:
                # smplx distortion can be undefined
                continue
            assert np.isnan(side_dict).sum() == 0, f"{seq_name}: {pkey}_{key} has NaN"
            assert np.isinf(side_dict).sum() == 0, f"{seq_name}: {pkey}_{key} has Inf"
            assert num_frames == side_dict.shape[0]

def get_obj_type(seq_name):
    for i, name in enumerate(arctic_objects):
        if name in seq_name:
            return name

def get_cont6d_params(poses, positions, obj_poses, obj_init_rot):
    '''Quaternion to continuous 6D'''

    pose_shape = poses.shape
    pose_mats = R.from_rotvec(poses.reshape(-1,3)).as_matrix()
    cont_6d_params = matrix_to_rotation_6d(torch.tensor(pose_mats)).numpy().reshape(pose_shape[0],-1,6)

    obj_rot = R.from_rotvec(obj_poses[:,3:]).as_matrix() #poses_quat.copy() #quat_params[:, 0].copy()
    obj_rot_quat = quaternion_scipy_to_mdm(R.from_rotvec(obj_poses[:,3:]).as_quat())

    '''Root Linear Velocity'''
    # (seq_len - 1, 3)
    velocity = (obj_poses[1:, :3] - obj_poses[:-1, :3]).copy()

    obj_lin_velocity = qrot_np(obj_rot_quat[1:], velocity)
    '''Root Angular Velocity'''
    # (seq_len - 1, 4)
    obj_ang_velocity = qmul_np(obj_rot_quat[1:], qinv_np(obj_rot_quat[:-1]))
    cont_6d_params[:,0] = matrix_to_rotation_6d(torch.tensor(np.matmul(obj_rot.swapaxes(1,2),pose_mats.reshape(pose_shape[0],-1,3,3)[:,0])))
    cont_6d_params[:,16] = matrix_to_rotation_6d(torch.tensor(np.matmul(obj_rot.swapaxes(1,2),pose_mats.reshape(pose_shape[0],-1,3,3)[:,16])))

    return cont_6d_params, obj_ang_velocity, obj_lin_velocity, obj_rot_quat, None, None, obj_rot

def get_obj_velocity(obj_poses):
    obj_rot = R.from_rotvec(obj_poses[:,3:]).as_quat()
    '''Obj Linear Velocity'''
    obj_lvelocity = (obj_poses[1:, :3] - obj_poses[:-1, :3]).copy()
    obj_rot_lvel = qrot_np(obj_rot[1:], obj_lvelocity)
    '''Obj Angular Velocity'''
    obj_rot_qvel = R.from_quat(qmul_np(obj_rot[1:], qinv_np(obj_rot[:-1]))).as_rotvec()
    return obj_rot_lvel, obj_rot_qvel

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

def generate_feature_vector(positions, rotations, obj_poses, obj_arti, obj_init_rot, obj_verts_sampled=None, only_hands=False):
    '''
        positions: [nf, 42, 3]
        rotations: [nf, 32, 3]
        obj_poses: [nf, 6]
        
    '''
    obj_poses[:, 3:] = R.from_matrix(np.matmul(
        np.repeat(obj_init_rot[None, :], obj_poses.shape[0], axis=0).swapaxes(1, 2),
        R.from_rotvec(obj_poses[:, 3:]).as_matrix()
    )).as_rotvec()

    obj_pos_init =  obj_poses[:1,:3]
    positions = positions - obj_pos_init[:,np.newaxis,:]

    obj_poses[:,:3] = obj_poses[:,:3] - obj_pos_init

    # obj_verts = np.tile(obj_verts_sampled, (obj_poses.shape[0], 1,1))
    # obj_verts[:,0] = np.zeros_like(obj_verts[:,0])
    # obj_verts = np.matmul(obj_verts, R.from_rotvec(obj_poses[:,3:]).as_matrix())
    # obj_verts += obj_poses[:,np.newaxis,:3]
    # use single point for now
    obj_verts = obj_poses[:,np.newaxis,:3].copy()

    global_object_positions = obj_verts.copy()
    global_positions = positions.copy()

    cont_6d_params, obj_ang_velocity, obj_lin_velocity, object_root_rot, lwrist_rel_or, rwrist_rel_or, obj_rot_mat = get_cont6d_params(rotations, positions, obj_poses, obj_init_rot)

    positions, positions_object = get_rifke(positions, obj_rot_mat, obj_verts.copy())

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
    obj_rot_6D = matrix_to_rotation_6d(torch.tensor(R.from_quat(obj_rot_quat).as_matrix())).numpy()

    '''Object velocity'''
    ### Let's take it relative to the last frame
    obj_lvel, obj_qvel = get_obj_velocity(obj_poses)
    local_obj_vel = qrot_np(np.repeat(object_root_rot[:-1, None], global_object_positions.shape[1], axis=1),
                        global_object_positions[1:] - global_object_positions[:-1])
    local_obj_vel = local_obj_vel.reshape(len(local_obj_vel), -1)

    '''Contact information'''
    ### TO ADD
    feature_vector = np.concatenate([relative_positions_left, relative_positions_right,
                                     relative_rotations_left, relative_rotations_right,
                                     local_vel_left, local_vel_right,
                                     relative_obj_positions[:,:3], obj_rot_6D[:-1], obj_arti[:-1],
                                     obj_lvel, obj_qvel], axis=-1)
    return feature_vector, global_positions


arctic_out = 'arctic_reca'
def normalize_data(data, split):
    MANO_MODEL_DIR = './data/ARCTIC/body_models/mano'
    right_pose_mean = MANO(
        MANO_MODEL_DIR,
        create_transl=True,
        use_pca=False,
        flat_hand_mean=False,
        is_rhand=True,
    ).hand_mean.view(1, 45).numpy()
    left_pose_mean = MANO(
        MANO_MODEL_DIR,
        create_transl=True,
        use_pca=False,
        flat_hand_mean=False,
        is_rhand=False,
    ).hand_mean.view(1, 45).numpy()

    # For Arctic Coarsly only keep manipulation frames
    data_all = {}
    vel_eps = 2.0 # mm
    for seq_name, seq in data.items():
        # vel = data[seq_name]['trans_obj'][:-1] - data[seq_name]['trans_obj'][1:]
        # vel = np.abs(vel).sum(-1)
        # start_i = 0
        # end_i = vel.shape[0] - 1
        # while vel[start_i] < vel_eps:
        #     start_i = start_i + 1
        #     assert start_i < end_i, vel
        # while vel[end_i] < vel_eps:
        #     end_i = end_i - 1
        #     assert end_i > start_i, vel
        data[seq_name] = {k: v if k=='type_obj' else v[30:-30] for k, v in seq.items()}

    for seq_name, seq in data.items():
        seq['trans_obj'] = seq['trans_obj'] / 1000
        nf = seq['rot_l'].shape[0]
        positions = seq['world_coord']
        
        full_pose = np.concatenate(
            (
                seq['rot_l'],
                seq['pose_l'] + left_pose_mean,
                seq['rot_r'],
                seq['pose_r'] + right_pose_mean
            ),
            axis=-1
        )
        obj_pose = np.concatenate((seq['trans_obj'], seq['rot_obj']), axis=-1)
        obj_arti = seq['arti_obj'].reshape(nf, 1)

        obj_init_rot = OBJ_INIT_ROT[seq['type_obj']]
        feature_vector, glob_pos = generate_feature_vector(positions, full_pose, obj_pose, obj_arti, obj_init_rot)
        # data = process_combined_feat(feature_vector)
        seq["combined_feat"] = feature_vector
        seq["obj_init_rot"] = obj_init_rot
        # x = left_hand_mano(
        #     betas=torch.from_numpy(seq['shape_l']).to(torch.float),
        #     global_orient=torch.from_numpy(seq['rot_l']).to(torch.float),
        #     hand_pose=torch.from_numpy((seq['pose_l']+left_pose_mean).reshape(nf, -1)).to(torch.float),
        # ).joints.detach().numpy()
        # x1 = right_hand_mano(
        #     betas=torch.from_numpy(seq['shape_r']).to(torch.float),
        #     global_orient=torch.from_numpy(seq['rot_r']).to(torch.float),
        #     hand_pose=torch.from_numpy((seq['pose_r']+right_pose_mean).reshape(nf, -1)).to(torch.float),
        # ).joints.detach().numpy()
        # import ipdb; ipdb.set_trace()
        # 6d continouse rotatation
        seq['rot_l'] = axis_angle_to_rotation_6d_np(seq['rot_l'].reshape(nf, 3))
        seq['pose_l'] = axis_angle_to_rotation_6d_np((seq['pose_l']+left_pose_mean).reshape(nf, 15, 3)).reshape(nf, 90)
        seq['rot_r'] = axis_angle_to_rotation_6d_np(seq['rot_r'].reshape(nf, 3))
        seq['pose_r'] = axis_angle_to_rotation_6d_np((seq['pose_r']+right_pose_mean).reshape(nf, 15, 3)).reshape(nf, 90)
        seq['rot_obj'] = axis_angle_to_rotation_6d_np(seq['rot_obj'].reshape(nf, 3))
        # recenter with initial object position
        seq['trans_l'] = seq['trans_l'] - seq['trans_obj'][0:1]
        seq['trans_r'] = seq['trans_r'] - seq['trans_obj'][0:1]
        seq['rel_world_coord'] = (seq['world_coord'] - seq['world_coord'][:, 20:21, :]).reshape(nf, -1)
        seq['vol_world_coord'] = seq['world_coord'][1:] - seq['world_coord'][:-1]
        seq['vol_world_coord'] = np.concatenate([seq['vol_world_coord'], seq['vol_world_coord'][-1:]], axis=0).reshape(nf, -1)
        seq['world_coord'] = (seq['world_coord'] - seq['trans_obj'][0:1, None, :]).reshape(nf, -1)
        seq['trans_obj'] = seq['trans_obj'] - seq['trans_obj'][0:1]
        data[seq_name] = seq
        for k in seq:
            if k in data_all.keys():
                data_all[k].append(seq[k])
            else:
                data_all[k] = []

    mean = {}
    std = {}
    # need_normalize = lambda k: 'trans' in k or 'rot' in k or 'pose' in k or 'arti' in k or 'world_coord' in k
    # if split == 'train':
    #     for k in data_all.keys():
    #         if need_normalize(k):
    #             data_all[k] = np.concatenate(data_all[k], axis=0)
    #     nf_all = data_all['rot_l'].shape[0]
    #     for k in data_all.keys():
    #         if need_normalize(k):
    #             mean[k] = np.mean(data_all[k].reshape(nf_all, -1), axis=0, keepdims=True)
    #             std[k] = np.std(data_all[k].reshape(nf_all, -1), axis=0, keepdims=True)
    # elif split == 'val':
    #     mean = np.load(f"./data/motion/{arctic_out}/train_mean.npy", allow_pickle=True).item()
    #     std = np.load(f"./data/motion/{arctic_out}/train_std.npy", allow_pickle=True).item()

    # for seq_name, seq in data.items():
    #     for k in seq.keys():
    #         if need_normalize(k):
    #             std[k][std[k] == 0.] = 1
    #             assert (std[k] > 0).all()
    #             data[seq_name][k] = (seq[k] - mean[k]) / std[k]
    return data, mean, std


def build_split(protocol, split, process_folder):
    logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    logger.info(f"Constructing split {split} for protocol {protocol}")
    # extract seq_names
    # unpack protocol
    sanity_check_splits(protocol)
    chosen_seqs = get_selected_seqs(protocol, split)
    logger.info(f"Chosen {len(chosen_seqs)} seqs:")
    logger.info(chosen_seqs)
    chosen_views = get_selected_views(protocol, split)
    logger.info(f"Chosen {len(chosen_views)} views:")
    logger.info(chosen_views)
    fseqs = chosen_seqs

    # do not need world in reconstruction
    data_dict = {}
    for seq in tqdm(fseqs):
        seq_p = op.join(process_folder, f"{seq}.npy")
        if "_verts" in seq_p:
            logger.warning(
                "Trying to build split with verts. This will require lots of storage"
            )
        data = np.load(seq_p, allow_pickle=True).item()
        sanity_check_annot(seq_p, data)
        data_dict[seq] = {
            "world_coord": np.concatenate([data['world_coord']['joints.left'], data['world_coord']['joints.right']], axis=1),
            # "world2ego": data["params"]["world2ego"],
            "cam_coord":  np.concatenate((data['cam_coord']['joints.left'][:, 0, ...], data['cam_coord']['joints.right'][:, 0, ...]), axis=1),
            "rot_l": data["params"]["rot_l"],
            "pose_l": data["params"]["pose_l"],
            "trans_l": data["params"]["trans_l"],
            "shape_l": data["params"]["shape_l"],
            "rot_r": data["params"]["rot_r"],
            "pose_r": data["params"]["pose_r"],
            "trans_r": data["params"]["trans_r"],
            "shape_r": data["params"]["shape_r"],
            "arti_obj": data["params"]["obj_arti"].reshape(-1, 1),
            "rot_obj": data["params"]["obj_rot"],
            "trans_obj": data["params"]["obj_trans"],
            "type_obj": get_obj_type(seq),
        }
        if split == 'train':
            OBJ_INIT_ROT[get_obj_type(seq)].append(
                R.from_rotvec(data["params"]["obj_rot"][0]).as_quat()
            )
    if split == 'train':
        for k, v in OBJ_INIT_ROT.items():
            quat_sum = np.array(v).sum(axis=0)
            OBJ_INIT_ROT[k] = R.from_quat(
                quat_sum / np.linalg.norm(quat_sum)
            ).as_matrix()

    out_data = data_dict
    out_p = f"./data/motion/{arctic_out}/{split}.npy"
    out_folder = op.dirname(out_p)
    if not op.exists(out_folder):
        os.makedirs(out_folder)
    print("Dumping data")
    np.save(out_p, out_data)

    data, mean, std = normalize_data(out_data, split)
    np.save(f"./data/motion/{arctic_out}/{split}_normalized.npy", data)
    np.save(f"./data/motion/{arctic_out}/{split}_mean.npy", mean)
    np.save(f"./data/motion/{arctic_out}/{split}_std.npy", std)

# MANO's id in A101 order
MANO_TO_A101 = np.array(
               [16, 15, 14, 13, 
                17,  3,  2,  1,
                18,  6,  5,  4,
                19, 12, 11, 10,
                20,  9,  8,  7,
                0])

def mano_to_a101(left, right):
    new_left = left[:, MANO_TO_A101, :]
    new_right = right[:, MANO_TO_A101, :]
    a101 = np.concatenate([new_right, new_left], axis=1)
    return a101

def construct_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test", "all"],
        default=None,
    )
    parser.add_argument(
        "--process_folder", type=str, default="./data/ARCTIC/arctic_data/data/processed/seqs"
    )
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    from IPython import get_ipython
    from ipdb.__main__ import _get_debugger_cls
    import ipdb
    debugger = _get_debugger_cls()
    shell = get_ipython()
    shell.debugger_history_file = '.pdbhistory'
    args = construct_args()
    protocol = 'p2'
    split = args.split


    build_split(protocol, 'train', args.process_folder)
    build_split(protocol, 'val', args.process_folder)
