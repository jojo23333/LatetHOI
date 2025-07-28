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
    flat_hand_mean=False,
    is_rhand=True,
)
left_hand_mano = MANO(
    MANO_MODEL_DIR,
    create_transl=True,
    use_pca=False,
    flat_hand_mean=False,
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

obj_init_pose={
    "capsulemachine": "s01/capsulemachine_grab_01",
    "box": 's01/box_grab_01',
    "ketchup": "s01/ketchup_grab_01",
    "laptop": 's01/laptop_grab_01',
    "microwave": "s01/microwave_grab_01",
    "mixer": 's01/mixer_grab_01',
    "notebook": "s04/notebook_grab_01",
    "espressomachine": 's01/espressomachine_grab_01',
    "waffleiron": "s01/waffleiron_grab_01",
    "scissors": 's01/scissors_grab_01',
    "phone": 's01/phone_grab_01',
}


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

def get_cont6d_params(poses, positions, obj_poses):
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

def generate_feature_vector(positions, rotations, obj_poses, obj_arti, obj_init_rot):
    '''
        positions: [nf, 42, 3]
        rotations: [nf, 32, 3]
        obj_poses: [nf, 6]
        
    '''
    positions = copy.deepcopy(positions)
    rotations = copy.deepcopy(rotations)
    obj_poses = copy.deepcopy(obj_poses)
    obj_arti = copy.deepcopy(obj_arti)
    if obj_init_rot is not None:
        obj_init_rot = copy.deepcopy(obj_init_rot)

        obj_poses[:, 3:] = R.from_matrix(np.matmul(
            np.repeat(obj_init_rot[None, :], obj_poses.shape[0], axis=0).swapaxes(1, 2),
            R.from_rotvec(obj_poses[:, 3:]).as_matrix()
        )).as_rotvec()

    obj_pos_init =  obj_poses[:1,:3]
    positions = positions - obj_pos_init[:,np.newaxis,:]

    obj_poses[:,:3] = obj_poses[:,:3] - obj_pos_init
    obj_verts = obj_poses[:,np.newaxis,:3].copy()

    global_object_positions = obj_verts.copy()
    global_positions = positions.copy()

    cont_6d_params, obj_rot_mat = get_cont6d_params(rotations, positions, obj_poses)
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
    relative_obj_positions = global_object_positions
    relative_obj_positions[:,1:] = relative_obj_positions[:,1:] - global_object_positions[:,:1]#obj_distance

    relative_obj_positions[:,0] = positions_object[:,0]

    relative_obj_positions = relative_obj_positions[:-1].reshape(len(positions)-1, -1)
    '''Object rotation 6d'''
    ### Let's take it relative to the initial frame
    obj_rot_quat = R.from_rotvec(obj_poses[:, 3:6]).as_quat()
    obj_rot_6D = matrix_to_rotation_6d(torch.tensor(R.from_quat(obj_rot_quat).as_matrix())).numpy()

    '''Contact information'''
    feature_vector = np.concatenate([relative_positions_left, relative_positions_right,
                                     relative_rotations_left, relative_rotations_right,
                                     local_vel_left, local_vel_right,
                                     ], axis=-1)
    cond = np.concatenate([relative_obj_positions[:,:3], obj_rot_6D[:-1], obj_arti[:-1],], axis=-1)
    return feature_vector, cond

def read_from_raw(fseqs, process_folder):
    right_pose_mean = right_hand_mano.hand_mean.view(1, 45).numpy()
    left_pose_mean = left_hand_mano.hand_mean.view(1, 45).numpy()
    data_dict = {}
    for seq in tqdm(fseqs):
        seq_p = op.join(process_folder, f"{seq}.npy")
        if "_verts" in seq_p:
            logger.warning(
                "Trying to build split with verts. This will require lots of storage"
            )
        data = np.load(seq_p, allow_pickle=True).item()
        sanity_check_annot(seq_p, data)

        positions = np.concatenate([data['world_coord']['joints.left'], data['world_coord']['joints.right']], axis=1)
        full_pose = np.concatenate((
                data["params"]["rot_l"], data["params"]["pose_l"] + left_pose_mean,
                data["params"]["rot_r"], data["params"]["pose_r"] + right_pose_mean), axis=-1)
        obj_pose = np.concatenate((data["params"]["obj_trans"]/1000, data["params"]["obj_rot"]), axis=-1)
        obj_arti = data["params"]["obj_arti"].reshape(-1, 1)

        data_dict[seq] = {
            "keypoint": positions,
            "pose": full_pose,
            "shape_l": data["params"]["shape_l"],
            "shape_r": data["params"]["shape_r"],
            "obj_pose": obj_pose,
            "obj_arti": obj_arti,
            "obj_type": get_obj_type(seq),
            "seq_name": seq,
        }
    return data_dict

arctic_out = 'arctic'
def build_split(protocol, process_folder, export_single=True):
    # extract seq_names
    # unpack protocol
    sanity_check_splits(protocol)
    train_seqs = get_selected_seqs(protocol, 'train')
    logger.info(f"Train {len(train_seqs)} seqs:")
    logger.info(train_seqs)
    val_seqs = get_selected_seqs(protocol, 'val')
    logger.info(f"Val {len(train_seqs)} seqs:")
    logger.info(val_seqs)

    all_seqs = train_seqs + val_seqs

    data_dict = read_from_raw(all_seqs, process_folder)
    # for k, v in OBJ_INIT_ROT.items():
    #     quat_sum = np.array(v).sum(axis=0)
    #     OBJ_INIT_ROT[k] = R.from_quat(
    #         quat_sum / np.linalg.norm(quat_sum)
    #     ).as_matrix()
    # for k, v in obj_init_pose.items():
    #     seq_p = op.join(process_folder, f"{v}.npy")
    #     data = np.load(seq_p, allow_pickle=True).item()
    #     OBJ_INIT_ROT[k] = R.from_rotvec(data["params"]["obj_rot"][0]).as_matrix()

    # create combined feat
    feat_all = []
    for seq, data in data_dict.items():
        # data_dict[seq]['obj_init_rot'] = OBJ_INIT_ROT[data['obj_type']]
        combined_feat, obj_10d = generate_feature_vector(
            data['keypoint'], data['pose'], data['obj_pose'], data['obj_arti'], None#data['obj_init_rot']
        )
        data_dict[seq]['combined_feat'] = combined_feat
        data_dict[seq]['obj_10d'] = obj_10d
        feat_all.append(data['combined_feat'])

    # normalize combined feat
    feat_all = np.concatenate(feat_all, axis=0)
    mean_all = np.mean(feat_all, axis=0, keepdims=True)
    std_all = np.std(feat_all, axis=0, keepdims=True)
    mean_all = np.zeros_like(mean_all)
    std_all = np.ones_like(std_all)
    for seq, data in data_dict.items():
        data_dict[seq]['combined_feat'] = (data['combined_feat'] - mean_all) /  std_all

    meta = {
        "std": std_all,
        "mean": mean_all
    }

    out_folder = f'./data/motion/{arctic_out}'
    if not op.exists(out_folder):
        os.makedirs(out_folder)

    print("Dumping Training data")
    train_data_dict = {k: v for k, v in data_dict.items() if k in train_seqs}
    val_data_dict = {k: v for k, v in data_dict.items() if k in val_seqs}
    np.save(os.path.join(out_folder, 'train.npy'), train_data_dict)
    np.save(os.path.join(out_folder, 'val.npy'), val_data_dict)
    np.save(os.path.join(out_folder, 'meta.npy'), meta)
    if export_single:
        for seq, data in tqdm(data_dict.items()):
            out = op.join(out_folder, 'seq', data['seq_name']+'.npy')
            os.makedirs(op.dirname(out), exist_ok=True)
            np.save(out, {"gt": data})
    return


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

    build_split(protocol, args.process_folder)
