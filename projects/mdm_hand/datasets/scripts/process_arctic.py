# modified from src/arctic/split.py in the ARCTIC CODE

import json
import os
import os.path as op

import argparse

import torch
import numpy as np
from loguru import logger
from tqdm import tqdm
from utils.rotation_conversions import axis_angle_to_rotation_6d, rotation_6d_to_axis_angle
from smplx import MANO


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

    # For Arctic Coarsly jump over calibration Frame
    data_all = {}
    for seq_name, seq in data.items():
        data[seq_name] = {k: v if k=='type_obj' else v[30:-30] for k, v in seq.items()}

    for seq_name, seq in data.items():
        nf = seq['rot_l'].shape[0]
        # 6d continouse rotatation
        seq['rot_l'] = axis_angle_to_rotation_6d_np(seq['rot_l'].reshape(nf, 3))
        seq['pose_l'] = axis_angle_to_rotation_6d_np((seq['pose_l']+left_pose_mean).reshape(nf, 15, 3)).reshape(nf, 90)
        seq['rot_r'] = axis_angle_to_rotation_6d_np(seq['rot_r'].reshape(nf, 3))
        seq['pose_r'] = axis_angle_to_rotation_6d_np((seq['pose_r']+right_pose_mean).reshape(nf, 15, 3)).reshape(nf, 90)
        seq['rot_obj'] = axis_angle_to_rotation_6d_np(seq['rot_obj'].reshape(nf, 3))
        # recenter with initial object position
        seq['trans_obj'] = seq['trans_obj'] / 1000
        seq['trans_l'] = seq['trans_l'] - seq['trans_obj']
        seq['trans_r'] = seq['trans_r'] - seq['trans_obj']
        seq['rel_world_coord'] = (seq['world_coord'] - seq['world_coord'][:, 20:21, :]).reshape(nf, -1)
        seq['world_coord'] = (seq['world_coord'] - seq['trans_obj'][0:1, None, :]).reshape(nf, -1)
        seq['vol_world_coord'] = seq['world_coord'][1:] - seq['world_coord'][:-1]
        seq['vol_world_coord'] = np.concatenate([seq['vol_world_coord'], seq['vol_world_coord'][-1:]], axis=0)
        seq['trans_obj'] = seq['trans_obj'] - seq['trans_obj'][0:1]
        data[seq_name] = seq
        print(data[seq_name]['type_obj'])
        for k in seq:
            if k in data_all.keys():
                data_all[k].append(seq[k])
            else:
                data_all[k] = []

    need_normalize = lambda k: 'trans' in k or 'rot' in k or 'pose' in k or 'arti' in k or 'world_coord' in k
    if split == 'train':
        mean = {}
        std = {}
        for k in data_all.keys():
            if need_normalize(k):
                data_all[k] = np.concatenate(data_all[k], axis=0)
        nf_all = data_all['rot_l'].shape[0]
        for k in data_all.keys():
            if need_normalize(k):
                mean[k] = np.mean(data_all[k].reshape(nf_all, -1), axis=0, keepdims=True)
                std[k] = np.std(data_all[k].reshape(nf_all, -1), axis=0, keepdims=True)
    elif split == 'val':
        mean = np.load("./data/motion/arctic/train_mean.npy", allow_pickle=True).item()
        std = np.load("./data/motion/arctic/train_std.npy", allow_pickle=True).item()

    for seq_name, seq in data.items():
        for k in seq.keys():
            if need_normalize(k):
                std[k][std[k] == 0.] = 1
                assert (std[k] > 0).all()
                data[seq_name][k] = (seq[k] - mean[k]) / std[k]
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
            "world_coord": mano_to_a101(data['world_coord']['joints.left'], data['world_coord']['joints.right']),
            # "world2ego": data["params"]["world2ego"],
            "cam_coord":  mano_to_a101(data['cam_coord']['joints.left'][:, 0, ...], data['cam_coord']['joints.right'][:, 0, ...]),
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

    out_data = data_dict
    out_p = f"./data/motion/arctic/{split}.npy"
    out_folder = op.dirname(out_p)
    if not op.exists(out_folder):
        os.makedirs(out_folder)
    print("Dumping data")
    np.save(out_p, out_data)

    data, mean, std = normalize_data(out_data, split)
    np.save(f"./data/motion/arctic/{split}_normalized.npy", data)
    np.save(f"./data/motion/arctic/{split}_mean.npy", mean)
    np.save(f"./data/motion/arctic/{split}_std.npy", std)

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
    if protocol == "all":
        protocols = [
            "p1",  # allocentric
            "p2",  # egocentric
        ]
    else:
        protocols = [protocol]

    if split == "all":
        if protocol in ["p1", "p2"]:
            splits = ["train", "val", "test"]
        else:
            raise ValueError("Unknown protocol for option 'all'")
    else:
        splits = [split]

    for protocol in protocols:
        for split in splits:
            if protocol in ["p1", "p2"]:
                assert split not in ["test"], "val/test are hidden"
            build_split(protocol, split, args.process_folder)
