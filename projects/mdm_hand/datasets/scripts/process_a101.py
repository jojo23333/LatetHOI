
import numpy as np
import logging

import os.path as osp
import json
import h5py
from tqdm import tqdm
from pycocotools.coco import COCO
import argparse
import os

ANNOT_VERSION = "v1-1"
root_joint_idx = {"right": 20, "left": 41}

img_path_base = "/mnt/graphics_ssd/nimble/datasets/AssemblyHands/cvpr_release/images"
annot_path = "/mnt/graphics_ssd/nimble/datasets/AssemblyHands/cvpr_release/annotations"
modality = "ego"
logger = logging.getLogger(__name__)
joint_num = 21

def preprocess_data(data_mode):
    db = COCO(
        osp.join(
            annot_path,
            data_mode,
            "assemblyhands_"
            + data_mode
            + f"_{modality}_data_{ANNOT_VERSION}.json",
        )
    )
    with open(
        osp.join(
            annot_path,
            data_mode,
            "assemblyhands_"
            + data_mode
            + f"_{modality}_calib_{ANNOT_VERSION}.json",
        )
    ) as f:
        cameras = json.load(f)["calibration"]
    with open(
        osp.join(
            annot_path,
            data_mode,
            "assemblyhands_" + data_mode + f"_joint_3d_{ANNOT_VERSION}.json",
        )
    ) as f:
        joints = json.load(f)["annotations"]

    logger.info("Get bbox and root depth from groundtruth annotation")
    
    # NOTE we do not care about unvalid pose for now
    
    annot_list = db.anns.keys()
    cam_dict = {}
    pose_dict = {}
    for i, aid in tqdm(enumerate(annot_list)):
        ann = db.anns[aid]
        image_id = ann["image_id"]
        img = db.loadImgs(image_id)[0]

        seq_name = str(img["seq_name"])
        camera_name = img["camera"]
        frame_idx = img["frame_idx"]
        file_name = img["file_name"]
        img_path = osp.join(img_path_base, file_name)
        assert osp.exists(img_path), f"Image path {img_path} does not exist"
        K = np.array(
            cameras[seq_name]["intrinsics"][camera_name + "_mono10bit"],
            dtype=np.float32,
        )
        Rt = np.array(
            cameras[seq_name]["extrinsics"][f"{frame_idx:06d}"][
                camera_name + "_mono10bit"
            ],
            dtype=np.float32,
        )
        world2ego = np.concatenate([Rt, np.array([[0, 0, 0, 1]])], axis=0)
        joint_world = np.array(
            joints[seq_name][f"{frame_idx:06d}"]["world_coord"], dtype=np.float32
        )
        # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
        # joint_valid[joint_type['right']] *= joint_valid[root_joint_idx['right']]
        # joint_valid[joint_type['left']] *= joint_valid[root_joint_idx['left']]
        
        if camera_name not in cam_dict.keys():
            cam_dict[camera_name] = {}
        if seq_name not in cam_dict[camera_name]:
            cam_dict[camera_name][seq_name] = {}
        cam_dict[camera_name][seq_name][frame_idx] = world2ego

        if seq_name not in pose_dict.keys():
            pose_dict[seq_name] = {}
        if frame_idx not in pose_dict[seq_name]:
            pose_dict[seq_name][frame_idx] = joint_world
        else:
            assert (pose_dict[seq_name][frame_idx] == joint_world).all()

    cam_names = list(cam_dict.keys())
    cam_names = ['HMC_84358933', 'HMC_84355350', 'HMC_21179183', 'HMC_21110305']

    data = {}
    for seq_name in pose_dict:
        sub_seq_id = 0
        frames = sorted(list(pose_dict[seq_name].items()), key=lambda x: x[0])
        len_frames = len(frames)
        queue = [frames[0]]
        for fid, joint_world in frames[1:]:
            if fid - queue[-1][0] > 2 or fid == frames[-1][0]:
                if len(queue) < 10:
                    queue = [(fid, joint_world)]
                    continue
                new_seq_name = f"{seq_name}_SPLIT{sub_seq_id}"
                print("New Seq:", new_seq_name, f"From frame {queue[0][0]} to {queue[-1][0]}")
                selected_frame_ids = [x[0] for x in queue]
                data[new_seq_name] = {}
                data[new_seq_name]['world_coord'] = np.array([x[1] for x in queue])
                data[new_seq_name]['world2ego'] = []
                for cam in cam_names:
                    if seq_name not in cam_dict[cam]:
                        print(f"Warning {seq_name} have {cam_dict[cam].keys()} and do not contain {cam}. Probably due to split difference.")
                        continue
                    default_cam_pos = list(cam_dict[cam][seq_name].values())[0]
                    cam_l = []
                    for x in selected_frame_ids:
                        if x in cam_dict[cam][seq_name]:
                            cam_l.append(cam_dict[cam][seq_name][x])
                        else:
                            cam_l.append(default_cam_pos)
                    data[new_seq_name]['world2ego'].append(cam_l)
                queue = [(fid, joint_world)]
                sub_seq_id = sub_seq_id + 1
            else:
                queue.append((fid, joint_world))

    out_p = f"./data/motion/a101/{data_mode}.npy"
    out_folder = osp.dirname(out_p)
    if not osp.exists(out_folder):
        os.makedirs(out_folder)
    logger.info("Dumping data")
    np.save(out_p, data)
    return data

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
    preprocess_data('train')
    preprocess_data('val')

    