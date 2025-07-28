import os
import json
import pickle
import torch
import glob
import tqdm
from scipy.spatial.transform import Rotation as Rt
import numpy as np

FPS = 15
ACTIONS = ['none']

def read_rtd(file, num=0):
    with open(file, 'r') as f:
        cont = f.read()
        cont = eval(cont)
    if "dataList" in cont:
        anno = cont["dataList"][num]
    else:
        anno = cont["objects"][num]

    trans, rot, dim = anno["center"], anno["rotation"], anno["dimensions"]
    trans = np.array([trans['x'], trans['y'], trans['z']], dtype=np.float32)
    rot = np.array([rot['x'], rot['y'], rot['z']])
    dim = np.array([dim['length'], dim['width'], dim['height']], dtype=np.float32)
    rot = Rt.from_euler('XYZ', rot).as_rotvec()
    return np.array(rot, dtype=np.float32), trans, dim

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def read_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    return data

def read_hand_pose(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        rot, pose, shape, trans = data['poseCoeff'][:3], data['poseCoeff'][3:], data['beta'], data['trans']
        return {
            "rot": rot,
            "pose": pose,
            "shape": shape, 
            "trans": trans
        }
    except:
        return None

def load_data(ann_root, hand_root, seq_id):    
    # TODO Load Object Pose
    obj_pose_path = os.path.join(ann_root, seq_id, 'objpose', '*.json')  # Replace with actual filename
    # obj_pose = read_rtd(obj_pose_path)
    frames = glob.glob(obj_pose_path)
    
    # load Action Label
    action_label_path = os.path.join(ann_root, seq_id, 'action', 'color.json')
    action_annt = read_json(action_label_path)
    per_frame_action = torch.zeros(len(frames), dtype=int)
    if 'events' not in action_annt:
        with open("./data/hoi4d_error.txt", 'a') as f:
            print("Problematic Action")
            print(action_label_path, file=f)
    else:
        for action_seg in action_annt["events"]:
            action = action_seg["event"]
            s = int(action_seg["startTime"] * FPS)
            e = int(action_seg["endTime"] * FPS)
            if action not in ACTIONS:
                ACTIONS.append(action)
            aid = ACTIONS.index(action)
            per_frame_action[s:e] = aid

    # Load Hand Pose
    left_hand_path = os.path.join(hand_root, 'refinehandpose_left', seq_id)
    right_hand_path = os.path.join(hand_root, 'refinehandpose_right', seq_id)
    available_r_pose = glob.glob(os.path.join(right_hand_path, '*.pickle'))
    available_l_pose = glob.glob(os.path.join(left_hand_path, '*.pickle'))
    
    if len(available_r_pose) < 10:
        print("Not enough pose to form seq")
        return None

    # Seems that the dataset gurantee to have right hand while have a few left hand samples
    single_l_pose = []
    for x in available_l_pose:
        x_ = x.replace('refinehandpose_left', 'refinehandpose_right')
        if x_ not in available_r_pose:
            print(ann_root, hand_root, seq_id)
            # import ipdb; ipdb.set_trace()
            single_l_pose.append(x)
    
    available_r_pose.sort(key=lambda x:int(os.path.basename(x).split('.')[0]))
    available_r_pose = [(int(os.path.basename(x).split('.')[0]), x) for x in available_r_pose]
    r_pose_splits = []
    start = 0
    # Split on in-continous pose
    for i in range(1, len(available_r_pose)):
        if available_r_pose[i][0] - available_r_pose[i-1][0] > 1 or i == len(available_r_pose)-1:
            if i - start < 10:
                print("Not enough pose to form seq")
            else:
                r_pose_splits.append(available_r_pose[start:i])
            start = i

    # for each pose seq
    datas = []
    for r_pose_seq in r_pose_splits:
        end = r_pose_seq[-1][0]
        start = r_pose_seq[0][0]
        if end - start + 1 != len(r_pose_seq):
            import ipdb; ipdb.set_trace()

        hand_poses = []
        for fid, r_pose in r_pose_seq:
            r_hand = read_hand_pose(r_pose)
            if r_hand is None:
                with open("./data/hoi4d_error.txt", 'a') as f:
                    print("Problematic right Hand File")
                    print(r_pose, file=f)
                break
            # right hand
            l_pose = r_pose.replace('refinehandpose_right', 'refinehandpose_left')
            if l_pose in available_l_pose:
                l_hand = read_hand_pose(l_pose)
                if l_hand is None:
                    with open("./data/hoi4d_error.txt", 'a') as f:
                        print("Problematic Left Hand File")
                        print(l_pose, file=f)
                    l_hand = {k: np.zeros_like(v) for k, v in r_hand.items()}
            else:
                l_hand = {k: np.zeros_like(v) for k, v in r_hand.items()}
            r_hand = {k + '_r': v for k, v in r_hand.items()}
            l_hand = {k + '_l': v for k, v in l_hand.items()}
            hand_poses.append({**r_hand, **l_hand})
        
        if len(hand_poses) < 10:
            continue
        data = {
            k: np.stack([x[k] for x in hand_poses], axis=0)
            for k in hand_poses[0].keys()
        }
        data["action_labels"] = per_frame_action[start:end+1]
        datas.append(data)
    if len(single_l_pose) > 0:
        print("Left out", len(single_l_pose))
        if len(single_l_pose) > 10:
            import ipdb; ipdb.set_trace()
    return datas

def main(split='train'):
    root='./data/HOI4D/'
    ann_root = root + 'HOI4D_annotations/'
    hand_root = root + 'handpose/'
    pattern = os.path.join(ann_root, 'ZY2021080000*/H*/C*/N*/S*/s*/T*/')

    # Use glob to get all matching directories
    matching_dirs = glob.glob(pattern)
    out_data = {}
    # Print the matching directories
    for dir_path in tqdm.tqdm(matching_dirs):
        seq_id = dir_path.split(ann_root)[-1]
        print(f"Handling Sequence {seq_id}")
        datalist = load_data(ann_root, hand_root, seq_id)
        if datalist is None:
            print("Sequence ", seq_id, " do not have pose Annotation")
        else:
            for i, data in enumerate(datalist):
                out_data[seq_id + f'_split{i}'] = data

    out_p = f"./data/motion/hoi4d/{split}.npy"
    out_folder = os.path.dirname(out_p)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    print("Dumping data")
    np.save(out_p, out_data)

if __name__ == '__main__':
    from IPython import get_ipython
    from ipdb.__main__ import _get_debugger_cls
    import ipdb
    debugger = _get_debugger_cls()
    shell = get_ipython()
    shell.debugger_history_file = '.pdbhistory'
    main()


