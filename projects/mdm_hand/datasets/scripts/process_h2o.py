import torch
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse
import os.path as osp

train_sequences = ['subject1/h1', 'subject1/h2', 'subject1/k1', 'subject1/k2', 'subject1/o1', 'subject1/o2', 'subject2/h1', 'subject2/h2', 'subject2/k1', 'subject2/k2', 'subject2/o1', 'subject2/o2', 'subject3/h1', 'subject3/h2', 'subject3/k1']
val_sequences = ['subject3/k2', 'subject3/o1', 'subject3/o2']
test_sequences = ['subject4/h1', 'subject4/h2', 'subject4/k1', 'subject4/k2', 'subject4/o1', 'subject4/o2'] 

H2O_TO_A101 = torch.tensor(
               [4, 3, 2, 1,
                8, 7, 6, 5,
                12,11,10,9,
                16,15,14,13,
                20,19,18,17,
                0]
            )

# from visualize.hand_visualizer import HandPoseVisualizer
# vis = HandPoseVisualizer()

# read one split
def read_split(split):
    if split == 'train':
        return train_sequences
    elif split == 'val':
        return val_sequences
    else:
        return test_sequences

def action2verb(action):
    action2verb = {
        0: 0,
        1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1,
        9: 2, 10: 2, 11: 2, 12: 2, 13: 2, 13: 2, 14: 2, 15: 2, 16: 2,
        17: 3, 18: 3, 19: 3,
        20: 4, 21: 4, 22: 4,
        23: 5,
        24: 6, 25: 6, 26: 6, 27: 6,
        28: 7, 29: 7, 30: 7,
        31: 8, 32: 8,
        33: 9, 34: 9,
        35: 10,
        36: 11,
    }
    return action2verb[action]

# find the acton label of one clip in txt
def find_action_label(root, sample_short_path, vaild_start_frame):
    id = (6-len(str(vaild_start_frame))) * '0' + str(vaild_start_frame)
    with open(os.path.join(root, sample_short_path, 'cam4', 'action_label', id+'.txt'), 'r') as f:
        res = f.readline().strip()
    
    return int(res)

def read_split_df(root, split, with_label=True):
    if split == 'all':
        df = pd.concat([read_split(root, 'train', False), read_split(root, 'val', False), read_split(root, 'test', False)])
    elif split == 'train' or split == 'val' or split == 'test':
        path = os.path.join(root, 'label_split', 'action_'+split+'.txt')

        df = pd.read_csv(path, delimiter=' ', header=0)
        
        if split == 'train' and with_label:
            if not df[df['action_label'].isin([0])].empty:
                print(df[df['action_label'].isin([0])])
            df['verb_label'] = df.apply(lambda df: action2verb(df['action_label']), axis=1)
        elif split == 'val' and with_label:
            df['action_label'] = df.apply(lambda df: find_action_label(root, df['path'], df['start_act']), axis=1)
            df['verb_label'] = df.apply(lambda df: action2verb(df['action_label']), axis=1)

        df['valid_frame_len'] = df['end_act'] - df['start_act'] + 1
    else:
        raise NotImplementedError('data split only supports train/val/test/all')

    return df

# get data in single frame
def read_single_frame(path, type='hands'):
    if type=='hands':
        df = pd.read_csv(path, delimiter=' ', header=None).dropna(axis=1, how='any')

        assert df.shape == (1, 128)

        temp_tensor = torch.zeros((2, 21, 3)) # M, V, C

        for i in range(64):
            if i == 0:
                if df.iloc[0,i] == 0:
                    break
            else:
                temp_tensor[0][(i-1) // 3][(i-1) % 3] = df.iloc[0,i]

        for i in range(64, 128):
            if i == 64:
                if df.iloc[0,i] == 0:
                    break
            else:
                k = i - 64
                temp_tensor[1][(k-1) // 3][(k-1) % 3] = df.iloc[0,i]
        
        temp_tensor = torch.stack([temp_tensor[1, H2O_TO_A101, :], temp_tensor[0, H2O_TO_A101, :]], dim=0)
        temp_tensor = temp_tensor.view(42, 3)
    elif type== 'hands_mano':
        temp_tensor = torch.zeros(2, 61)
        with open(path, 'r') as f:
            raw = f.read().split(' ')
            raw_list = [float(x) for x in raw]
            if int(raw_list[0]) == 1:
                temp_tensor[1] = torch.tensor(raw_list[1:62])
            if int(raw_list[62]) == 1:
                temp_tensor[0] = torch.tensor(raw_list[63:])
        return temp_tensor

    elif type=='object':
        df = pd.read_csv(path, delimiter=' ', header=None).dropna(axis=1, how='any')

        assert df.shape == (1, 64)

        temp_tensor = torch.zeros((1, 21, 3)) # M, V, C

        for i in range(64):
            if i == 0:
                continue
            else:
                temp_tensor[0][(i-1) // 3][(i-1) % 3] = df.iloc[0,i]
    elif type=='cam':
        df = pd.read_csv(path, delimiter=' ', header=None).dropna(axis=1, how='any')
        assert df.shape == (1, 16)
        temp_tensor = torch.zeros((4, 4))
        for i in range(16):
            temp_tensor[i//4][i%4] = df.iloc[0, i]
    return temp_tensor # M, V, C

# get data of one sample
def read_sample(path, start=0, end=-1, type='hands'):
    frame_list = []

    if end == -1:
        end = len(os.listdir(path)) - 1
    for i in range(start, end+1):
        id = (6-len(str(i))) * '0' + str(i)
        frame_list.append(read_single_frame(os.path.join(path, id+'.txt'), type))
    
    sample = torch.stack(frame_list, dim=0)
    return sample # T, M, V, C

# main func
def get_H2O(root, split):
    tqdm_desc = 'Get H2O '+split+' set(s)'
    # test split has no label
    print(tqdm_desc)

    action_labels = {}
    verb_labels = {}
    df = read_split_df(root, split, with_label=True)
    for i in range(len(df)):
        path = df.at[i, 'path']
        st = df.at[i, 'start_act']
        end = df.at[i, 'end_act']
        if path not in action_labels:
            action_labels[path] = np.zeros(df.at[i, 'end_frame']+1, dtype=int)
            verb_labels[path] = np.zeros(df.at[i, 'end_frame']+1, dtype=int)
        if split != 'test':
            action_labels[path][st:end] = df.at[i, 'action_label']
            verb_labels[path][st:end] = df.at[i, 'verb_label']


    out_data = {}
    seq = read_split(split)
    for seq_name in seq:
        for dir_name in '01234567':
            seq_path = osp.join(root, seq_name, dir_name, 'cam4')
            if os.path.exists(seq_path):
                images = sorted([osp.join(seq_path, 'rgb', x) for x in os.listdir(osp.join(seq_path, 'rgb'))])
                seq_actions = action_labels[osp.join(seq_name, dir_name)]
                seq_verbs = verb_labels[osp.join(seq_name, dir_name)]
                hand_pose = read_sample(osp.join(seq_path, 'hand_pose'))
                cam_pose = read_sample(osp.join(seq_path, 'cam_pose'), type='cam')
                obj_pose = read_sample(osp.join(seq_path, 'obj_pose'), type='object')
                hand_pose_mano = read_sample(osp.join(seq_path, 'hand_pose_mano'), type='hands_mano')
                print(hand_pose_mano.shape)
                assert hand_pose.shape[0] == seq_actions.shape[0] and seq_actions.shape[0] == len(images), (hand_pose.shape, seq_actions.shape, len(images))
                # vis.export_simple_vis(hand_pose[:30, :, :].numpy(), "./exps/cam_pose.mp4")
                hand_pose_padded = torch.cat([hand_pose, torch.ones_like(hand_pose[...,0:1])], axis=-1).view(-1, 42, 1, 4)
                cam2world = cam_pose.view(-1, 1, 4, 4)
                world_coord_pose = torch.matmul(hand_pose_padded, cam2world[..., :3, :].transpose(-1, -2)).view(-1, 42, 3)
                # vis.export_simple_vis(world_coord_pose[:30, :, :].numpy(), "./exps/cam_pose2.mp4")
                print(world_coord_pose.shape)
                wolrd2cam = torch.linalg.inv(cam_pose.transpose(-1, -2)).transpose(-1, -2)
                print(cam_pose[0])
                # print(wolrd2cam[0])
                mano_r = hand_pose_mano[:, 0, ...]
                mano_l = hand_pose_mano[:, 1, ...]
                out_data[seq_path] = {
                    "world_coord": world_coord_pose.numpy(),
                    "cam_coord": hand_pose.numpy(),
                    "ego2world": cam_pose.numpy(),
                    "world2ego": wolrd2cam.numpy(),
                    "obj_pose": obj_pose.numpy(),
                    "action_labels": seq_actions,
                    "verb_labels": seq_verbs,
                    "images": images,
                    # MANO Parameters
                    "trans_l": mano_l[:,:3],
                    "rot_l": mano_l[:,3:6],
                    "pose_l": mano_l[:,6:51],
                    "shape_l": mano_l[:,51:61],
                    "trans_r": mano_r[:,:3],
                    "rot_r": mano_r[:,3:6],
                    "pose_r": mano_r[:,6:51],
                    "shape_r": mano_r[:,51:61],
                }

    out_p = f"./data/motion/h2o/{split}.npy"
    out_folder = osp.dirname(out_p)
    if not osp.exists(out_folder):
        os.makedirs(out_folder)
    print("Dumping data")
    np.save(out_p, out_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate H2O pth files")
    parser.add_argument('--root', type=str, help = 'Path to downloaded files.', default='./data/H2O')

    from IPython import get_ipython
    from ipdb.__main__ import _get_debugger_cls
    import ipdb
    debugger = _get_debugger_cls()
    shell = get_ipython()
    shell.debugger_history_file = '.pdbhistory'

    args = parser.parse_args()

    get_H2O(args.root, 'val')
    get_H2O(args.root, 'train')
    get_H2O(args.root, 'test')
