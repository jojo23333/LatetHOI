import torch
import os
from tqdm import tqdm
import pandas as pd
import argparse

train_sequences = ['subject1/h1', 'subject1/h2', 'subject1/k1', 'subject1/k2', 'subject1/o1', 'subject1/o2', 'subject2/h1', 'subject2/h2', 'subject2/k1', 'subject2/k2', 'subject2/o1', 'subject2/o2', 'subject3/h1', 'subject3/h2', 'subject3/k1']
val_sequences = ['subject3/k2', 'subject3/o1', 'subject3/o2']
test_sequences = ['subject4/h1', 'subject4/h2', 'subject4/k1', 'subject4/k2', 'subject4/o1', 'subject4/o2'] 


# verb label index -> action label index
def verb2action(verb):
    verb2action = {
        0: [0], # background (no verb)
        1: [1, 2, 3, 4, 5, 6, 7, 8], # grab
        2: [9, 10, 11, 12, 13, 14, 15, 16], # place
        3: [17, 18, 19], # open
        4: [20, 21, 22], # close
        5: [23], # pour
        6: [24, 25, 26, 27], # take out
        7: [28, 29, 30], # put in
        8: [31, 32], # apply
        9: [33, 34], # read
        10: [35], # spray
        11: [36], # squeeze
    }
    return verb2action[verb]

# action label index -> verb label index
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

# read one split
def read_split(root, split, with_label=True):
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

    elif type=='object':
        df = pd.read_csv(path, delimiter=' ', header=None).dropna(axis=1, how='any')

        assert df.shape == (1, 64)

        temp_tensor = torch.zeros((1, 21, 3)) # M, V, C

        for i in range(64):
            if i == 0:
                continue
            else:
                temp_tensor[0][(i-1) // 3][(i-1) % 3] = df.iloc[0,i]

    return temp_tensor # M, V, C

# pad tensor according to args.frames
def pad_tensor(sample_tensor, max_frame_num=120):
    if sample_tensor.size(0) < max_frame_num:
        zero_tensor = torch.zeros((max_frame_num-sample_tensor.size(0), sample_tensor.size(1), sample_tensor.size(2), sample_tensor.size(3)))
        sample_tensor = torch.cat([sample_tensor, zero_tensor], dim=0)

    if sample_tensor.size(0) > max_frame_num:
        st = (sample_tensor.size(0)-max_frame_num)//2
        sample_tensor = sample_tensor[st:st+max_frame_num,:,:,:]
    
    return sample_tensor

# get data of one sample
def read_sample(path, start, end, type='hands'):
    frame_list = []

    for i in range(start, end+1):
        id = (6-len(str(i))) * '0' + str(i)
        frame_list.append(read_single_frame(os.path.join(path, id+'.txt'), type))
    
    sample = torch.stack(frame_list, dim=0)

    return sample # T, M, V, C

# main func
def get_H2O(root, split):
    tqdm_desc = 'Get H2O '+split+' set(s)'
    if split == 'test' or split == 'all':
        # test split has no label
        df = read_split(root, split, False)

        sample_list = []

        with tqdm(total=len(df), desc=tqdm_desc, ncols=100) as pbar:
            for i in range(len(df)):
                p_hands = os.path.join(root, df.at[i, 'path'], 'cam4', 'hand_pose')
                p_obj = os.path.join(root, df.at[i, 'path'], 'cam4', 'obj_pose')
                st = df.at[i, 'start_act']
                end = df.at[i, 'end_act']
                temp = torch.cat([read_sample(p_hands, st, end, 'hands'), read_sample(p_obj, st, end, 'object')], dim=1)
                sample_list.append(pad_tensor(temp, frame_num))
                pbar.update(1)

        # N, T, M, V, C
        data = torch.stack(sample_list, dim=0)
        ground_truth = None

        # N, T, M, V, C ->  N, C, T, V, M
        data = data.permute(0, 4, 1, 3, 2)

    elif split == 'train' or split == 'val':
        df = read_split(root, split, True)

        sample_list = []
        label_list = []

        with tqdm(total=len(df), desc=tqdm_desc, ncols=100) as pbar:
            for i in range(len(df)):
                p_hands = os.path.join(root, df.at[i, 'path'], 'cam4', 'hand_pose')
                p_obj = os.path.join(root, df.at[i, 'path'], 'cam4', 'obj_pose')
                st = df.at[i, 'start_act']
                end = df.at[i, 'end_act']
                temp = torch.cat([read_sample(p_hands, st, end, 'hands'), read_sample(p_obj, st, end, 'object')], dim=1)
                sample_list.append(pad_tensor(temp))
                label_list.append(df.at[i, 'action_label'])
                pbar.update(1)

        # N, T, M, V, C
        data = torch.stack(sample_list, dim=0)
        # N
        ground_truth = torch.tensor(label_list)

        # N, T, M, V, C ->  N, C, T, V, M
        data = data.permute(0, 4, 1, 3, 2)

    else:
        raise NotImplementedError('data split only supports train/val/test/all')

    # N, C, T, V, M
    return data, ground_truth


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate H2O pth files")
    parser.add_argument('--root', type=str, help = 'Path to downloaded files.', default='downloads')
    parser.add_argument('--dest', type=str, help = 'Destination path to save pth files.', default='h2o_pth')
    parser.add_argument('--frames', type=int, help = 'Input frame numbers', default=120)

    args = parser.parse_args()

    a, b = get_H2O(args.root, 'val', args.frames)
    print(a.shape)
    print(b.shape)

    torch.save(a.clone(), os.path.join(args.dest, 'val', 'data.pth'))
    torch.save(b.clone(), os.path.join(args.dest, 'val', 'gt.pth'))

    c, d = get_H2O(args.root, 'train', args.frames)
    print(c.shape)
    print(d.shape)

    torch.save(c.clone(), os.path.join(args.dest, 'train', 'data.pth'))
    torch.save(d.clone(), os.path.join(args.dest, 'train', 'gt.pth'))

    e, _ = get_H2O(args.root, 'test', args.frames)
    print(e.shape)

    torch.save(e.clone(), os.path.join(args.dest, 'test', 'data.pth'))
