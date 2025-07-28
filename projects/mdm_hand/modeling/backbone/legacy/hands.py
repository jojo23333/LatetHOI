import torch


def xyz_to_wrist_xyz(batch_pose, scale=10):
    right_hand = batch_pose[:, :, :21, :]
    left_hand = batch_pose[:, :, 21:, :]
    relative_right_hand = right_hand[..., :20, :] - right_hand[:, :, 20:21, :]
    relative_left_hand = left_hand[..., :20, :] - left_hand[:, :, 20:21, :]

    mean_wrist_f0 = (left_hand[:, 0:1, 20:21, :] + right_hand[:, 0:1, 20:21, :]) / 2
    right_wrist = right_hand[:, :, 20:21, :] - mean_wrist_f0
    left_wrist = left_hand[:, :, 20:21, :] - mean_wrist_f0
    
    result = torch.cat([relative_right_hand, right_wrist, relative_left_hand, left_wrist],
                       dim=2)
    return result / scale

def get_hand_pose_condition(batch_pose, condition=None):
    if condition == 'wrist':
        right_hand = batch_pose[:, :, 20, :]
        left_hand = batch_pose[:, :, 41, :]
        right_motion = right_hand[:, 1:, :] - right_hand[:, :-1, :]
        right_motion = torch.cat([right_motion, right_motion[:, -1:, :]], dim=1)
        left_motion = left_hand[:, 1:, :] - left_hand[:, :-1, :]
        left_motion = torch.cat([left_motion, left_motion[:, -1:, :]], dim=1)
        hand_dist = right_hand - left_hand
        wrist_condition_vec = torch.cat([right_motion, left_motion, hand_dist], dim=-1)
        return wrist_condition_vec
    return None

# TODO fix this
def wrist_xyz_to_xyz(pred_pose, pose_0=None, scale=10):
    if pose_0 is None:
        pose_0 = torch.zeros_like(pred_pose[:, 0, ...])
        pose_0[:, 1:21, 1] = 1
        pose_0[:, 1:21, 1] = -1
    # import ipdb; ipdb.set_trace()
    relative_right_hand = pred_pose[:, :, :20, :]
    relative_left_hand = pred_pose[:, :, 21:41, :]

    right_wrist = pred_pose[:, :, 20:21, :]
    left_wrist = pred_pose[:, :, 41:42, :]

    # left_motion = pred_pose[:, 0:-1, 0:1, :]
    # right_motion = pred_pose[:, 0:-1, 21:22, :]
    # left_wrist = torch.cat([pose_0[:, None, 0:1, :], left_motion], dim=1).cumsum(dim=1)
    # right_wrist = torch.cat([pose_0[:, None, 21:22, :], right_motion], dim=1).cumsum(dim=1)

    right_hand = relative_right_hand + right_wrist
    left_hand = relative_left_hand + left_wrist
    return torch.cat([right_hand, right_wrist, left_hand, left_wrist], dim=2) * scale

    relative_left_hand = left_hand[..., 1:, :] + left_hand[:, :, 0:1, :]
    relative_right_hand = right_hand[..., 1:, :] - right_hand[:, :, 0:1, :]
    left_motion = left_hand[:, 1:, 0:1, :] - left_hand[:, :-1, 0:1, :]
    left_motion = torch.cat([left_motion, left_motion[:, -1:, ...]], dim=1)
    right_motion = right_hand[:, 1:, 0:1, :] - right_hand[:, :-1, 0:1, :]
    right_motion = torch.cat([right_motion, right_motion[:, -1:, ...]], dim=1)

def xyz_relative_xyz(batch_pose, scale=10):
    mid_ref = (batch_pose[:, :1, 20:21, :] + batch_pose[:, :1, 41:42, :]) / 2
    return (batch_pose - mid_ref) / scale


def pose_l2_dist(diff_pose, gt_pose, result, valid=None, prefix='', detailed=False):
    pred_left = diff_pose[:, :, 21:, :]
    pred_right = diff_pose[:, :, :21, :]
    pred_r_left = pred_left - pred_left[:, :, 20:21, :]
    pred_r_right = pred_right - pred_right[:, :, 20:21, :]
    
    gt_left = gt_pose[:, :, 21:, :]
    gt_right = gt_pose[:, :, :21, :]
    gt_r_left = gt_left - gt_left[:, :, 20:21, :]
    gt_r_right = gt_right - gt_right[:, :, 20:21, :]
    
    l2_left = ((gt_r_left - pred_r_left) ** 2).sum(dim=-1).sqrt()
    l2_right = ((gt_r_right - pred_r_right) ** 2).sum(dim=-1).sqrt()
    l2_wrist_left = ((pred_left[:,:,20:21,:] - gt_left[:,:,20:21,:]) ** 2).sum(dim=-1).sqrt()
    l2_wrist_right = ((pred_right[:,:,20:21,:] - gt_right[:,:,20:21,:]) ** 2).sum(dim=-1).sqrt()
    if valid is not None:
        l2_left = l2_left[valid[:, :, 21:]]
        l2_right = l2_right[valid[:, :, :21]]
        l2_wrist_left = l2_wrist_left[valid[:, :, 41:]]
        l2_wrist_right = l2_wrist_right[valid[:, :, 20:21]]
    
    if detailed:
        result[f"{prefix}_l2_dist_ljoint"] = l2_left.mean().cpu().item()
        result[f"{prefix}_l2_dist_lwrist"] = l2_wrist_left.mean().cpu().item()
        result[f"{prefix}_l2_dist_rjoint"] = l2_right.mean().cpu().item()
        result[f"{prefix}_l2_dist_rwrist"] = l2_wrist_right.mean().cpu().item()
    result[f"{prefix}_l2_dist_joint"] = (l2_left.mean() + l2_right.mean()).cpu().item() / 2
    result[f"{prefix}_l2_dist_wrist"] = (l2_wrist_left.mean() + l2_wrist_right.mean()).cpu().item() / 2
    return result