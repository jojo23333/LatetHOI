import os
import sys
import smplx

import chamfer_distance as chd
import numpy as np
import torch
import logging
from utils.rotation_conversions import rotation_6d_to_matrix, rotation_6d_to_axis_angle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = './data/body_models'

logger = logging.getLogger('d2')

def point2point_signed(
        x,
        y,
        x_normals=None,
        y_normals=None,
        return_vector=False,
):
    """
    signed distance between two pointclouds

    Args:
        x: FloatTensor of shape (N, P1, D) representing a batch of point clouds
            with P1 points in each batch element, batch size N and feature
            dimension D.
        y: FloatTensor of shape (N, P2, D) representing a batch of point clouds
            with P2 points in each batch element, batch size N and feature
            dimension D.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).

    Returns:

        - y2x_signed: Torch.Tensor
            the sign distance from y to x
        - y2x_signed: Torch.Tensor
            the sign distance from y to x
        - yidx_near: Torch.tensor
            the indices of x vertices closest to y

    """


    N, P1, D = x.shape
    P2 = y.shape[1]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")

    ch_dist = chd.ChamferDistance()
    x_near, y_near, xidx_near, yidx_near = ch_dist(x,y)
    # x_near, y_near, xidx_near, yidx_near = chd.ChamferDistance(x,y)

    xidx_near_expanded = xidx_near.view(N, P1, 1).expand(N, P1, D).to(torch.long)
    x_near = y.gather(1, xidx_near_expanded)

    yidx_near_expanded = yidx_near.view(N, P2, 1).expand(N, P2, D).to(torch.long)
    y_near = x.gather(1, yidx_near_expanded)

    x2y = x - x_near  # y point to x
    y2x = y - y_near  # x point to y

    if x_normals is not None:
        y_nn = x_normals.gather(1, yidx_near_expanded)
        in_out = torch.bmm(y_nn.view(-1, 1, 3), y2x.view(-1, 3, 1)).view(N, -1).sign()
        y2x_signed = y2x.norm(dim=2) * in_out

    else:
        y2x_signed = y2x.norm(dim=2)

    if y_normals is not None:
        x_nn = y_normals.gather(1, xidx_near_expanded)
        in_out_x = torch.bmm(x_nn.view(-1, 1, 3), x2y.view(-1, 3, 1)).view(N, -1).sign()
        x2y_signed = x2y.norm(dim=2) * in_out_x
    else:
        x2y_signed = x2y.norm(dim=2)

    if not return_vector:
        return y2x_signed, x2y_signed, yidx_near, xidx_near
    else:
        return y2x_signed, x2y_signed, yidx_near, xidx_near, y2x, x2y


class RotMatSolver:
    def __init__(self, normal, normal_rot) -> None:
        self.normal = normal
        self.normal_rot = normal_rot
        self.rot_mat = torch.eye(3, device=normal.device, dtype=normal.dtype).unsqueeze(0).repeat(len(normal), 1, 1)

    def solve_lsq(self):
        A = self.normal_rot
        B = self.normal
        rot_mat = torch.linalg.lstsq(A, B).solution
        return rot_mat
    
class HandSolver:
    def __init__(self, betas, is_rhand):
        self.hand = smplx.create(
            model_path=MODEL_DIR,
            model_type='mano',
            is_rhand=is_rhand,
            use_pca=False,
            flat_hand_mean=True
        )
        self.betas = betas.to(torch.float32)
        self.is_rhand = is_rhand
        for param in self.hand.parameters():
            param.requires_grad = False
        
        self.hand = self.hand.to(betas.device)
        ###################################################################################################################
        # With smplx definition, translation = root joint location + an offset effected by "beta" and "is_rhand"
        # Get that offset first by dry run the with flat hand
        self.offset = self.hand(
                betas=self.betas,
                global_orient=torch.eye(3).view(1, 1, 3, 3).to(self.betas),
                hand_pose=torch.eye(3).view(1, 1, 3, 3).repeat(1, 15, 1, 1).to(self.betas),
                transl=torch.zeros(1, 3).to(self.betas),
                use_rot_matrix=True
        ).joints[0, 0]
        ###################################################################################################################
    
    def gloabl_orient_init(self, joint, hand_pose):
        def get_R(joints):
            import torch.nn.functional as F
            root = joints[:, 0, :]
            if joints.shape[1] == 21:
                index = joints[:, 1, :]
                ring = joints[:, 10, :]
            else:
                index = joints[:, 320, :]
                ring = joints[:, 554, :]
            x = F.normalize(index - root, dim=-1)
            y = F.normalize(torch.cross(index-root, index-ring), dim=-1)
            z = torch.cross(x, y)
            return torch.stack([x, y, z], dim=-1)
        
        with torch.no_grad():
            bsz = joint.shape[0]
            global_rot = torch.eye(
                3, 
                device=joint.device,
                dtype=joint.dtype
            ).unsqueeze(0).repeat(bsz, 1, 1).view(bsz, 1, 3, 3)
            
            out = self.hand(
                betas=self.betas,
                global_orient=global_rot.to(self.betas),
                hand_pose=hand_pose.to(self.betas),
                transl=torch.zeros(bsz, 3).to(self.betas),
                use_rot_matrix=True
            )
        R1 = get_R(joint).to(torch.float64)
        R2 = get_R(out.joints).to(torch.float64)
        R = R1 @ R2.transpose(1, 2)
        return R.to(torch.float32)
    
    def solve(self, joint_rot, joint, global_rot=None):
        '''
            Use 'Projected' Gradient Descent to solve Global Rotation of Hand Pose
        '''
        bsz, num_kp = joint.shape[0], joint.shape[1]
        hand_pose = rotation_6d_to_matrix(joint_rot.view(bsz, 15, 6))#.view(bsz, 45)
        if global_rot is not None:
            rot_init = global_rot
        else:
            rot_init = self.gloabl_orient_init(joint, hand_pose)[..., :2, :].reshape(bsz, 6)
        global_rot = torch.tensor(rot_init, requires_grad=True)
        global_rot = rot_init.clone().detach().requires_grad_(True)
        transl = (joint[:, 0] - self.offset).clone().detach().requires_grad_(True)
        
        optimizer = torch.optim.Adam([{'params': global_rot, 'lr': 0.1},#) 
                                     {'params': transl, 'lr': 0.01}])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        losses = []
        for i in range(200):
            global_rot_matrix = rotation_6d_to_matrix(global_rot).view(bsz, 1, 3, 3)
            out = self.hand(
                betas=self.betas.to(hand_pose),
                global_orient=global_rot_matrix.to(hand_pose),
                hand_pose=hand_pose,
                transl=transl,
                use_rot_matrix=True
            )
            if num_kp == 21:
                loss = (out.joints - joint).abs().mean()
            else:
                loss = (out.vertices - joint).abs().mean()
            loss.backward()
            if loss.item() < 1e-5:
                break
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            scheduler.step()
            # with torch.no_grad():
            #     global_rot_matrix = rotation_6d_to_matrix(global_rot_6d).view(bsz, 1, 3, 3)
            #     global_rot_6d.data = global_rot_matrix[:, :, :2, :].view(bsz, 6).data
            # if optimize  < 1e-5 for 5 epoch terminate
            if i > 50 and np.max(losses[-50:]) - loss.item() < 1e-4:
                break
            # print(i, loss, optimizer.param_groups[0]['lr'])
        logger.info(f'Hand Solver: {i} iterations, loss: {loss.item()}')
        return out.vertices, global_rot


class DistMatrixSolver:
    def __init__(self, P1, M, contact_mask=None ,initial_lr=0.1, final_lr=0.01, epochs=500):
        # TODO mask here has a issue of not being able to extend for batch size > 1
        if contact_mask is None:
            self.P1 = P1.clone().detach()
            self.M = M.clone().detach()
        else:
            mask = contact_mask[0, :].squeeze() > 0.5
            self.P1 = P1[:, mask].clone().detach()
            self.M = M[:, mask].clone().detach()
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.epochs = epochs
        bsz, n1, n2 = self.M.shape
        P2_init = P1.mean(dim=1).unsqueeze(1).repeat(1, n2, 1)
        P2_init = P2_init + torch.randn_like(P2_init) * 0.01
        self.P2 = torch.tensor(P2_init.detach(), requires_grad=True, device=self.P1.device)  # Initial guess

    def solve_lsq(self):
        bsz, n1, n2 = self.M.shape
        right_side = (self.P1 ** 2).sum(dim=-1).view(bsz, n1, 1).repeat(1, 1, n2) - self.M ** 2
        right_side = right_side.permute(0, 2, 1).reshape(bsz*n2, n1)
        left_size = 2 * self.P1.view(bsz, 1, n1, 3).repeat(1, n2, 1, 1).reshape(bsz*n2, n1, 3)
        
        B = right_side[:, 1:] - right_side[:, :-1]
        A = left_size[:, 1:] - left_size[:, :-1]
        p2 = torch.linalg.lstsq(A, B).solution
        return p2.reshape(bsz, n2, 3)

    def train(self):
        optimizer = torch.optim.Adam([self.P2], lr=self.initial_lr)
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            loss = self.compute_loss()
            print(loss, self.P2[0, 0, :].detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            self.adjust_learning_rate(optimizer, epoch)
        return self.P2.detach()

    def compute_loss(self):
        distances = self.P1.unsqueeze(2) - self.P2.unsqueeze(1)
        distances = torch.norm(distances, p=2, dim=-1)
        loss = torch.mean((distances - self.M)**2, dim=1).sum()
        return loss

    def adjust_learning_rate(self, optimizer, epoch):
        lr = self.initial_lr - (self.initial_lr - self.final_lr) * (epoch / self.epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def solve(self):
        return self.train()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.trace_func is not None:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop

def save_ckp(state, checkpoint_dir):
    f_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
    torch.save(state, f_path)

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']


def get_forward_joint(joint_start):
    """ Joint_start: [B, N, 3] in xyz """
    x_axis = joint_start[:, 2, :] - joint_start[:, 1, :]
    x_axis[:, -1] = 0
    x_axis = x_axis / torch.norm(x_axis, dim=-1).unsqueeze(1)
    z_axis = torch.tensor([0, 0, 1]).float().unsqueeze(0).repeat(len(x_axis), 1).to(device)
    y_axis = torch.cross(z_axis, x_axis)
    y_axis = y_axis / torch.norm(y_axis, dim=-1).unsqueeze(1)
    transf_rotmat = torch.stack([x_axis, y_axis, z_axis], dim=1)
    return y_axis, transf_rotmat

def prepare_traj_input(joint_start, joint_end, traj_Xmean, traj_Xstd):
    """ Joints: [B, N, 3] in xyz """
    B, N, _ = joint_start.shape
    T = 62
    joint_sr_input_unnormed = torch.ones(B, 4, T)  # [B, xyr, T]
    y_axis, transf_rotmat = get_forward_joint(joint_start)
    joint_start_new = joint_start.clone()
    joint_end_new = joint_end.clone()  # to check whether original joints change or not
    joint_start_new = torch.matmul(joint_start - joint_start[:, 0:1], transf_rotmat)
    joint_end_new = torch.matmul(joint_end - joint_start[:, 0:1], transf_rotmat)

    # start_forward, _ = get_forward_joint(joint_start_new)
    start_forward = torch.tensor([0, 1, 0]).unsqueeze(0)
    end_forward, _ = get_forward_joint(joint_end_new)

    joint_sr_input_unnormed[:, :2, 0] = joint_start_new[:, 0, :2]  # xy
    joint_sr_input_unnormed[:, :2, -2] = joint_end_new[:, 0, :2]   # xy
    joint_sr_input_unnormed[:, 2:, 0] = start_forward[:, :2]  # r
    joint_sr_input_unnormed[:, 2:, -2] = end_forward[:, :2]  # r

    # normalize
    traj_mean = traj_Xmean.unsqueeze(2).cpu()
    traj_std = traj_Xstd.unsqueeze(2).cpu()

    # linear interpolation
    joint_sr_input_normed = (joint_sr_input_unnormed - traj_mean) / traj_std
    for t in range(joint_sr_input_normed.size(-1)):
        joint_sr_input_normed[:, :, t] = joint_sr_input_normed[:, :, 0] + (joint_sr_input_normed[:, :, -2] - joint_sr_input_normed[:, :, 0])*t/(joint_sr_input_normed.size(-1)-2)
        joint_sr_input_normed[:, -2:, t] = joint_sr_input_normed[:, -2:, t] / torch.norm(joint_sr_input_normed[:, -2:, t], dim=1).unsqueeze(1)

    for t in range(joint_sr_input_unnormed.size(-1)):
        joint_sr_input_unnormed[:, :, t] = joint_sr_input_unnormed[:, :, 0] + (joint_sr_input_unnormed[:, :, -2] - joint_sr_input_unnormed[:, :, 0])*t/(joint_sr_input_unnormed.size(-1)-2)
        joint_sr_input_unnormed[:, -2:, t] = joint_sr_input_unnormed[:, -2:, t] / torch.norm(joint_sr_input_unnormed[:, -2:, t], dim=1).unsqueeze(1)

    return joint_sr_input_normed.float().to(device), joint_sr_input_unnormed.float().to(device), transf_rotmat, joint_start_new, joint_end_new
