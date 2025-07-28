import sys

sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from modeling.vae.pointnet import (PointNetFeaturePropagation,
                                            PointNetSetAbstraction)
from .grabnet import ResBlock

class PointNetEncoder(nn.Module):

    def __init__(self,
                 hc,
                 in_feature):

        super(PointNetEncoder, self).__init__()
        self.hc = hc
        self.in_feature = in_feature + 3

        self.enc_sa1 = PointNetSetAbstraction(npoint=256, radius=0.1, nsample=32, in_channel=self.in_feature, mlp=[self.hc, self.hc*2], group_all=False)
        self.enc_sa2 = PointNetSetAbstraction(npoint=128, radius=0.15, nsample=64, in_channel=self.hc*2 + 3, mlp=[self.hc*2, self.hc*4], group_all=False)
        self.enc_sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=self.hc*4 + 3, mlp=[self.hc*4, self.hc*8], group_all=True)

    def forward(self, l0_xyz, l0_points):
        l1_xyz, l1_points = self.enc_sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.enc_sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.enc_sa3(l2_xyz, l2_points)
        x = l3_points.view(-1, self.hc*8)
        return x


class GrabPointnet(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(GrabPointnet, self).__init__()

        self.cfg = cfg
        self.latentD = cfg.latentD
        latentD = cfg.latentD
        pointnet_hc = cfg.pointnet_hc
        self.contact_cond = cfg.contact_cond
        in_bps = pointnet_hc * 8
        in_pose = 16*9+3
        n_neurons = pointnet_hc * 8
        if self.contact_cond:
            cond_dim = in_bps + n_neurons
        else:
            cond_dim = in_bps

        self.pointnet = PointNetEncoder(hc=cfg.pointnet_hc, in_feature=3)

        self.enc_bn1 = nn.BatchNorm1d(cond_dim + in_pose)
        self.enc_rb1 = ResBlock(cond_dim + in_pose , n_neurons)
        self.enc_rb2 = ResBlock(n_neurons + cond_dim + in_pose, n_neurons)

        self.enc_mu = nn.Linear(n_neurons, latentD)
        self.enc_var = nn.Linear(n_neurons, latentD)
        self.do = nn.Dropout(p=.1, inplace=False)

        dec_cond_dim = cond_dim 
        self.dec_bn1 = nn.BatchNorm1d(dec_cond_dim)  # normalize the bps_torch for object
        self.dec_rb1 = ResBlock(latentD + dec_cond_dim, n_neurons)
        self.dec_rb2 = ResBlock(n_neurons + latentD + dec_cond_dim, n_neurons)

        self.dec_pose = nn.Linear(n_neurons, 16 * 6)
        # self.dec_trans = nn.Linear(n_neurons, 21 * 3)
        if self.contact_cond:
            self.contact_emb = nn.Embedding(2, n_neurons)


    def encode(self, bps_object, trans_rhand, global_orient_rhand_rotmat, fpose_rhand_rotmat, contact_emb, **kwargs):
        
        bs = bps_object.shape[0]

        if self.contact_cond:
            X = torch.cat([bps_object, contact_emb, global_orient_rhand_rotmat.view(bs, -1), trans_rhand.view(bs, -1), fpose_rhand_rotmat.view(bs, -1)], dim=1)
        else:
            X = torch.cat([bps_object, global_orient_rhand_rotmat.view(bs, -1), trans_rhand.view(bs, -1), fpose_rhand_rotmat.view(bs, -1)], dim=1)

        X0 = self.enc_bn1(X)
        X  = self.enc_rb1(X0, True)
        X  = self.enc_rb2(torch.cat([X0, X], dim=1), True)

        return torch.distributions.normal.Normal(self.enc_mu(X), F.softplus(self.enc_var(X)))

    def decode(self, Zin, bps_object, trans_rhand, contact_emb):

        bs = Zin.shape[0]
        # o_bps = self.dec_bn1(bps_object)
        # o_bps = self.dec_bn1(torch.cat([bps_object, trans_rhand], dim=-1))
        
        if self.contact_cond:
            o_bps = self.dec_bn1(torch.cat([bps_object, contact_emb], dim=-1))
        else:
            o_bps = self.dec_bn1(bps_object)

        X0 = torch.cat([Zin, o_bps], dim=1)
        X = self.dec_rb1(X0, True)
        X = self.dec_rb2(torch.cat([X0, X], dim=1), True)

        pose = self.dec_pose(X)
        # trans = self.dec_trans(X)
        
        results = parms_decode(pose, trans_rhand)
        results['z'] = Zin

        return results

    def forward(self, verts_object, trans_rhand, global_orient_rhand_rotmat, fpose_rhand_rotmat, z_input=None, decode=False, encode=False, **kwargs):
        '''
        :param bps_object: bps_delta of object: Nxn_bpsx3
        :param delta_hand_mano: bps_delta of subject, e.g. hand: Nxn_bpsx3
        :param output_type: bps_delta of something, e.g. hand: Nxn_bpsx3
        :return:
        '''
        
        if self.contact_cond:
            has_contact = kwargs['has_contact']
            contact_emb = self.contact_emb(has_contact)
        else:
            contact_emb = None
        feat_obj = verts_object - trans_rhand[:, None, :]
        cond_object = self.pointnet(l0_xyz=verts_object.transpose(1, 2), l0_points=feat_obj.transpose(1, 2))
        results = {}
        if not decode:
            z = self.encode(cond_object, trans_rhand, global_orient_rhand_rotmat, fpose_rhand_rotmat, contact_emb=contact_emb)
            z_s = z.rsample()
            results.update({'mean': z.mean, 'std': z.scale})
        else:
            z_s = z_input
            
        if encode:
            return z
        else: 
            hand_parms = self.decode(z_s, cond_object, trans_rhand, contact_emb=contact_emb)            
            results.update(hand_parms)
            return results

    def sample_poses(self, verts_object, has_contact, trans_rhand, seed=None, **kwargs):
        if self.contact_cond:
            has_contact = kwargs['has_contact']
            contact_emb = self.contact_emb(has_contact)
        else:
            contact_emb = None
        feat_obj = verts_object - trans_rhand[:, None, :]
        cond_object = self.pointnet(l0_xyz=verts_object.transpose(1, 2), l0_points=feat_obj.transpose(1, 2))
        bs = cond_object.shape[0]
        dtype = cond_object.dtype
        device = cond_object.device
        np.random.seed(seed)
        self.eval()
        with torch.no_grad():
            Zgen = np.random.normal(0., 1., size=(bs, self.latentD))
            Zgen = torch.tensor(Zgen,dtype=dtype).to(device)

        return self.decode(Zgen, cond_object, trans_rhand, contact_emb=contact_emb)

def parms_decode(pose,trans):

    bs = trans.shape[0]

    pose_full = CRot2rotmat(pose)
    pose = pose_full.view([bs, 1, -1, 9])
    pose = rotmat2aa(pose).view(bs, -1)

    global_orient = pose[:, :3]
    hand_pose = pose[:, 3:]
    pose_full = pose_full.view([bs, -1, 3, 3])

    hand_parms = {'global_orient': global_orient, 'hand_pose': hand_pose, 'transl': trans, 'fullpose': pose_full}

    return hand_parms


def CRot2rotmat(pose):

    reshaped_input = pose.view(-1, 3, 2)

    b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

    dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
    b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=1)

    return torch.stack([b1, b2, b3], dim=-1)

def rotmat2aa(rotmat):
    '''
    :param rotmat: Nx1xnum_jointsx9
    :return: Nx1xnum_jointsx3
    '''
    batch_size = rotmat.size(0)
    homogen_matrot = F.pad(rotmat.view(-1, 3, 3), [0,1])
    pose = rotation_matrix_to_angle_axis(homogen_matrot).view(batch_size, 1, -1, 3).contiguous()
    return pose

def rotation_matrix_to_angle_axis(rotation_matrix):
    """Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = rotation_matrix_to_angle_axis(input)  # Nx3
    """
    # todo add check that matrix is a valid rotation matrix
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_angle_axis(quaternion)

def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * (~mask_d0_d1)
    mask_c2 = (~mask_d2) * mask_d0_nd1
    mask_c3 = (~mask_d2) * (~mask_d0_nd1)
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q

def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis