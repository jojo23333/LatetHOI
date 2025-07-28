import sys

sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from modeling.vae.pointnet import (PointNetFeaturePropagation,
                                            PointNetSetAbstraction)


class ResBlock(nn.Module):
    def __init__(self,
                 Fin,
                 Fout,
                 n_neurons=256):

        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout

        self.fc1 = nn.Linear(Fin, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)

        self.fc2 = nn.Linear(n_neurons, Fout)
        self.bn2 = nn.BatchNorm1d(Fout)

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))

        Xout = self.fc1(x)  # n_neurons
        Xout = self.bn1(Xout)
        Xout = self.ll(Xout)

        Xout = self.fc2(Xout)
        Xout = self.bn2(Xout)
        Xout = Xin + Xout

        if final_nl:
            return self.ll(Xout)
        return Xout

class PoseNet(nn.Module):
    def __init__(self, cfg, n_neurons=1024, obj_cond=1024, cond_dim=0, latentD=16, in_feature=15*6,  **kwargs):
        super(PoseNet, self).__init__()

        self.cfg = cfg
        ## condition features
        self.cond_dim = cond_dim

        self.enc_pose = ResBlock(in_feature, n_neurons)
        self.enc_bn1 = nn.BatchNorm1d(n_neurons)
        self.enc_rb1 = ResBlock(obj_cond + n_neurons + self.cond_dim, n_neurons)
        self.enc_rb2 = ResBlock(n_neurons, n_neurons)

        self.dec_rb1 = ResBlock(latentD + obj_cond, n_neurons)
        self.dec_rb2 = ResBlock(n_neurons + latentD + obj_cond + self.cond_dim, n_neurons)
        # self.dec_rb2_p = ResBlock(n_neurons + latentD + in_cond, n_neurons)

        self.dec_output = nn.Linear(n_neurons, in_feature)

    def enc(self, pose, cond_object, cond_feat):
        _, _, _, _, _, cond_object = cond_object

        # if self.cond_dim > 0:
        #     X = torch.cat([pose, cond_feat], dim=-1).float()

        X0 = self.enc_bn1(self.enc_pose(pose))
        X0 = torch.cat([X0, cond_object, cond_feat], dim=-1)

        X = self.enc_rb1(X0, True)
        X = self.enc_rb2(X, True)

        return X

    def dec(self, Z, cond_object, cond_feat):
        _, _, _, _, _, object_cond = cond_object
        X0 = torch.cat([Z, object_cond], dim=1).float()

        X = self.dec_rb1(X0, True)
        if self.cond_dim > 0:
            X = self.dec_rb2(torch.cat([X0, X, cond_feat], dim=1).float(), True)
        else:
            X = self.dec_rb2(torch.cat([X0, X], dim=1).float(), True)

        pred = self.dec_output(X)

        return pred

class PointNetEncoder(nn.Module):

    def __init__(self,
                 hc,
                 in_feature):

        super(PointNetEncoder, self).__init__()
        self.hc = hc
        self.in_feature = in_feature

        self.enc_sa1 = PointNetSetAbstraction(npoint=256, radius=0.2, nsample=32, in_channel=self.in_feature, mlp=[self.hc, self.hc*2], group_all=False)
        self.enc_sa2 = PointNetSetAbstraction(npoint=128, radius=0.25, nsample=64, in_channel=self.hc*2 + 3, mlp=[self.hc*2, self.hc*4], group_all=False)
        self.enc_sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=self.hc*4 + 3, mlp=[self.hc*4, self.hc*8], group_all=True)

    def forward(self, l0_xyz, l0_points):

        l1_xyz, l1_points = self.enc_sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.enc_sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.enc_sa3(l2_xyz, l2_points)
        x = l3_points.view(-1, self.hc*8)
        
        return l1_xyz, l1_points, l2_xyz, l2_points, l3_xyz, x

class ContactNet(nn.Module):
    def __init__(self, cfg, latentD=16, hc=64, object_feature=6, in_feature=778, **kwargs):
        super(ContactNet, self).__init__()
        self.latentD = latentD
        self.hc = hc
        self.object_feature  = object_feature
        self.in_feature = in_feature
        self.enc_pointnet = PointNetEncoder(self.hc, in_feature + object_feature + 1)
        

        # self.dec_fc1 = nn.Linear(self.latentD, self.hc*2)
        # self.dec_bn1 = nn.BatchNorm1d(self.hc*2)
        # self.dec_drop1 = nn.Dropout(0.1)
        # self.dec_fc2 = nn.Linear(self.hc*2, self.hc*4)
        # self.dec_bn2 = nn.BatchNorm1d(self.hc*4)
        # self.dec_drop2 = nn.Dropout(0.1)
        # self.dec_fc3 = nn.Linear(self.hc*4, self.hc*8)
        # self.dec_bn3 = nn.BatchNorm1d(self.hc*8)
        # self.dec_drop3 = nn.Dropout(0.1)

        self.dec_fc4 = nn.Linear(self.hc*8+self.latentD+self.hc, self.hc*8)
        self.dec_bn4 = nn.BatchNorm1d(self.hc*8)
        self.dec_drop4 = nn.Dropout(0.1)

        self.dec_fp3 = PointNetFeaturePropagation(in_channel=self.hc*8+self.hc*4, mlp=[self.hc*8, self.hc*4])
        self.dec_fp2 = PointNetFeaturePropagation(in_channel=self.hc*4+self.hc*2, mlp=[self.hc*4, self.hc*2])
        self.dec_fp1 = PointNetFeaturePropagation(in_channel=self.hc*2+self.object_feature, mlp=[self.hc*2, self.hc*2])

        self.dec_conv1 = nn.Conv1d(self.hc*2, self.hc*2, 1)
        self.dec_conv_bn1 = nn.BatchNorm1d(self.hc*2)
        self.dec_conv_drop1 = nn.Dropout(0.1)
        self.dec_conv2 = nn.Conv1d(self.hc*2, self.in_feature, 1)

        self.contactness = nn.Conv1d(self.hc*2, 1, 1)

    def enc(self, contacts_object, verts_object, feat_object, contactness=None):
        l0_xyz = verts_object[:, :3, :]
        l0_points = torch.cat([feat_object, contacts_object], 1) if feat_object is not None else contacts_object
        l0_points = torch.cat([l0_points, contactness], 1) if contactness is not None else l0_points
        _, _, _, _, _, x = self.enc_pointnet(l0_xyz, l0_points)
        return x  

    def dec(self, z, verts_object, feat_object, cond_object, cond_feat):
        l0_xyz = verts_object[:, :3, :]
        l0_points = feat_object

        l1_xyz, l1_points, l2_xyz, l2_points, l3_xyz, l3_points = cond_object

        l3_points = torch.cat([l3_points, z, cond_feat], 1)
        l3_points = self.dec_drop4(F.relu(self.dec_bn4(self.dec_fc4(l3_points)), inplace=True))
        l3_points = l3_points.view(l3_points.size()[0], l3_points.size()[1], 1)

        l2_points = self.dec_fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.dec_fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        if l0_points is None:
            l0_points = self.dec_fp1(l0_xyz, l1_xyz, l0_xyz, l1_points)
        else:
            l0_points = self.dec_fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz,l0_points],1), l1_points)
        feat =  F.relu(self.dec_conv_bn1(self.dec_conv1(l0_points)), inplace=True)
        x = self.dec_conv_drop1(feat)
        x = self.dec_conv2(x)

        contact_logits = self.contactness(feat)
        return x, contact_logits


class HOIGraspNet(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(HOIGraspNet, self).__init__()

        self.cfg = cfg
        self.latentD = cfg.latentD
        self.in_feature_list = {}
        # self.in_feature_list['joints'] = 778
        self.in_feature_list['joints'] = 21

        self.in_feature = self.in_feature_list[cfg.data_representation]

        self.contact_net = ContactNet(cfg, latentD=cfg.latentD, hc=cfg.pointnet_hc, object_feature=cfg.obj_feature, in_feature=self.in_feature)

        self.posenet = PoseNet(cfg, n_neurons=cfg.n_neurons, latentD=cfg.latentD, in_feature=15*6, obj_cond=cfg.pointnet_hc*8, cond_dim=cfg.pointnet_hc)
        self.pointnet = PointNetEncoder(hc=cfg.pointnet_hc, in_feature=cfg.obj_feature)
        # encoder fusion
        self.enc_fusion = ResBlock(2*self.cfg.pointnet_hc*8+cfg.pointnet_hc+cfg.n_neurons, cfg.n_neurons)
        # TODO: Consider whether need translation z axis condition here
        # self.enc_trans = nn.Linear(1, cfg.pointnet_hc)
        self.enc_cond = nn.Embedding(2, cfg.pointnet_hc)

        self.enc_mu = nn.Linear(cfg.n_neurons, cfg.latentD)
        self.enc_var = nn.Linear(cfg.n_neurons, cfg.latentD)

    def encode(self, dist_matrix, joint_rot, verts_object, feat_object, cond_obj, cond_feat, contactness=None):
        # contact branch
        _, _, _, _, _, obj_token = cond_obj
        # hand_object_feat = torch.cat([dist_matrix, normal_obj_rot], 1)
        contact_feat = self.contact_net.enc(dist_matrix, verts_object, feat_object, contactness)  # [B, hc*8]
        pose_feat = self.posenet.enc(joint_rot, cond_obj, cond_feat)  # [B, n_neurons]
        # fusion
        X = torch.cat([contact_feat, obj_token, cond_feat, pose_feat], dim=-1)
        X = self.enc_fusion(X, True)
        
        mu = self.enc_mu(X)
        var = self.enc_var(X)

        return mu, var, torch.distributions.normal.Normal(self.enc_mu(X), F.softplus(self.enc_var(X)))

    def decode(self, Z, verts_object, feat_object, cond_object, cond_feat):
        # contact branch
        dist_matrix, contact_logits = self.contact_net.dec(Z, verts_object, feat_object, cond_object, cond_feat)
        joint_rot = self.posenet.dec(Z, cond_object, cond_feat)
        # normal_object = 2*F.sigmoid(x[:, -3:])-1
        contact_logits = F.sigmoid(contact_logits)
        return dist_matrix, joint_rot, contact_logits

    def forward(self, joint_rot, dist_matrix, verts_object, feat_object, obj_trans, is_rhand, contactness=False, sample=False, encode_only=False, **kwargs):
        if sample:
            return self.sample(verts_object, feat_object, obj_trans, is_rhand, **kwargs)

        cond_object = self.pointnet(l0_xyz=verts_object, l0_points=feat_object)
        # TODO: Consider whether need translation z axis condition here
        cond_feat = self.enc_cond(is_rhand) #+ self.enc_trans(obj_trans[:, 2:])
        mu, var, z = self.encode(dist_matrix, joint_rot, verts_object, feat_object, cond_object, cond_feat, contactness)
        if encode_only:
            return {"mean": mu, "std": var, "object_code": cond_object}
        z_s = z.rsample()

        dist_matrix, joint_rot, contact_logits = self.decode(z_s, verts_object, feat_object, cond_object, cond_feat)

        results = {"dist_matrix": dist_matrix, 'joint_rot': joint_rot, 'contactness': contact_logits,  'object_code': cond_object[-1], 'mean': z.mean, 'std': z.scale}

        return results

    def sample(self, verts_object, feat_object, obj_trans, is_rhand, seed=None, **kwargs):
        bs = verts_object.shape[0]
        if seed is not None:
            np.random.seed(seed)
        dtype = verts_object.dtype
        device = verts_object.device
        self.eval()
        with torch.no_grad():
            Zgen = np.random.normal(0., 1., size=(bs, self.latentD))
            Zgen = torch.tensor(Zgen,dtype=dtype).to(device)

        cond_object = self.pointnet(l0_xyz=verts_object, l0_points=feat_object)
        cond_feat = self.enc_cond(is_rhand) # + self.enc_trans(obj_trans[:, 2:]) 
        dist_matrix, joint_rot, contact_logits = self.decode(Zgen, verts_object, feat_object, cond_object, cond_feat)
        
        return dist_matrix, joint_rot, contact_logits
