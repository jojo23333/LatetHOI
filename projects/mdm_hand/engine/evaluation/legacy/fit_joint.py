import torch
import copy
import numpy as np
from tqdm import tqdm
from smplx import MANO

class FitJoint:
    def __init__(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        MANO_MODEL_DIR = './data/ARCTIC/body_models/mano'
        self.right_mano_layer = MANO(
            MANO_MODEL_DIR,
            create_transl=True,
            use_pca=False,
            flat_hand_mean=True,
            is_rhand=True,
        ).to(self.device)
        self.left_mano_layer = MANO(
            MANO_MODEL_DIR,
            create_transl=True,
            use_pca=False,
            flat_hand_mean=True,
            is_rhand=False,
        ).to(self.device)

    def optimize_pose(
        self,
        mano_layer,
        init_pose,
        init_trans,
        joints_target,
        beta,
        optimize_trans=True
    ):
        trans = copy.deepcopy(init_trans)
        pose = copy.deepcopy(init_pose)
        if isinstance(pose, np.ndarray):
            trans = torch.from_numpy(trans).to(self.device).to(torch.float32)
            pose = torch.from_numpy(pose).to(self.device).to(torch.float32)
            joints_target = torch.from_numpy(joints_target).to(self.device).to(torch.float32)
            beta = torch.from_numpy(beta).to(self.device).to(torch.float32)
        
        nf = init_pose.shape[0]
        if beta.ndim == 1:
            beta = beta.unsqueeze(0).repeat(nf, 1)
        if optimize_trans:
            trans.requires_grad_(True)
        pose.requires_grad_(True)
        opt = torch.optim.Adam([trans, pose], lr=0.01)

        # Optimization
        for i in tqdm(range(1000)):
            opt.zero_grad()
            mano_output = mano_layer(
                beta,
                global_orient=pose[:, :3],
                hand_pose=pose[:, 3:]
            )
            joints = mano_output.joints - mano_output.joints[:, :1] + trans.unsqueeze(1)

            l2_loss = torch.sum(torch.square(joints - joints_target), dim=-1).mean()
            pose_smooth = torch.sum(torch.square(pose[:-1] - pose[1:]), dim=-1).mean()
            wrist_smooth = torch.sum(torch.square(trans[:-1] - trans[1:]), dim=-1).mean()
            loss = l2_loss + pose_smooth*0.001 + wrist_smooth*0.1
            loss.backward()
            opt.step()
            # print(loss.item(), end='\t')
        return trans.detach().cpu().numpy(), pose.detach().cpu().numpy()

    def __call__(self, data):
        data = copy.deepcopy(data)
        trans_l, pose_l = self.optimize_pose(
            self.left_mano_layer,
            init_pose=np.concatenate([data['rot_l'], data['pose_l']], axis=-1),
            init_trans=data['trans_l'],
            joints_target=data['pos_l'],
            beta=data['shape_l']
        )
        trans_r, pose_r = self.optimize_pose(
            self.right_mano_layer,
            init_pose=np.concatenate([data['rot_r'], data['pose_r']], axis=-1),
            init_trans=data['trans_r'],
            joints_target=data['pos_r'],
            beta=data['shape_r']
        )
        data['trans_l'] = trans_l
        data['trans_r'] = trans_r
        data['rot_l'] = pose_l[:, :3]
        data['pose_l'] = pose_l[:, 3:]
        data['rot_r'] = pose_r[:, :3]
        data['pose_r'] = pose_r[:, 3:]
        return data
