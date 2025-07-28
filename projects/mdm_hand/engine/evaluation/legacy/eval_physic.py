import smplx
import trimesh
import os
import torch
import open3d as o3d
import numpy as np
from smplx import MANO
from scipy.spatial.transform import Rotation as R
from utils.metrics import ObjectContactMetrics
from utils.arctic.common.object_tensors import ObjectTensors
from utils.arctic.common.mesh import Mesh

GRAB_OBJECTS = ['cylinderlarge', 'mug', 'elephant', 'hand', 'cubelarge', 'stanfordbunny', 'airplane', 'alarmclock', 'banana', 'body', 'bowl', 'cubesmall', 'cup', 'doorknob', 'cubemedium',
             'eyeglasses', 'flashlight', 'flute', 'gamecontroller', 'hammer', 'headphones', 'knife', 'lightbulb', 'mouse', 'phone', 'piggybank', 'pyramidlarge', 'pyramidsmall', 'pyramidmedium',
             'duck', 'scissors', 'spherelarge', 'spheresmall', 'stamp', 'stapler', 'table', 'teapot', 'toruslarge', 'torussmall', 'train', 'watch', 'cylindersmall', 'waterbottle', 'torusmedium',
             'cylindermedium', 'spheremedium', 'wristwatch', 'wineglass', 'camera', 'binoculars', 'fryingpan', 'toothpaste', 'apple', 'toothbrush']

ARCTIC_OBJECTS = [
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

def compute_all_metrics(_vertices_sbj, _faces_sbj, _verts_obj, _faces_obj, obj_poses, start_id, end_id, fps=30, eval_contact_last_x_frames_only=False,
                        eval_contact_evenly_distributed=False, eval_contact_n_frames=5, device=None) -> dict:

    contact_collision_metric = ObjectContactMetrics(device, False)

    if eval_contact_last_x_frames_only:
        contact_collision_metric.compute_metrics(sbj_vertices=_vertices_sbj[-eval_contact_n_frames:],
                                                    sbj_faces=_faces_sbj,
                                                    obj_vertices=_verts_obj[-eval_contact_n_frames:], obj_faces=_faces_obj,
                                                    obj_poses=obj_poses, start_id=start_id, end_id=end_id)
    elif eval_contact_evenly_distributed:
        frames = np.linspace(start=0, stop=len(_verts_obj) - 1, num=eval_contact_n_frames).round()
        contact_collision_metric.compute_metrics(sbj_vertices=_vertices_sbj[frames],
                                                    sbj_faces=_faces_sbj,
                                                    obj_vertices=_verts_obj[frames], obj_faces=_faces_obj,
                                                    obj_poses=obj_poses, start_id=start_id, end_id=end_id)
    else:
        contact_collision_metric.compute_metrics(sbj_vertices=_vertices_sbj, sbj_faces=_faces_sbj,
                                                    obj_vertices=_verts_obj, obj_faces=_faces_obj,
                                                    obj_poses=obj_poses, start_id=start_id, end_id=end_id)

    volume_pred = contact_collision_metric.volume
    depth_pred = contact_collision_metric.depth
    contact_ratio_pred = contact_collision_metric.contact_ratio
    jerk = contact_collision_metric.jerk
    return {**volume_pred, **depth_pred, **contact_ratio_pred, **jerk}


class PhysicEvaluation:
    def __init__(self, cfg):
        MANO_MODEL_DIR = './data/ARCTIC/body_models/mano'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = {
            "right": MANO(
                MANO_MODEL_DIR,
                create_transl=True,
                use_pca=False,
                flat_hand_mean=True,
                is_rhand=True,
            ).to(self.device),
            "left": MANO(
                MANO_MODEL_DIR,
                create_transl=True,
                use_pca=False,
                flat_hand_mean=True,
                is_rhand=False,
            ).to(self.device),
            "object": ObjectTensors()
        }
        self.layers['object'].to('cuda')

    def decimate(self, v, f, target_nf=1000):
        mesh = o3d.geometry.TriangleMesh()
        vertices = []
        faces = []
        for i in range(v.shape[0]):
            mesh.vertices = o3d.utility.Vector3dVector(v[i])
            mesh.triangles = o3d.utility.Vector3iVector(f)
            # smp_mesh = mesh.simplify_vertex_clustering(
            #     voxel_size=0.005,
            #     contraction=o3d.geometry.SimplificationContraction.Average)
            smp_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_nf)
            vertices.append(np.asarray(smp_mesh.vertices))
            faces.append(np.asarray(smp_mesh.triangles))
            assert not np.isnan(vertices[-1]).any() 
            assert not np.isnan(faces[-1]).any()
        # Get vertices
        return vertices, faces

    def __call__(self, batch):
        all_meshes = {}
        for k, v in batch.items():
            if isinstance(v, np.ndarray):
                batch[k] = torch.from_numpy(v).to(self.device)
        with torch.no_grad():
            nf = batch["rot_obj"].shape[0]
            obj_name = ARCTIC_OBJECTS[batch['type_obj']]
            query_names = [obj_name] * nf
            obj_out = self.layers["object"](
                angles=batch["arti_obj"].view(-1, 1),
                global_orient=batch["rot_obj"],
                transl=batch["trans_obj"],
                query_names=query_names
            )
            obj_v = obj_out["v"]
            obj_f = Mesh(
                filename=f"./data/ARCTIC/arctic_data/data/meta/object_vtemplates/{obj_name}/mesh.obj"
            ).f

        shape_l = batch["shape_l"] if "shape_l" in batch else self.default_beta_left.repeat(nf, 1)
        left = self.layers["left"](
            global_orient=batch['rot_l'].to(torch.float32),
            hand_pose=batch['pose_l'].to(torch.float32),
            betas=shape_l.to(torch.float32),
            # transl=batch["trans_l"].to(torch.float32),
        )
        left_v = left.vertices - left.joints[:, 0:1, :]
        left_v = left_v + batch["trans_l"].view(nf, 1, 3)

        shape_r = batch["shape_r"] if "shape_r" in batch else self.default_beta_right.repeat(nf, 1)
        right = self.layers["right"](
            global_orient=batch['rot_r'].to(torch.float32),
            hand_pose=batch['pose_r'].to(torch.float32),
            betas=shape_r.to(torch.float32),
            # transl=batch["trans_r"].to(torch.float32)
        )
        right_v = right.vertices - right.joints[:, 0:1, :]
        right_v = right_v + batch["trans_r"].view(nf, 1, 3)
        
        # left_v, left_f = self.decimate(left_v.detach().cpu().numpy(), self.layers['left'].faces.astype(np.int32))
        # right_v, right_f = self.decimate(right_v.detach().cpu().numpy(), self.layers['right'].faces.astype(np.int32))
        obj_v, obj_f = self.decimate(obj_v.detach().cpu().numpy(), obj_f)
        obj_v = [torch.Tensor(v) for v in obj_v]
        obj_f = [torch.Tensor(f) for f in obj_f]
        left_f = torch.Tensor(self.layers['left'].faces.astype(np.int32))
        right_f = torch.Tensor(self.layers['right'].faces.astype(np.int32))

        # import ipdb; ipdb.set_trace()
        # TODO         
        res_dict_l = compute_all_metrics(
            left_v.detach().cpu(), left_f.detach().cpu(),
            obj_v, obj_f, torch.cat([batch["trans_obj"],batch["rot_obj"]], dim=-1).cpu().numpy(),
            0, nf, device=self.device
        )
        res_dict_r = compute_all_metrics(
            right_v.detach().cpu(), right_f.detach().cpu(),
            obj_v, obj_f, torch.cat([batch["trans_obj"],batch["rot_obj"]], dim=-1).cpu().numpy(),
            0, nf, device=self.device
        )
        return res_dict_l, res_dict_r


# Original Evaluation Code
class EvalNode:

    def __init__(self, model_path, sbj_vtemp_l, sbj_vtemp_r, batch_size=196):

        self.model_path = model_path
        self.batch_size = batch_size
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.mano_model_l = smplx.create(model_path, model_type="mano", use_pca=False,
                                    v_template=sbj_vtemp_l.vertices, flat_hand_mean=True,is_rhand=False,
                                    batch_size=batch_size).to(self.device)
        self.mano_model_r = smplx.create(model_path, model_type="mano", use_pca=False,
                                    v_template=sbj_vtemp_r.vertices, flat_hand_mean=True,is_rhand=True,
                                    batch_size=batch_size).to(self.device)


    def recover_from_ric(self, data, obj_rot, obj_r_pos, object_rot_relative = False, num_joints=15):
        #data = torch.tensor(data, dtype=torch.float32)
        obj_rot_quat = torch.tensor(R.from_matrix(obj_rot).as_quat(),dtype=torch.float32)
        positions_left = data[..., : num_joints * 3]
        positions_right = data[..., num_joints * 3: num_joints * 3 * 2]

        positions_left = positions_left.reshape(positions_left.shape[:-1] + (-1, 3))
        positions_right = positions_right.reshape(positions_right.shape[:-1] + (-1, 3))

        '''Add rotation to local joint positions'''
        if object_rot_relative:
            positions_left = np.matmul(obj_rot,positions_left.swapaxes(1,2)).swapaxes(1,2)
            positions_right = np.matmul(obj_rot,positions_right.swapaxes(1,2)).swapaxes(1,2)
        '''Add obj root to joints'''
        positions_left += obj_r_pos
        positions_right += obj_r_pos

        return torch.tensor(positions_left).to(self.device), torch.tensor(positions_right).to(self.device)

    def evaluate_seqs(self, samples, texts, bounding_box= False, only_mano=False):
        inter_volume = []
        inter_depth = []
        contact_ratio = []
        jerk_pos = []
        jerk_ang = []
        num_contact_frames = []
        print('len samples', len(samples[0]))
        for seq in range(len(samples[0])):
            print('Seq', seq)
            feature_vec = samples[0][seq]
            text = texts[seq]
            obj_name = [s for s in OBJECT_LIST if s in text]
            assert len(obj_name) >= 1
            if len(obj_name) > 1:
                print(obj_name)
                obj_name = ['wristwatch']
            obj_mesh = trimesh.load(os.path.join(self.model_path,'object_meshes',obj_name[0]+'.ply'), process=False)
            obj_mesh = obj_mesh.simplify_quadratic_decimation(5000)
            trimesh.repair.fix_normals(obj_mesh)
            res_l, res_r = self.evaluate(feature_vec, obj_mesh, bounding_box=bounding_box, only_mano=only_mano)
            inter_volume.append(np.mean([res_l['inter_volume_contact'],res_r['inter_volume_contact']]))
            inter_depth.append(np.mean([res_l['inter_depth_contact'],res_r['inter_volume_contact']]))
            contact_ratio.append(np.mean([res_l['contact_ratio_contact'],res_r['contact_ratio_contact']]))
            jerk_pos.append(res_l['jerk_pos'])
            jerk_ang.append(res_l['jerk_ang'])

            contact_frames = np.unique(np.concatenate([res_l['contact_frames'],res_r['contact_frames']]))
            num_contact_frames.append(contact_frames.shape[0])


        res_dict = {}
        res_dict['inter_volume'] = np.array(inter_volume)
        res_dict['inter_depth'] = np.array(inter_depth)
        res_dict['contact_ratio'] = np.array(contact_ratio)
        res_dict['jerk_pos'] = np.array(jerk_pos)
        res_dict['jerk_ang'] = np.array(jerk_ang)
        res_dict['num_contact_frames'] = np.array(num_contact_frames)

        res_dict['inter_volume_mean'] = np.mean(inter_volume)
        res_dict['inter_depth_mean'] = np.mean(inter_depth)
        res_dict['contact_ratio_mean'] = np.mean(contact_ratio)
        res_dict['jerk_pos_mean'] = np.mean(jerk_pos)
        res_dict['jerk_ang_mean'] = np.mean(jerk_ang)
        res_dict['num_contact_frames_mean'] = np.mean(num_contact_frames)

        return res_dict

    def evaluate(self, feature_vec, obj_mesh, bounding_box= False, only_mano=False, ftip=True, obj_relative=True):

        # get the subject and object vertices over the whole sequence from the sample


        obj_verts = obj_mesh.vertices
        if only_mano:
            n_joints = 1
        elif ftip:
            n_joints = 21
        else:
            n_joints = 15

        if not bounding_box:
            obj_rot = rotation_6d_to_matrix(torch.tensor(feature_vec[:,-12:-6])).reshape(-1,3,3)
            obj_verts = np.matmul(obj_verts,obj_rot).numpy()
            obj_verts +=  feature_vec[:,np.newaxis,-15:-12]
            ### 6D rot ###
            pos_left, pos_right = self.recover_from_ric(feature_vec, obj_rot,
            feature_vec[:,np.newaxis,-15:-12], object_rot_relative=True, num_joints=n_joints)
            obj_rot_vec = R.from_matrix(obj_rot).as_rotvec()
            obj_poses = np.concatenate((feature_vec[:,-15:-12],obj_rot_vec),axis=-1)
        else:
            ### BB ###
            obj_verts = feature_vec[:,-48:-24].reshape(-1,8,3)
            obj_verts[:,1:] =  obj_verts[:,1:] + obj_verts[:,:1]

            obj_sampled_verts = trimesh.bounds.corners(obj_mesh.bounding_box_oriented.bounds)
            obj_sampled_verts = np.matmul(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),np.expand_dims(obj_sampled_verts,axis=-1))[...,0]
            obj_vertices_full = []
            obj_vertices_rots = []

            obj_mesh.vertices = np.matmul(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),np.expand_dims(obj_mesh.vertices,axis=-1))[...,0]
            for k in range(obj_verts.shape[0]):
                obj_mesh_copy = obj_mesh.copy()
                transform_mesh = trimesh.registration.procrustes(obj_sampled_verts, obj_verts[k], scale=False)[0]
                obj_mesh_copy.apply_transform(transform_mesh)
                obj_vertices_full.append(obj_mesh_copy.vertices)
                obj_vertices_rots.append(transform_mesh)

            obj_verts = np.array(obj_vertices_full)
            obj_rots = np.array(obj_vertices_rots)


            obj_rot_vec = R.from_matrix(obj_rots[:,:3,:3]).as_rotvec()
            obj_poses = np.concatenate((obj_rots[:,:3,3],obj_rot_vec),axis=-1)

            pos_left, pos_right = self.recover_from_ric(feature_vec, obj_rots[:,:3,:3].reshape(-1,3,3),
            feature_vec[:,np.newaxis,-48:-45], object_rot_relative=False, num_joints=n_joints)

        joint_rotations_l = rotation_6d_to_matrix(torch.tensor(feature_vec[:,n_joints*3*2:n_joints*3*2+6*16]).reshape(-1,16,6)).numpy()
        if obj_relative:
            joint_rotations_l[:,0] = np.matmul(obj_rot,joint_rotations_l[:,0])
        joint_rotations_l = R.from_matrix(joint_rotations_l.reshape(-1,3,3)).as_rotvec().reshape(-1,16,3)
        joint_rotations_l = torch.tensor(joint_rotations_l.reshape(-1,48),dtype=torch.float32).to(self.device)

        joint_rotations_r = rotation_6d_to_matrix(torch.tensor(feature_vec[:,n_joints*3*2+6*16:n_joints*3*2+6*16*2]).reshape(-1,16,6)).numpy()
        if obj_relative:
            joint_rotations_r[:,0] = np.matmul(obj_rot,joint_rotations_r[:,0])
        joint_rotations_r = R.from_matrix(joint_rotations_r.reshape(-1,3,3)).as_rotvec().reshape(-1,16,3)
        joint_rotations_r = torch.tensor(joint_rotations_r.reshape(-1,48),dtype=torch.float32).to(self.device)

        trans_l = pos_left[:,0] -  self.mano_model_l(hand_pose=torch.zeros((self.batch_size,45)).to(self.device)).joints[0,0]
        trans_r = pos_right[:,0] - self.mano_model_r(hand_pose=torch.zeros((self.batch_size,45)).to(self.device)).joints[0,0]

        hand_seq_l = self.mano_model_l(
            hand_pose=joint_rotations_l[:,3:],
            global_orient=joint_rotations_l[:,:3],
            transl=trans_l,
        )

        hand_seq_r = self.mano_model_r(
            hand_pose=joint_rotations_r[:,3:],
            global_orient=joint_rotations_r[:,:3],
            transl=trans_r,
        )

        start_id_l, end_id_l = self.get_start_end_idcs(hand_seq_r, obj_poses, thresh=0.2)
        start_id_r, end_id_r = self.get_start_end_idcs(hand_seq_r, obj_poses, thresh=0.2)

        res_dict_l = compute_all_metrics(hand_seq_l.vertices.to(self.device), torch.tensor(self.mano_model_l.faces.astype(np.int32)).to(self.device), torch.tensor(obj_verts).to(self.device),
        torch.tensor(obj_mesh.faces.astype(np.int32)).to(self.device), torch.tensor(obj_poses).to(self.device), start_id_l, end_id_l, device=self.device)
        res_dict_r = compute_all_metrics(hand_seq_r.vertices.to(self.device), torch.tensor(self.mano_model_r.faces.astype(np.int32)).to(self.device), torch.tensor(obj_verts).to(self.device),
        torch.tensor(obj_mesh.faces.astype(np.int32)).to(self.device), torch.tensor(obj_poses).to(self.device),
        start_id_r, end_id_r, device=self.device)

        return res_dict_l, res_dict_r
        # get the correspondences between the

    def get_start_end_idcs(self, hand_seq, obj_poses, thresh=0.2):
        hand_obj_dist = np.linalg.norm(hand_seq.joints[:,0].cpu().detach().numpy() - obj_poses[:,:3],axis=-1)
        close_frames = np.where(hand_obj_dist < thresh)[0]
        if close_frames.shape[0] > 0:
            start_idx = close_frames[0]
            end_idx = close_frames[-1]
        else:
            start_idx = -1
            end_idx = -1

        return start_idx, end_idx

    