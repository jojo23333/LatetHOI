import multiprocessing
from itertools import repeat

import numpy as np
import smplx
import torch
import trimesh
from tqdm import tqdm

from . import contactutils
from .intersect import get_sample_intersect_volume
# from .intersect_cgal import do_intersect_single_frame_cgal
from .power_spectrum import power_spectrum, ps_entropy, split_into_equal_len_chunks, ps_kld, split_into_equal_len_chunks_list, \
    extract_chunk_stack_list


def _pre_compute_closest_dist(frame, obj_faces, obj_vertices, sbj_vertices):
    if isinstance(obj_faces, torch.Tensor):
        obj_faces = obj_faces.cpu().detach().numpy()
    if isinstance(obj_vertices, torch.Tensor):
        obj_vertices = obj_vertices.cpu().detach().numpy()
    if isinstance(sbj_vertices, torch.Tensor):
        sbj_vertices = sbj_vertices.cpu().detach().numpy()
    obj_mesh = trimesh.Trimesh(vertices=obj_vertices[frame], faces=obj_faces)
    trimesh.repair.fix_normals(obj_mesh)
    _, _dist_to_closets_point_on_obj, _, = trimesh.proximity.closest_point(obj_mesh, sbj_vertices[frame])
    return _dist_to_closets_point_on_obj


class ObjectContactMetrics:
    def __init__(self, device, use_multiprocessing=False):
        self.volume = {}
        self.depth = {}
        self.contact_ratio = {}

        self._use_multiprocessing = use_multiprocessing
        self.device = device
        # self.pool = multiprocessing.Pool(64)
        # return

    def compute_metrics(self, sbj_vertices, sbj_faces, obj_vertices, obj_faces, obj_poses=None, start_id=0, end_id=-1):
        assert (len(sbj_vertices.shape) == 3)
        n_frames = sbj_vertices.shape[0]
        n_sbj_vertices = sbj_vertices.shape[1]
        assert (sbj_vertices.shape[2] == 3)
        volume_list = []
        depth_list = []
        ratio_list = []

        contact_frames = []

        if end_id == -1:
            end_id = n_frames
        if start_id == -1 or start_id == end_id:
            volume_list = [0.0]
            depth_list = [0.0]
            ratio_list = [0.0]
        else:
            if self._use_multiprocessing:
                dist_to_closest_point_list = self.pool.starmap(_pre_compute_closest_dist, zip(np.arange(start=start_id, stop=end_id), repeat(obj_faces),
                                                                                     repeat(obj_vertices), repeat(sbj_vertices)))
                self.pool.close()
                self.pool.join()

            for frame in range(start_id, end_id):
                obj_triangles = obj_vertices[frame][obj_faces]
                exterior = contactutils.batch_mesh_contains_points(sbj_vertices[None, frame, :].float().to(self.device),
                                                    obj_triangles[None, :, :].float().to(self.device),
                                                    torch.Tensor([0.4395064455, 0.617598629942, 0.652231566745]).to(self.device))
                penetr_mask = ~exterior.squeeze(dim=0)

                if penetr_mask.sum() == 0:
                    max_depth = 0.0
                    volume = 0.0
                    contact_ratio = 0.0
                else:
                    if self._use_multiprocessing:
                        self._dist_to_closets_point_on_obj = dist_to_closest_point_list[frame-start_id]
                    else:
                        self._dist_to_closets_point_on_obj = _pre_compute_closest_dist(frame, obj_faces, obj_vertices, sbj_vertices)

                    max_depth, volume = self.compute_interpenetration_volume_depth_mesh_2_mesh(obj_faces=obj_faces,
                                                                                            obj_vertices=obj_vertices[frame],
                                                                                            sbj_faces=sbj_faces,
                                                                                            sbj_vertices=sbj_vertices[frame],
                                                                                            penetr_mask=penetr_mask.detach().cpu().numpy())

                    contact_ratio = self.compute_contact_ratio(obj_faces=obj_faces, obj_vertices=obj_vertices[frame],
                                                            sbj_vertices=sbj_vertices[frame])
                    contact_frames.append(frame-start_id)

                volume_list += [volume]
                depth_list += [max_depth]
                ratio_list += [contact_ratio]

        if len(contact_frames) == 0:
            contact_frames = [0]
            contact_frames_res = []
        else:
            contact_frames_res = contact_frames
        contact_frames = np.array(contact_frames)

        obj_vertices_dis = obj_vertices[1:] - obj_vertices[:-1]
        obj_vertices_dis = torch.norm(obj_vertices_dis, dim=-1).mean(dim=-1)
        off_ground_masks = (obj_vertices_dis > 0.01).to(torch.int)
        off_ground_masks = torch.cat([torch.zeros((1,), dtype=torch.int32), off_ground_masks], dim=0)

        self.volume = {
            "inter_volume_mean": np.mean(volume_list), 
            "inter_volume_contact": np.mean(np.array(volume_list)[contact_frames]),
            "inter_volume_max": np.max(volume_list),
            # "inter_volume_mean_last_5": np.mean(volume_list[-5:]),
            # "inter_volume_last": volume_list[-1],
        }
        self.depth = {
            "inter_depth_mean": np.mean(depth_list), 
            "inter_depth_contact": np.mean(np.array(depth_list)[contact_frames]), 
            "inter_depth_max": np.max(depth_list),
            # "inter_depth_mean_last_5": np.mean(depth_list[-5:]),
            # "inter_depth_last": depth_list[-1]
        }
        self.contact_ratio = {
            "contact_ratio_mean": np.mean(ratio_list), 
            "contact_ratio_contact": np.mean(np.array(ratio_list)[contact_frames]), 
            "contact_ratio_max": np.max(ratio_list),
            # "contact_ratio_mean_last_5": np.mean(ratio_list[-5:]),
            # "contact_ratio_last": ratio_list[-1]
            "contact_ratio_off_ground": np.mean(np.array(ratio_list)[off_ground_masks])
        }

        jerk = compute_jerk(sbj_vertices)

        self.jerk = {"jerk": float(jerk)}

        return volume_list, depth_list, ratio_list, off_ground_masks

    def compute_interpenetration_volume_depth_mesh_2_mesh(self, obj_faces, obj_vertices, sbj_faces, sbj_vertices, penetr_mask):
        """
        Original source: https://github.com/hwjiang1510/GraspTTA/tree/master/metric
        https://github.com/CGAL/cgal-swig-bindings
        """
        if isinstance(obj_faces, torch.Tensor):
            obj_faces = obj_faces.cpu().detach().numpy()
        if isinstance(obj_vertices, torch.Tensor):
            obj_vertices = obj_vertices.cpu().detach().numpy()
        if isinstance(sbj_faces, torch.Tensor):
            sbj_faces = sbj_faces.cpu().detach().numpy()
        if isinstance(sbj_vertices, torch.Tensor):
            sbj_vertices = sbj_vertices.cpu().detach().numpy()
        #if do_intersect_single_frame_cgal(sbj_vertices, sbj_faces, obj_vertices, obj_faces):
        volume = get_sample_intersect_volume(
            sample_info={
                "sbj_verts": sbj_vertices,
                "obj_verts": obj_vertices,
                "sbj_faces": sbj_faces,
                "obj_faces": obj_faces
            }, mode="voxels"  # voxels
        )
        if volume is None:
            volume = 0.0
            max_depth = 0.0
        else:
            float(volume*1e6)

        max_depth = self.compute_max_depth(obj_faces, obj_vertices, sbj_vertices, penetr_mask)
        # else:
        #     volume = 0.0
        #     max_depth = 0.0

        return max_depth, volume

    def compute_contact_ratio(self, obj_faces, obj_vertices, sbj_vertices, in_contact_threshold=0.005):
        """
        nr. sbj vertices / nr. sbj vertices close to object (below in_contact_threshold)
        """
        n_sbj_vertices = sbj_vertices.shape[0]

        n_verts_in_contact = np.sum(self._dist_to_closets_point_on_obj < in_contact_threshold)

        ratio = n_verts_in_contact / n_sbj_vertices

        return ratio

    def compute_max_depth(self, obj_faces, obj_vertices, sbj_vertices, penetr_mask):
        """
        Original source: https://github.com/hwjiang1510/GraspTTA/tree/master/metric/penetration.py
        """
        obj_mesh = trimesh.Trimesh(vertices=obj_vertices, faces=obj_faces)
        trimesh.repair.fix_normals(obj_mesh)

        if penetr_mask.sum() == 0:
            max_depth = 0.0
        else:
            max_depth = self._dist_to_closets_point_on_obj[penetr_mask == 1].max()

        return max_depth


def human_ground_interpenetration_depth(sbj_vertices, up_dir=2):
    verts_below_ground_mask = sbj_vertices[:, :, up_dir] < 0.0

    avg_depth = torch.mean(sbj_vertices[..., up_dir][verts_below_ground_mask]).item()

    return {"avg_ground_interpenetration_depth": avg_depth}




def compute_jerk(positions):
    # Calculate velocity by taking the first derivative of positions
    velocity = np.gradient(positions, axis=0)

    # Calculate acceleration by taking the first derivative of velocity
    acceleration = np.gradient(velocity, axis=0)

    # Calculate jerk by taking the first derivative of acceleration
    jerk = np.gradient(acceleration, axis=0)

    # If you want the magnitude of jerk for each time point:
    jerk_magnitude = np.linalg.norm(jerk, axis=-1)

    return np.mean(jerk_magnitude)


def diversity():
    """
    Average L2 Pairwise Distance over multiple samples

    https://github.com/Khrylx/DLow/blob/master/motion_pred/eval.py
    https://arxiv.org/pdf/2003.08386.pdf
    """

    return


def pos_2_acc(pos_sequence_list: list):
    """
    (n_motions, seq_len, n_joints, n_features)
    """
    acc_sequence_list = []
    for pos_seq in pos_sequence_list:
        acc_seq = np.diff(pos_seq, n=2, axis=0)
        acc_sequence_list.append(acc_seq)

    return acc_sequence_list


def eval_PSKL_J_list(joint_pos_pred, joint_pos_gt, eval_seq_len=30):
    """
    https://github.com/sanweiliti/LEMO
    https://github.com/eth-ait/motion-transformer/blob/c134b266d2cdcc247a169c5369e3b3a747e65ee3/spl/evaluation_dist_metrics_amass.py#L379

    (Predicted, Ground Truth) and (Ground Truth, Predicted)

    :param joint_pos_pred: [batch_size/n_motions, (sequence_length/n_steps, n_joints, n_features)]
    :param joint_pos_gt: [batch_size/n_motions, (sequence_length/n_steps, n_joints, n_features)]
    :param eval_seq_len: number of frames

    """

    # compute acceleration
    joint_accel_pred = pos_2_acc(joint_pos_pred)
    joint_accel_gt = pos_2_acc(joint_pos_gt)

    results = {}

    all_gt_target_eval_len = split_into_equal_len_chunks_list(joint_accel_gt, eval_seq_len)
    # -> (batch_size, n_joints, eval_seq_len, feature_size)
    all_gt_target_eval_len = all_gt_target_eval_len.swapaxes(1, 2)

    ps_gt_test = power_spectrum(all_gt_target_eval_len)
    ent_gt_test = ps_entropy(ps_gt_test)

    results["entropy_gt_test"] = ent_gt_test.mean()

    results["entropy_prediction"] = list()
    results["kld_prediction_test"] = list()
    results["kld_test_prediction"] = list()

    pred_len = max([p.shape[0] for p in joint_accel_pred])

    for sec, frame in enumerate(range(0, pred_len - eval_seq_len + 1, 1)):
        joint_accel_stacked_pred = extract_chunk_stack_list(joint_accel_pred, frame, eval_seq_len)
        # -> (batch_size, n_joints, eval_seq_len, feature_size)
        joint_accel_stacked_pred = joint_accel_stacked_pred.swapaxes(1, 2)

        # compare a chunk of the predictions with the real data
        ps_pred = power_spectrum(joint_accel_stacked_pred)

        ent_pred = ps_entropy(ps_pred)
        results["entropy_prediction"].append(ent_pred.mean())

        kld_pred_test = ps_kld(ps_pred, ps_gt_test)
        results["kld_prediction_test"].append(kld_pred_test.mean())

        kld_test_pred = ps_kld(ps_gt_test, ps_pred)
        results["kld_test_prediction"].append(kld_test_pred.mean())

    mean_results = {}
    for k, v in results.items():
        if not np.isscalar(v):
            mean_results[k + "_mean"] = np.mean(v[::eval_seq_len])  # only take non overlapping
            mean_results[k + "_mean_overlap"] = np.mean(v)

    return {**mean_results, **results}


def eval_PSKL_J(joint_pos_pred, joint_pos_gt, eval_seq_len=30):
    """
    https://github.com/sanweiliti/LEMO
    https://github.com/eth-ait/motion-transformer/blob/c134b266d2cdcc247a169c5369e3b3a747e65ee3/spl/evaluation_dist_metrics_amass.py#L379

    (Predicted, Ground Truth) and (Ground Truth, Predicted)

    :param joint_pos_pred: (n_motions, sequence_length, n_joints, n_features)
    :param joint_pos_gt: (n_motions, sequence_length, n_joints, n_features)
    :param eval_seq_len: number of frames

    """

    # compute acceleration
    joint_accel_pred = np.diff(joint_pos_pred, n=2, axis=1)
    joint_accel_gt = np.diff(joint_pos_gt, n=2, axis=1)

    results = {}

    all_gt_target_eval_len = split_into_equal_len_chunks(joint_accel_gt, eval_seq_len)
    # -> (batch_size, n_joints, seq_len, feature_size)
    all_gt_target_eval_len = all_gt_target_eval_len.swapaxes(1, 2)

    ps_gt_test = power_spectrum(all_gt_target_eval_len)
    ent_gt_test = ps_entropy(ps_gt_test)

    results["entropy_gt_test"] = ent_gt_test.mean()

    results["entropy_prediction"] = list()
    results["kld_prediction_test"] = list()
    results["kld_test_prediction"] = list()

    # -> (batch_size, n_joints, seq_len, feature_size)
    joint_accel_pred = joint_accel_pred.swapaxes(1, 2)
    pred_len = joint_accel_pred.shape[2]

    for sec, frame in enumerate(range(0, pred_len - eval_seq_len + 1, 1)):
        # Compare 1 second chunk of the predictions with 1 second real data.
        ps_pred = power_spectrum(joint_accel_pred[:, :, frame:frame + eval_seq_len])

        ent_pred = ps_entropy(ps_pred)
        results["entropy_prediction"].append(ent_pred.mean())

        kld_pred_test = ps_kld(ps_pred, ps_gt_test)
        results["kld_prediction_test"].append(kld_pred_test.mean())

        kld_test_pred = ps_kld(ps_gt_test, ps_pred)
        results["kld_test_prediction"].append(kld_test_pred.mean())

    mean_results = {}
    for k, v in results.items():
        mean_results[k + "_mean"] = np.mean(v)

    return {**mean_results, **results}


def foot_skating_heel(sbj_vertices, thresh_height=5e-2, thresh_vel=75e-3, frames_per_s=30):
    """
    SAGA: The heel is within 5cm of the ground and the heel speed of both feet exceeds 75mm/s.

    Original source: https://github.com/yz-cnsdqz/MOJO-release/blob/master/experiments/eval_bm_skating.py

    :param thresh_height: [m]
    :param thresh_vel: [m/s]
    :param framerate:
    """
    smplx_vertx_ids = smplx.vertex_ids.vertex_ids["smplx"]

    feetmarkeridx = [smplx_vertx_ids['LHeel'], smplx_vertx_ids['RHeel']]

    verts_feet = sbj_vertices[:, feetmarkeridx, :]
    verts_feet_horizon_vel = torch.linalg.norm(verts_feet[1:, :, :-1] - verts_feet[:-1, :, :-1], axis=-1)
    verts_feet_height = sbj_vertices[1:, feetmarkeridx, -1]

    thresh_vel_frame = thresh_vel / frames_per_s

    skating = (verts_feet_horizon_vel > thresh_vel_frame) * (torch.abs(verts_feet_height) < thresh_height)
    skating = torch.sum(torch.logical_and(skating[:, 0], skating[:, 1])) / verts_feet_horizon_vel.shape[0]

    return {"foot_skating": skating.item()}


def foot_skating_lowest_vertex(sbj_vertices, thresh_vel=3e-1, frames_per_s=30):
    """
    Similar like in GOAL https://arxiv.org/abs/2112.11454.

    Foot skating defined by: Lowest vertex velocity threshold.
    """
    _, lowest_height_vert_ids = torch.min(sbj_vertices[:, :, -1], dim=1)

    verts_feet = sbj_vertices[:, lowest_height_vert_ids, :]
    verts_feet_horizon_vel = torch.linalg.norm(verts_feet[1:, :, :-1] - verts_feet[:-1, :, :-1], axis=-1)

    # select lowest vertex per frame
    verts_feet_horizon_vel = torch.diag(verts_feet_horizon_vel)

    # vels = []
    # for i in range(verts_feet.shape[1] - 1):
    #     vels.append(torch.linalg.norm(verts_feet[i + 1, i, :-1] - verts_feet[i, i, :-1], axis=-1))
    #
    # vels = torch.stack(vels)
    # assert torch.allclose(vels, verts_feet_horizon_vel)

    thresh_vel_frame = thresh_vel / frames_per_s

    skating = (verts_feet_horizon_vel > thresh_vel_frame)
    skating = torch.sum(skating) / verts_feet_horizon_vel.shape[0]

    return {"foot_skating_lowest_vert": skating.item()}


def foot_skating_avg_velocity(sbj_vertices, ground_contact_threshold=0.0005, frames_per_s=30):
    """
    Like in SimPoE.

    Compute average velocity for vertices in contact with the ground.
    """

    vert_on_ground_consecutive_frames = torch.logical_and(
        sbj_vertices[:-1, :, 2] <= 0.001,
        sbj_vertices[1:, :, 2] <= 0.001
    )

    n_frames = sbj_vertices.shape[0]

    verts_in_ground_contact_vel = []
    for i in range(n_frames - 1):
        if torch.any(vert_on_ground_consecutive_frames[i]):
            verts = sbj_vertices[i, vert_on_ground_consecutive_frames[i]]
            verts_next = sbj_vertices[i + 1, vert_on_ground_consecutive_frames[i]]

            verts_horizon_vel_per_frame = torch.linalg.norm(verts_next[..., :-1] - verts[..., :-1], axis=-1)

            verts_horizon_vel = verts_horizon_vel_per_frame * frames_per_s
            verts_in_ground_contact_vel.append(torch.mean(verts_horizon_vel).item())
        # else:
        # print("no ground contact")

    return {"foot_skating_avg_vel": np.mean(verts_in_ground_contact_vel)}
