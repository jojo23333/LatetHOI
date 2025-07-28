import torch
import logging
import os
import glob
import torch.multiprocessing.spawn
import trimesh, smplx
import d2.utils.comm as comm
import numpy as np

from os import path as osp
from typing import Any
from utils.object_model import ObjectModel
from tqdm import tqdm
from .physic_metrics import PhysicEvaluation
from .render import MotionRender

logger = logging.getLogger("d2")

# NOTE finished here, waiting to test
def get_hand_models(cfg, n_comps=24):
    if cfg.DATASET.NAME.startswith('grab'):
        smplx_path = cfg.PATH.SMPLX
        grab_sbj_mesh = os.path.join(cfg.PATH.GRAB, 'tools', 'subject_meshes')
        rhand_mesh = glob.glob(os.path.join(grab_sbj_mesh, '**/*_rhand.ply'))
        
        models = {}
        for mesh in rhand_mesh:
            lh_vtemp = trimesh.load(mesh).vertices
            rh_vtemp = trimesh.load(mesh.replace('_rhand.ply', '_lhand.ply')).vertices
            sbj_id = os.path.basename(mesh).split('_')[0]
            models[sbj_id] = {
                    "left" : smplx.create(
                        model_path=smplx_path,
                        model_type='mano',
                        is_rhand=False,
                        v_template=lh_vtemp,
                        num_pca_comps=n_comps,
                        flat_hand_mean=True
                    ),
                    "right": smplx.create(
                        model_path=smplx_path,
                        model_type='mano',
                        is_rhand=True,
                        v_template=rh_vtemp,
                        num_pca_comps=n_comps,
                        flat_hand_mean=True
                    )
            }
    else:
        raise NotImplementedError

def average_dicts(dict_list):
    if not dict_list:
        return {}

    def add_dicts(d1, d2):
        """ Adds two dictionaries, summing values for common keys and recursively handling nested dictionaries. """
        for key, value in d2.items():
            if key in d1:
                if isinstance(value, dict) and isinstance(d1[key], dict):
                    add_dicts(d1[key], value)
                else:
                    d1[key] += value
            else:
                d1[key] = value if not isinstance(value, dict) else value.copy()
        return d1

    def divide_dict(d, n):
        """ Divides all values in the dictionary by n, recursively handling nested dictionaries. """
        for key in d:
            if isinstance(d[key], dict):
                divide_dict(d[key], n)
            else:
                d[key] /= n

    # Sum all dictionaries
    total_dict = dict_list[0].copy()
    for d in dict_list[1:]:
        add_dicts(total_dict, d)

    # Divide by the number of dictionaries to get the average
    divide_dict(total_dict, len(dict_list))

    return total_dict


# TODO here do generation for mdm then do evaluation
class EvalGeneration:
    def __init__(self, cfg, diff_wrapper) -> None:
        self.cfg = cfg
        self.diff_wrapper = diff_wrapper
        self.dataset = cfg.DATASET.NAME
        
        self.objs = {}
        self.full_objs = {}
        if self.dataset.startswith('grab'):
            if 'latent' in self.dataset:
                n_comps = 45
                use_pca = False
            else:
                n_comps = 24
                use_pca = True
            NF = 160
            objs = os.path.join(cfg.PATH.GRAB, "tools/object_meshes/contact_meshes/")
            for obj in os.listdir(objs):
                obj_path = os.path.join(objs, obj)
                if obj_path.endswith('.ply'):
                    obj_mesh = trimesh.load(obj_path)
                    self.full_objs[obj.split('.ply')[0]] = ObjectModel(obj_mesh.vertices, obj_mesh.faces).to(self.cfg.DEVICE)
                    obj_mesh = obj_mesh.simplify_quadratic_decimation(1000)
                    trimesh.repair.fix_normals(obj_mesh)
                    self.objs[obj.split('.ply')[0]] = ObjectModel(obj_mesh.vertices, obj_mesh.faces).to(self.cfg.DEVICE)
        elif self.dataset.startswith('oakink'):
            if 'latent' in self.dataset:
                n_comps = 45
                use_pca = False
            else:
                n_comps = 24
                use_pca = True
            NF = 160
            ###########################################################################
            # TODO: This is for the 100 shape aligned split
            mesh_path = './data/oakink/oakink_aligned'
            for obj in os.listdir(mesh_path):
                if obj.endswith('.ply'):
                    obj_path = os.path.join(mesh_path, obj)
                    obj_name = obj.split('.')[0]
                    obj_mesh = trimesh.load(obj_path)
                    self.full_objs[obj_name] = ObjectModel(obj_mesh.vertices, obj_mesh.faces).to(self.cfg.DEVICE)
                    obj_mesh = obj_mesh.simplify_quadratic_decimation(1000)
                    trimesh.repair.fix_normals(obj_mesh)
                    self.objs[obj_name] = ObjectModel(obj_mesh.vertices, obj_mesh.faces).to(self.cfg.DEVICE)
            ###########################################################################
            # TODO this is for the colored extured split
            # mesh_path = './data/oakink/selected_texture_uv'
            # for obj in os.listdir(mesh_path):
            #     if os.path.isdir(os.path.join(mesh_path, obj)):
            #         obj_path = os.path.join(mesh_path, obj, obj+'.obj')
            #         obj_name = obj.split('.')[0]
            #         obj_mesh = trimesh.load(obj_path)
            #         self.full_objs[obj_name] = ObjectModel(obj_mesh.vertices, obj_mesh.faces).to(self.cfg.DEVICE)
            #         # obj_mesh = obj_mesh.simplify_quadratic_decimation(1000)
            #         # trimesh.repair.fix_normals(obj_mesh)
            #         self.objs[obj_name] = ObjectModel(obj_mesh.vertices, obj_mesh.faces).to(self.cfg.DEVICE)
            logger.info(self.objs.keys())
        else:
            n_comps = 45
            use_pca = False
            NF = 96
            mesh_path = './data/dexYCB/raw/models'
            for obj_name in os.listdir(mesh_path):
                mesh_file = os.path.join(mesh_path, obj_name, 'textured_simple.obj')
                # print("Preprocessing: ", obj_name)
                obj_mesh = trimesh.load(mesh_file)
                self.full_objs[obj_name] = ObjectModel(obj_mesh.vertices, obj_mesh.faces).to(self.cfg.DEVICE)
                # obj_mesh = obj_mesh.simplify_quadratic_decimation(1000)
                # trimesh.repair.fix_normals(obj_mesh)
                self.objs[obj_name] = ObjectModel(obj_mesh.vertices, obj_mesh.faces).to(self.cfg.DEVICE)

        self.rhand = smplx.create(
            model_path=cfg.PATH.SMPLX,
            model_type='mano',
            is_rhand=True,
            num_pca_comps=n_comps,
            use_pca=use_pca,
            flat_hand_mean=True,
            batch_size=cfg.TEST.BATCH_SIZE * NF
        ).to(self.cfg.DEVICE)
        self.lhand = smplx.create(
            model_path=cfg.PATH.SMPLX,
            model_type='mano',
            is_rhand=False,
            num_pca_comps=n_comps,
            use_pca=use_pca,
            flat_hand_mean=True,
            batch_size=cfg.TEST.BATCH_SIZE * NF
        ).to(self.cfg.DEVICE)

        self.physic_eval = PhysicEvaluation(cfg)
        if comm.is_main_process():
            self.MOTION_RENDER = MotionRender(interactive=False)

    def __call__(self, dataloader, it, eval_physic=False, vis=True) -> Any:
        # TODO finish here
        results = []
        cnt = 0
        pbar1 = tqdm(total=len(dataloader), position=1)
        logger.info("Begin Evaluating With Evaluation Physics: " + str(eval_physic))
        for data in dataloader:
            # make sure sample function is implemented
            seqs = self.diff_wrapper.sample(data, dataloader.dataset.unnormalize)
            if 'lhand' not in seqs:
                assert self.dataset.startswith('dexycb')
                result = self.evaluate_rhand(
                    seqs['rhand'], seqs['obj'], 
                    data['action_name'], data['obj_name'], eval_physic, vis, f"{it}/b{cnt}"
                )
            else:
                assert self.dataset.startswith('grab') or self.dataset.startswith('oakink')
                result = self.evaluate(
                    seqs['rhand'], seqs['lhand'], seqs['obj'], 
                    data['action_name'], data['obj_name'], eval_physic, vis, f"{it}/b{cnt}"
                )
            cnt = cnt + 1
            results.extend(result)
            pbar1.update(1)
            if cnt % 10 == 0:
                logger.info(f"It {cnt}: " + str(average_dicts(results)))
        avg_result = average_dicts(results)
        result = average_dicts(comm.all_gather(avg_result))
        return result

    def flatten_all_batch(self, dict, flat_n):
        for key in dict:
            dict[key] = dict[key].reshape(flat_n, -1)
        return dict

    def evaluate_rhand(self, rhand_params, obj_params, action_names, obj_names, eval_physic=False, vis=False, it='0'):
        if rhand_params['global_orient'].ndim == 3:
            bsz, nf, _ = rhand_params['global_orient'].shape
            rhand_params = self.flatten_all_batch(rhand_params, bsz*nf)

        logger.info("Getting hand mesh")
        rhand_v = self.rhand(**rhand_params).vertices.reshape(bsz, nf, -1, 3).detach().cpu()#.numpy()
        rhand_f = self.rhand.faces

        obj_v = []
        obj_f = []
        logger.info("Visualizing first: " + str(vis))
        obj_v = [self.objs[obj_name](obj_params['global_orient'][i], obj_params['transl'][i]).vertices for i, obj_name in enumerate(obj_names)]
        obj_f = [self.objs[obj_name].faces.view(np.ndarray) for obj_name in obj_names]
        rhand_v_list = [rhand_v[i] for i in range(len(obj_names))]
        rhand_f_list = [rhand_f for i in range(len(obj_names))]
        
        result = []
        if eval_physic:
            for i in range(len(obj_names)):
                result.append(self.physic_eval(
                    obj_v[i].cpu(), obj_f[i],
                    rhand_v_list[i].cpu(), rhand_f_list[i],
                    None, None
                ))
        for i, obj_name in enumerate(obj_names):
            device = comm.get_local_rank()
            export_path = osp.join(
                self.cfg.OUTPUT_DIR,
                f"vis/{it}{i}_{obj_name}_{action_names[i].replace(' ', '_')}_node{device}.pth"
            )
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            data = obj_v[i], obj_f[i], rhand_v[i], rhand_f, None, None
            torch.save(data, export_path)
        if vis:
            all_data = comm.all_gather((obj_v[0], obj_f[0], rhand_v[0], rhand_f))
            # TODO potential bug for multi node training
            if comm.is_main_process():
                for device, data in enumerate(all_data):
                    self.MOTION_RENDER.render_seq({
                        "pred": (data[0], data[1], data[2], data[3], None, None),
                    }, vid_p=osp.join(self.cfg.OUTPUT_DIR, f"vis/{it}_{action_names[0].replace(' ', '_')}_node{device}.mp4"))

        return result
        

    def evaluate(self, rhand_params, lhand_params, obj_params, action_names, obj_names, eval_physic=False, vis=False, it='0'):
        if rhand_params['global_orient'].ndim == 3:
            bsz, nf, _ = rhand_params['global_orient'].shape
            rhand_params = self.flatten_all_batch(rhand_params, bsz*nf)
            lhand_params = self.flatten_all_batch(lhand_params, bsz*nf)
        
        logger.info("Getting hand mesh")
        rhand_v = self.rhand(**rhand_params).vertices.reshape(bsz, nf, -1, 3).detach().cpu()#.numpy()
        rhand_f = self.rhand.faces
        lhand_v = self.lhand(**lhand_params).vertices.reshape(bsz, nf, -1, 3).detach().cpu()#.numpy()
        lhand_f = self.lhand.faces

        obj_v = []
        obj_f = []
        logger.info("Visualizing first: " + str(vis))
        obj_v = [self.objs[obj_name](obj_params['global_orient'][i], obj_params['transl'][i]).vertices for i, obj_name in enumerate(obj_names)]
        obj_f = [self.objs[obj_name].faces.view(np.ndarray) for obj_name in obj_names]
        rhand_v_list = [rhand_v[i] for i in range(len(obj_names))]
        rhand_f_list = [rhand_f for i in range(len(obj_names))]
        lhand_v_list = [lhand_v[i] for i in range(len(obj_names))]
        lhand_f_list = [lhand_f for i in range(len(obj_names))]
        
        result = []
        if eval_physic:
            for i in range(len(obj_names)):
                result.append(self.physic_eval(
                    obj_v[i], obj_f[i],
                    rhand_v_list[i], rhand_f_list[i],
                    lhand_v_list[i], lhand_f_list[i],
                ))
        # if eval_physic:
        #     logger.info("Evaluating physics metrics")
        #     pool = torch.multiprocessing.Pool(16)
        #     result = pool.starmap(self.physic_eval, zip(rhand_v_list, rhand_f_list, lhand_v_list, lhand_f_list, obj_v, obj_f))
        #     pool.close()
        #     pool.join()
        #     logger.info("Done Evaluating")
        # else:
        #     result = {}
        # dist_to_closest_point_list = pool.starmap(_pre_compute_closest_dist, zip(np.arange(start=start_id, stop=end_id), repeat(obj_faces),
        #                                                                              repeat(obj_vertices), repeat(sbj_vertices)))
        # torch.multiprocessing.spawn(self.physic_eval, args=(rhand_v, rhand_f, lhand_v, lhand_f, obj_v, obj_f), nprocs=2)
        for i, obj_name in enumerate(obj_names):
            device = comm.get_local_rank()
            export_path = osp.join(
                self.cfg.OUTPUT_DIR,
                f"vis/{it}{i}_{obj_name}_{action_names[i].replace(' ', '_')}_node{device}.pth"
            )
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            data = obj_v[i], obj_f[i], rhand_v[i], rhand_f, lhand_v[i], lhand_f
            torch.save(data, export_path)

        if vis:
            all_data = comm.all_gather((obj_v[0], obj_f[0], rhand_v[0], rhand_f, lhand_v[0], lhand_f))
            # TODO potential bug for multi node training
            if comm.is_main_process():
                for device, data in enumerate(all_data):
                    self.MOTION_RENDER.render_seq(
                        {"pred": (data[0], data[1], data[2], data[3], data[4], data[5])},
                        vid_p=osp.join(self.cfg.OUTPUT_DIR, f"vis/{it}_{action_names[0].replace(' ', '_')}_node{device}.mp4"),
                        save_pth=False
                    )
                # self.MOTION_RENDER.render_seq({
                #     "pred": (obj_v[0], obj_f[0], rhand_v[0], rhand_f, lhand_v[0], lhand_f),
                # }, vid_p=osp.join(self.cfg.OUTPUT_DIR, f"vis/{it}_{action_names[0].replace(' ', '_')}_node{device}.mp4"))


        return result
        
        
        