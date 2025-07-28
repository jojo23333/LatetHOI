# from model.mdm import MDM
from modeling.backbone.mdm_hand import MDMHand
from modeling.backbone.mdm_hand_lr import MDMHandLR
from modeling.backbone.mdm import MDMHandMANO
from modeling.backbone.tgformer import TGFormer
from utils.parser_util import get_cond_mode

from modeling.diffusion.edm_diffusion import VEPrecond, VPPrecond, iDDPMPrecond, EDMPrecond

def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])

def create_model_and_diffusion(cfg):
    model = MDMHand(**get_model_args(cfg))
    diffusion = create_gaussian_diffusion(cfg)
    return model, diffusion

def create_model(cfg):
    diffusion_wrapper = f'{cfg.DIFFUSION.TYPE}Precond'
    if cfg.MODEL.NAME == "MDMHand":
        model = MDMHand(**get_model_args(cfg))
    elif cfg.MODEL.NAME == "MDMHandLR":
        model = MDMHandLR()
    elif cfg.MODEL.NAME == "MDMHandMANO":
        model = MDMHandMANO()
    elif cfg.MODEL.NAME == "TGFormer":
        model = TGFormer(condition_dim=15 if cfg.DIFFUSION.CONDITION == 'wrist' else 0)
    return eval(diffusion_wrapper)(cfg, model, use_fp16=cfg.AMP_ENABLED)

def create_diffusion(cfg):
    return create_gaussian_diffusion(cfg)

def get_model_args(cfg):
    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    cond_mode = 'no_cond'
    num_actions = 1
    if cfg.DIFFUSION.CONDITION == 'wrist':
        condition_dim = 9
    else:
        condition_dim = 0

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 42
    nfeats = 3

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions, 'condition_dim': condition_dim,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': cfg.MODEL.LATENT_DIM, 'ff_size': 1024, 'num_layers': cfg.MODEL.LAYERS, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': cfg.MODEL.COND_MASK_PROB, 'action_emb': action_emb, 'arch': cfg.MODEL.ARCH,
            'emb_trans_dec': cfg.MODEL.EMB_TRANS_DEC, 'clip_version': clip_version, 'dataset': cfg.DATASET.NAME}

def create_gaussian_diffusion(cfg):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = 1000
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(cfg.DIFFUSION.NOISE_SCHEDULE, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not cfg.DIFFUSION.SIGMA_SMALL
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=cfg.MODEL.LAMBDA_VEL,
        lambda_rcxyz=cfg.MODEL.LAMBDA_RCXYZ,
        lambda_fc=cfg.MODEL.LAMBDA_FC,
    )
