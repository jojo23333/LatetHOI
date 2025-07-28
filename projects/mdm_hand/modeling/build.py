from modeling.diffusion.edm_diffusion import VEPrecond, VPPrecond, iDDPMPrecond, EDMPrecond
from d2.modeling import build_backbone
from modeling.diffusion.respace import SpacedDiffusion, space_timesteps

from modeling.diffusion import gaussian_diffusion as gd

def create_model(cfg):
    backbone = build_backbone(cfg)
    diffusion_wrapper = f'{cfg.DIFFUSION.TYPE}Precond'
    return eval(diffusion_wrapper)(cfg, backbone, use_fp16=cfg.AMP_ENABLED)


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
        cfg=cfg,
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