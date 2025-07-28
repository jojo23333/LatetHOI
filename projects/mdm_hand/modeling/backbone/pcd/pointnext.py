
from openpoints.models import build_model_from_cfg
from openpoints.utils import EasyConfig, load_checkpoint
import logging
# logging.basicConfig(level=logging.INFO)

def get_pointnext(freeze_encoder=True):
    cfg = EasyConfig()
    cfg.load('./config/pointnext-s.yaml', recursive=True)
    model = build_model_from_cfg(cfg.model)
    load_checkpoint(model, '/mnt/graphics_ssd/nimble/users/muchenli/modelnet40ply2048-train-pointnext-s.pth')
    # freeze encoder
    if freeze_encoder:
        for n, p in model.encoder.named_parameters():
            if p.requires_grad:
                p.requires_grad = False
                logging.info(f"Freezing parameters from encoder: {n} : {str(p.shape)}")
    return model

def get_pointnext_normal():
    cfg = EasyConfig()
    cfg.load('./config/pointnext-s.yaml', recursive=True)
    cfg.model.encoder_args.in_channels = 6
    model = build_model_from_cfg(cfg.model)
    # load_checkpoint(model, '/mnt/graphics_ssd/nimble/users/muchenli/modelnet40ply2048-train-pointnext-s.pth')
    return model

def get_pointnext_96(in_channels=64):
    cfg = EasyConfig()
    cfg.load('./config/pointnext-s_c96.yaml', recursive=True)
    cfg.model.encoder_args.in_channels=in_channels
    model = build_model_from_cfg(cfg.model)
    return model