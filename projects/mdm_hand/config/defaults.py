from d2.config import CfgNode as CN


_C = CN()

_C.PROJECT_NAME = 'HDM'
_C.VERSION = 2.0

_C.CUDA = True
_C.DEVICE = 'cuda'
_C.SEED = 10
_C.OVERWRITE = False
_C.OUTPUT_DIR = ''

_C.AMP_ENABLED = False

# general path for pretrained models
_C.PATH = CN()
_C.PATH.SMPLX = "./data/body_models"
_C.PATH.dexYCB = "./data/dexYCB"
_C.PATH.GRAB = "./data/grab"

# VAE options
_C.VAE = CN()
_C.VAE.NAME = 'grabnet'
_C.VAE.batch_size = 64
_C.VAE.n_workers = 8
_C.VAE.use_multigpu = False
_C.VAE.kl_coef = 0.5
_C.VAE.dataset_dir = './dataset/GraspPose'
_C.VAE.base_dir = '/mnt/nimble/nimble-dgx/users/muchenli/SAGA'
_C.VAE.work_dir = 'logs/GraspPose/male'
_C.VAE.base_lr = 0.0005
_C.VAE.best_net = None
_C.VAE.gender = 'male'
_C.VAE.exp_name = 'male'
_C.VAE.bps_size = 4096
_C.VAE.c_weights_path = None
_C.VAE.cuda_id = 0
_C.VAE.latentD = 16
_C.VAE.log_every_epoch = 10
_C.VAE.n_epochs = 100
_C.VAE.n_markers = 512
_C.VAE.n_neurons = 512
_C.VAE.reg_coef = 0.0005
_C.VAE.seed = 4815
_C.VAE.try_num = 0
_C.VAE.vpe_path = None
_C.VAE.load_on_ram = False
_C.VAE.cond_object_height = True
_C.VAE.motion_intent = False
_C.VAE.object_class = ['all']
_C.VAE.robustkl = False
_C.VAE.kl_annealing = False
_C.VAE.kl_annealing_epoch = 50
_C.VAE.marker_weight = 1
_C.VAE.foot_weight = 0
_C.VAE.collision_weight = 0
_C.VAE.consistency_weight = 1
_C.VAE.dropout = 0.1
_C.VAE.obj_feature = 12
_C.VAE.pointnet_hc = 64
_C.VAE.continue_train = False
_C.VAE.data_representation = 'joints'
_C.VAE.contact_cond = False
_C.VAE.freeze_pointnet = False

# Diffusion options
_C.DIFFUSION = CN()
_C.DIFFUSION.VAE_CHECKPOINT = '.exps/GRABNET_EXPS/contact_cond/snapshots/TR00_E024_cnet.pt'
_C.DIFFUSION.NOISE_SCHEDULE = 'cosine'
_C.DIFFUSION.DIFFUSION_STEPS = 1000
_C.DIFFUSION.SIGMA_SMALL = True
_C.DIFFUSION.POSE_MODELING = 'xyz'

_C.DIFFUSION.TYPE = 'EDM'
_C.DIFFUSION.LOSS = 'EDMLoss'
_C.DIFFUSION.CONDITION = 'none'
_C.DIFFUSION.CONTROL = None
_C.DIFFUSION.COORD_TYPE = 'joint_world'
_C.DIFFUSION.ANGLE_LOSS = 0.
_C.DIFFUSION.LDM_SHARE_REPARAM = True
_C.DIFFUSION.LDM_USE_MEAN = False
_C.DIFFUSION.LDM_USE_MEAN_VAR = False

# Model options
_C.MODEL = CN()
_C.MODEL.NAME = 'MDMHand'
_C.MODEL.ROT_REP = '6d'
_C.MODEL.BACKBONE = 'mdm_unet'
_C.MODEL.ARCH = 'trans_enc'
_C.MODEL.EMB_TRANS_DEC = False
_C.MODEL.LAYERS = 8
_C.MODEL.INPUT_DIM = 26
_C.MODEL.LATENT_DIM = 512
_C.MODEL.COND_MASK_PROB = .1
_C.MODEL.LAMBDA_RCXYZ = 0.0
_C.MODEL.LAMBDA_VEL = 0.0
_C.MODEL.LAMBDA_FC = 0.0
_C.MODEL.UNCONSTRAINED = False
_C.MODEL.WEIGHTS = ''
_C.MODEL.CONTACT_MASK_STD = 0.

_C.MODEL.UNET = CN()
_C.MODEL.UNET.LATENT_DIM = 512
_C.MODEL.UNET.DIM_MULTS = [2, 2, 2, 2]
_C.MODEL.UNET.ADAGN = False


# Dataset options
_C.DATASET = CN()
_C.DATASET.NAME = 'arctic'
_C.DATASET.DATA_DIR = ''
_C.DATASET.FPS = 30
_C.DATASET.NUM_FRAMES = 60
_C.DATASET.NUM_WORKERS = 4

_C.DATASET.POSE = CN()
_C.DATASET.POSE.GRAB = True
_C.DATASET.POSE.DexYCB = True
_C.DATASET.POSE.RIGHT_ONLY = False

_C.DATASET.ARCTIC = CN()
_C.DATASET.ARCTIC.CANON_HAND_TRANS = False
_C.DATASET.ARCTIC.OBJECT = ""

_C.DATASET.AUG = CN() 
_C.DATASET.AUG.ROTATE_X = [0, -0.2, -0.2]
_C.DATASET.AUG.ROTATE_Y = [0, -0.2, -0.2]
_C.DATASET.AUG.ROTATE_Z = [0, -0.2, -0.2]
_C.DATASET.AUG.NOISE = 0.

# Training options
_C.TRAINING = CN()
_C.TRAINING.BATCH_SIZE = 512
_C.TRAINING.TRAIN_PLATFORM_TYPE = 'NoPlatform'
_C.TRAINING.LR = 1e-4

_C.TRAINING.EVAL_BATCH_SIZE = 32
_C.TRAINING.EVAL_SPLIT = 'test'
_C.TRAINING.EVAL_DURING_TRAINING = False
_C.TRAINING.EVAL_REP_TIMES = 3
_C.TRAINING.EVAL_NUM_SAMPLES = 1000
_C.TRAINING.LOG_INTERVAL = 100
_C.TRAINING.SAVE_INTERVAL = 50000
_C.TRAINING.NUM_STEPS = 600000
_C.TRAINING.NUM_FRAMES = 60
_C.TRAINING.RESUME_CHECKPOINT = ''

_C.SOLVER = CN()

_C.SOLVER.BASE_LR = 2e-4
_C.SOLVER.BASE_LR_END = 5e-5   # The end lr, only used by WarmupCosineLR
_C.SOLVER.WEIGHT_DECAY = 2e-5

# Options: WarmupMultiStepLR, WarmupCosineLR.
# See detectron2/solver/build.py for definition.
_C.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = () # constant lr for now
_C.SOLVER.NUM_DECAYS = 3 # Number of decays in WarmupStepWithFixedGammaLR schedule
_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"
_C.SOLVER.RESCALE_INTERVAL = False

_C.SOLVER.MAX_ITER = 300000
_C.SOLVER.MAX_EPOCH = 100
_C.SOLVER.CHECKPOINT_PERIOD = 10000
_C.SOLVER.CKPT_KEEP_NUM = 5

_C.TEST = CN()
_C.TEST.SPLIT = 'test'
_C.TEST.BATCH_SIZE = 4
_C.TEST.EVAL_PERIOD = 200000
_C.TEST.VIS_PERIOD = 200000
_C.TEST.VIS_NUM = 30
_C.TEST.MODE = 'inpaint'
_C.TEST.INPAINT_RATE = 2
_C.TEST.EVAL_IP = False # evaluate inter penetration

# Sampling options
_C.SAMPLING = CN()
_C.SAMPLING.MODEL_PATH = ''
_C.SAMPLING.OUTPUT_DIR = ''
_C.SAMPLING.NUM_SAMPLES = 10
_C.SAMPLING.NUM_REPETITIONS = 3
_C.SAMPLING.GUIDANCE_PARAM = 2.5

# Generate options
_C.GENERATE = CN()
_C.GENERATE.MOTION_LENGTH = 6.0
_C.GENERATE.INPUT_TEXT = ''
_C.GENERATE.ACTION_FILE = ''
_C.GENERATE.TEXT_PROMPT = ''
_C.GENERATE.ACTION_NAME = ''
_C.GENERATE.SAMPLE_NUM = 1

# Edit options
_C.EDIT = CN()
_C.EDIT.EDIT_MODE = 'in_between'
_C.EDIT.TEXT_CONDITION = ''
_C.EDIT.PREFIX_END = 0.25
_C.EDIT.SUFFIX_START = 0.75

# Evaluation options
_C.EVAL = CN()
_C.EVAL.MODEL_PATH = ''
_C.EVAL.EVAL_MODE = 'wo_mm'
_C.EVAL.GUIDANCE_PARAM = 2.5


def get_cfg():
    return _C


