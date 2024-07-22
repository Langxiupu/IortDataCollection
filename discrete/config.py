import itertools


class TrainConfig:
    LEARNING_RATE = 2.5e-4
    SEED = 1
    TOTAL_TIMESTEPS = 100000
    CUDA = True
    WANDB_NAME = "IortCollection"

    NUM_ENVS = 4
    NUM_STEPS = 1024
    GAMMA = 0.99
    BATCH_SIZE = 128
    UPDATE_EPOCHS = 8
    CLIP_COEF = 0.2
    ENT_COEF = 0.01


class EnvConfig:
    velocity = (.0, 5.0, 10.0, 15.0, 20.0)
    direction = ('up', 'right', 'down', 'left')
    FLY_STATE = list(itertools.product(direction, velocity))
    FLY_DIM = len(FLY_STATE)

    N_IORTS = 10

    SINGLE_ASSOC_INTERVAL = (0, 1)
    ASSOC_STATE_DIM = 2**N_IORTS

    AREA_RANGE = 1000
    AREA_WIDTH = AREA_RANGE / 5
    POS_INTERVAL = (0, (AREA_WIDTH)**2-1)
    SINGLE_DR_INTERVAL = (0, 19)
    SINGLE_DIST_INTERVAL = (0, AREA_WIDTH*2)
    SINGLE_POW = (0, 1)
    SINGLE_DR_TH = (0, 1)
    ENGY = (0, 9)
    STATE_DIM = 1 + N_IORTS + N_IORTS + N_IORTS + N_IORTS + 1

    # related to the energy consumption of flying
    PROP_C1 = 9.26e-4
    PROP_C2 = 2250



class nnConfig:
    critic_feat_list = [EnvConfig.STATE_DIM, 64, 32, 1]

    common_feat_list = [EnvConfig.STATE_DIM, 128, 64, 64]
    fly_fc = [64, EnvConfig.FLY_DIM]
    assoc_fc = [64, 512, 1024, EnvConfig.ASSOC_STATE_DIM]
    act_feat_list = [common_feat_list, fly_fc, assoc_fc]
