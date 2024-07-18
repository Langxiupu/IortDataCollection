import os
import random
import time
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import wandb

from config import TrainConfig



def main():
    run_name = "{}__{}".format(TrainConfig.WANDB_NAME, int(time.time()))
    writer = SummaryWriter(f"runs/{run_name}")

    exp_settings = {key: value for key, value in TrainConfig.__dict__.items() 
                 if not callable(value) and not key.startswith('__')}
    # initialize wandb
    wandb.init(
        project=TrainConfig.WANDB_NAME,
        sync_tensorboard=True,
        config= exp_settings,
        name=run_name,
        save_code=True
    )

    # seeding
    random.seed(TrainConfig.SEED)
    np.random.seed(TrainConfig.SEED)
    torch.manual_seed(TrainConfig.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() and TrainConfig.CUDA else "cpu")

    # env setup
    


if __name__ == "__main__":
    main()