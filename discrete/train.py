import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import wandb

from env import UAVScheduling
from config import TrainConfig
from stable_baselines3.common.env_util import make_vec_env
from agent import Agent


def calc_runtime(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        run_time_hours = run_time / 3600.0
        print(f"Training executed in {run_time:.4f} seconds")
        return 
    return wrapper


@ calc_runtime
def model_train(envs, device):
    ppo = Agent().to(device)
    optimizer = optim.Adam(ppo.parameters(), lr=TrainConfig.LEARNING_RATE, eps=1e-5)

    # ALGO: storage setup
    obs = torch.zeros((TrainConfig.NUM_STEPS, TrainConfig.NUM_ENVS)+envs.single_observation_space.shape).to(device)
    actions = torch.zeros((TrainConfig.NUM_STEPS, TrainConfig.NUM_ENVS)+envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((TrainConfig.NUM_STEPS, TrainConfig.NUM_ENVS)+(2,)).to(device)
    rewards = torch.zeros((TrainConfig.NUM_STEPS, TrainConfig.NUM_ENVS)).to(device)
    dones = torch.zeros((TrainConfig.NUM_STEPS, TrainConfig.NUM_ENVS)).to(device)
    values = torch.zeros((TrainConfig.NUM_STEPS, TrainConfig.NUM_ENVS)).to(device)

    next_obs, info = envs.reset()
    next_obs = torch.tensor(next_obs).to(device)
    next_mask = torch.tensor(info["mask"]).to(device)   # the shape should be (NUM_ENVS, 20)
    next_done = torch.zeros((TrainConfig.NUM_ENVS,)).to(device)
    num_updates = TrainConfig.TOTAL_TIMESTEPS // TrainConfig.NUM_STEPS
    global_steps = 0

    for update in range(1, num_updates+1):
        for step in range(TrainConfig.NUM_STEPS):
            global_steps += TrainConfig.NUM_ENVS
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = ppo.get_action_and_value(next_obs, next_mask)
                values[step] = value.flatten()
            actions[step] = torch.stack(action, axis=1)
            logprobs[step] = torch.stack(logprob, axis=1)


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
    env = UAVScheduling
    vec_env = make_vec_env(env, n_envs=TrainConfig.NUM_ENVS)

    # start training
    model_train(vec_env, device)



if __name__ == "__main__":
    main()