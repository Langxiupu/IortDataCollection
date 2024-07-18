import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np

from config import EnvConfig, TrainConfig


class UAVScheduling(gym.Env):
    def __init__(self) -> None:
        super().__init__()

        self.observation_space = self._generate_obs_state()
        self.action_space = self._generate_action_space()

        self.taboo = self._generate_taboo_pos()
    
    def _calc_energy(self, vel, prop_c1=EnvConfig.PROP_C1, prop_c2=EnvConfig.PROP_C2):
        # the default time slot is 1s.
        energy_consump = prop_c1 * vel ** 3 + prop_c2 / vel
        return energy_consump
    
    def _generate_obs_state(self):
        n_iorts = EnvConfig.N_IORTS

        pos_max = EnvConfig.POS_INTERVAL[1] + 1
        dr_max = EnvConfig.SINGLE_DR_INTERVAL[1] + 1
        dist_max = EnvConfig.SINGLE_DIST_INTERVAL[1] + 1
        pow_max = EnvConfig.SINGLE_POW[1] + 1
        dr_th_max = EnvConfig.SINGLE_DR_TH[1] + 1
        energy_max = EnvConfig.ENGY[1] + 1
        obs_space = [pos_max] + [dr_max]*n_iorts + [dist_max]*n_iorts + \
            [pow_max]*n_iorts + [dr_th_max]*n_iorts + [energy_max]
        obs_space = MultiDiscrete(obs_space, seed=TrainConfig.SEED)
        return obs_space
    
    def _generate_action_space(self):
        fly_max = EnvConfig.FLY_DIM
        assoc_max = EnvConfig.SINGLE_ASSOC_INTERVAL[1] + 1
        action_space = [fly_max] + [assoc_max]*EnvConfig.N_IORTS
        return MultiDiscrete(action_space)
    
    def _generate_mask(self, pos):
        mask = np.arange(EnvConfig.FLY_DIM)
        vel_dim = len(EnvConfig.velocity)
        up = list(range(vel_dim))
        right = list(range(vel_dim, 2*vel_dim))
        down = list(range(2*vel_dim, 3*vel_dim))
        left = list(range(3*vel_dim, 4*vel_dim))
        if pos in self.taboo["left"]:
            mask[left] = 0
        if pos in self.taboo["lower"]:
            mask[down] = 0
        if pos in self.taboo["right"]:
            mask[right] = 0
        if pos in self.taboo["upper"]:
            mask[up] = 0
        return mask

    def _generate_taboo_pos(self, width=EnvConfig.AREA_WIDTH):
        left_bound = [(0, y) for y in range(width)]
        lower_bound = [(x, 0) for x in range(width)]
        right_bound = [(width-1, y) for y in range(width)]
        upper_bound = [(x, width-1) for x in range(width)]

        taboo = {}
        taboo["left"] = [self._pos_2to1(ele) for ele in left_bound]
        taboo["lower"] = [self._pos_2to1(ele) for ele in lower_bound]
        taboo["right"] = [self._pos_2to1(ele) for ele in right_bound]
        taboo["upper"] = [self._pos_2to1(ele) for ele in upper_bound]
        return taboo
    
    def _pos_2to1(self, pos, width=EnvConfig.AREA_WIDTH):
        return width*pos[0] + pos[1]