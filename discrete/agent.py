import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from utils.common_tools import make_mlp
from config import nnConfig, EnvConfig


class Agent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor_feat_list = nnConfig.act_feat_list
        self.critic_feat_list = nnConfig.critic_feat_list

        self.make_actor()
        self.critic = make_mlp(self.critic_feat_list, act_last=False, std_last=1.0)

    def make_actor(self):
        common_mlp = self.actor_feat_list[0]
        fly_fc = self.actor_feat_list[1]
        assoc_fc = self.actor_feat_list[2]

        self.mlp_common = make_mlp(common_mlp)
        self.fly_mlp = make_mlp(fly_fc, act_last=False, std_last=0.01)
        self.assoc_mlp = make_mlp(assoc_fc, act_last=False, std_last=0.01)

    def get_action_logits(self, state):
        hidden_state = self.mlp_common(state)
        fly_action = self.fly_mlp(hidden_state)
        assoc_action = self.assoc_mlp(hidden_state)
        return fly_action, assoc_action
    
    def get_value(self, state):
        return self.critic(state)
    
    def get_action_and_value(self, state, mask, actions=None):
        logit_fly, logit_assoc = self.get_action_logits(state)
        logit_fly = self._mask_process(logit_fly, mask)
        prob_fly = Categorical(logits=logit_fly)
        prob_assoc = Categorical(logits=logit_assoc)
        if actions is None:
            action_fly = prob_fly.sample()
            action_assoc = prob_assoc.sample()
        else:
            action_fly, action_assoc = actions
        
        actions = (action_fly, action_assoc)
        log_probs = (prob_fly.log_prob(action_fly), prob_assoc.log_prob(action_assoc))
        entropy = (prob_fly.entropy(), prob_assoc.entropy())
        return actions, log_probs, entropy, self.get_value(state)
    
    def _mask_process(self, logit_fly, mask):
        penalty = (1-mask) * 1e7
        logit_fly_masked = logit_fly - penalty
        return logit_fly_masked


if __name__ == "__main__":
    # validate the agent
    ppo = Agent()
    state = torch.normal(mean=0, std=2, size=(2, EnvConfig.STATE_DIM))
    mask = torch.ones((1, 20))
    actions, log_probs, entropy, value = ppo.get_action_and_value(state, mask)
    print(actions[0].shape)
    print(log_probs[0].shape)
    # print(f"fly action: {actions[0]}")
    # print(f"assoc action: {actions[1]}")
    # print(f"fly log_prob: {log_probs[0]}")
    # print(f"assoc log_prob: {log_probs[1]}")
    # print(f"fly entropy: {entropy[0]}")
    # print(f"assoc entropy: {entropy[1]}")
    # print(f"The value estimation: {value}")

    print("The agent works correctly")
