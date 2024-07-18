import torch.nn as nn
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def make_fc(in_feature, out_feature, std=np.sqrt(2)):
    fc_layer = nn.Linear(in_feature, out_feature)
    return layer_init(fc_layer, std)


def make_mlp(feature_list, act_last=True, act_func=nn.ReLU, std_last=0.01):
    layers = []
    for i in range(len(feature_list)-2):
        fc_layer = make_fc(feature_list[i], feature_list[i+1])
        layers.append(fc_layer)
        layers.append(act_func())
    layers.append(make_fc(feature_list[-2], feature_list[-1], std=std_last))
    if act_last:
        layers.append(act_func())
    layers = nn.Sequential(*layers)    
    return layers