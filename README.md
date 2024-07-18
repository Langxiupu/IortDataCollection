- [IortDataCollection](#iortdatacollection)
  - [一、技术栈](#一技术栈)
  - [二、代码目录](#二代码目录)
  - [三、代码进度](#三代码进度)
  - [四、代码待完成部分](#四代码待完成部分)

# IortDataCollection
## 一、技术栈
PPO+gymnasium
- PPO：未调用Stable_Baselines等成熟的RL代码库，自定义了神经网络结构，增加了action mask，并提供了一版PPO的clean implementation。
- gymnasium：通过继承gymnasium.Env自定义了环境，可支持multi-envs的包装。

## 二、代码目录
```sh
- discrete
|-- __init__.py
|-- agent.py
|-- config.py
|-- env.py
|-- train.py
|-- utils
```

- agent.py: 定义了智能体的神经网络结构。
- config.py: 保存了训练、环境、网络的配置参数。
- env.py: 实现了无人机数据采集的环境。
- train.py: 训练的入口函数。
- utils：其目录下的common_tools.py保存了功能函数，例如mlp生成等。

## 三、代码进度
目前已实现了：
- 参数配置、环境生成等基本的训练框架；
- 状态/动作空间的定义与实现；
- 自定义神经网络结构的实现；
- 动作掩码生成的实现。

## 四、代码待完成部分
1. 训练过程：包含经验采集与模型更新。
2. 无人机飞行与数据采集的环境实现。
