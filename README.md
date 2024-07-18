- [IortDataCollection](#iortdatacollection)
  - [技术栈](#技术栈)
  - [代码进度](#代码进度)
  - [代码待完成部分](#代码待完成部分)

# IortDataCollection
## 技术栈
PPO+gymnasium
- PPO：未调用Stable_Baselines等成熟的RL代码库，自定义了神经网络结构，增加了action mask，并提供了一版PPO的clean implementation。
- gymnasium：通过继承gymnasium.Env自定义了环境，可支持multi-envs的包装。

## 代码进度
目前已实现了：
- 参数配置、环境生成等基本的训练框架；
- 状态/动作空间的定义与实现；
- 自定义神经网络结构的实现；
- 动作掩码生成的实现。

## 代码待完成部分
1. 训练过程：包含经验采集与模型更新。
2. 无人机飞行与数据采集的环境实现。
