# 基于MCEO的无梯度强化学习算法


包括了基于MCEO的策略搜索与基于模型的强化学习框架下的MCEO优化两部分算法
其中环境的配置如下：
```
conda env create -f environment.yml
 ```
## 实验部分

### 实验一：离散线性系统控制

```
python CEM_LQR.py
```

### 实验二：倒立摆控制

```
python cap-pets/run_cap_pets.py --algo cem --env Pendulum-v0 
```

配置环境主要是配置gym与mujoco-py,此外还需手动下载mujoco210，链接如下：
https://blog.csdn.net/weixin_51844581/article/details/128454472
