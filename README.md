# 基于MCEO的无梯度强化学习算法


包括了基于MCEO的策略搜索与基于模型的强化学习框架下的MCEO优化两部分算法
其中环境的配置如下：
```
conda env create -f environment.yml
 ```
## 实验

### 实验一：离散线性系统控制

```
python 
```

**CAP**

```
python cap-pets/run_cap_pets.py --algo cem --env HalfCheetah-v3 --cost_lim 152 \
--cost_constrained --penalize_uncertainty --learn_kappa --seed 1
```

**CAP with fixed kappa**

```
python cap-pets/run_cap_pets.py --algo cem --env HalfCheetah-v3 --cost_lim 152 \
--cost_constrained --penalize_uncertainty --kappa 1.0 --seed 1
```

**CCEM**

```
python cap-pets/run_cap_pets.py --algo cem --env HalfCheetah-v3 --cost_lim 152 \
--cost_constrained --seed 1
```

**CEM**

```
python cap-pets/run_cap_pets.py --algo cem --env HalfCheetah-v3 --cost_lim 152 \
--seed 1
```

### The commands for the PlaNet experiment on the CarRacing environment are:

**CAP**

```
python cap-planet/run_cap_planet.py --env CarRacingSkiddingConstrained-v0 \
--cost-limit 0 --binary-cost \
--cost-constrained --penalize-uncertainty \
--learn-kappa --penalty-kappa 0.1 \
--id CarRacing-cap --seed 1
```

**CAP with fixed kappa**

```
python cap-planet/run_cap_planet.py --env CarRacingSkiddingConstrained-v0 \
--cost-limit 0 --binary-cost \
--cost-constrained --penalize-uncertainty \
--penalty-kappa 1.0 \
--id CarRacing-kappa1 --seed 1
```

**CCEM**

```
python cap-planet/run_cap_planet.py --env CarRacingSkiddingConstrained-v0 \
--cost-limit 0 --binary-cost \
--cost-constrained \
--id CarRacing-ccem --seed 1
```

**CEM**

```
python cap-planet/run_cap_planet.py --env CarRacingSkiddingConstrained-v0 \
--cost-limit 0 --binary-cost \
--id CarRacing-cem --seed 1
```

## Contact
If you have any questions regarding the code or paper, feel free to contact jasonyma@seas.upenn.edu or open an issue on this repository.

## Acknowledgement
This repository contains code adapted from the 
following repositories: [PETS](https://github.com/quanvuong/handful-of-trials-pytorch) and
[PlaNet](https://github.com/Kaixhin/PlaNet). We thank the
 authors and contributors for open-sourcing their code.  
