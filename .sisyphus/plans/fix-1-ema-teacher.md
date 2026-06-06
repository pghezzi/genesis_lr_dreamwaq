# Fix 1: EMA Teacher for Stable Latent Targets

## 问题描述

**严重程度**: HIGH

当机器人地形难度上升时，特权编码器（privilege encoder）的输出分布会突然变化，因为特权观测中包含了地形高度测量值（171维）。学生编码器试图匹配的目标会突然改变，导致latent reconstruction loss剧烈波动。

**代码位置**: `rsl_rl/algorithms/ppo_ts_depth.py:227-233`

```python
with torch.no_grad():
    unpadded_student_privileged_obs = unpad_trajectories(student_privileged_obs_batch, masks_batch)
    latent_targets = self.actor_critic.privilege_encoder(unpadded_student_privileged_obs)

latent_reconstruction_loss = nn.functional.mse_loss(latent, latent_targets)
```

## 修复方案

实现EMA（Exponential Moving Average）教师编码器，提供稳定的目标分布。

### 核心思路

- 维护一个特权编码器的EMA副本
- 教师PPO更新后，用EMA方式更新副本（而不是直接拷贝）
- 学生编码器学习匹配EMA版本的输出，而非波动的原始输出

### 实现步骤

#### Step 1: 添加EMA编码器

**文件**: `rsl_rl/algorithms/ppo_ts_depth.py`

在 `__init__` 方法中添加：

```python
import copy

# 在 __init__ 中添加
self.ema_decay = 0.995  # EMA衰减系数，越接近1越稳定
self.ema_privilege_encoder = copy.deepcopy(self.actor_critic.privilege_encoder)
# 冻结EMA编码器，不参与梯度计算
for param in self.ema_privilege_encoder.parameters():
    param.requires_grad = False
```

#### Step 2: 更新EMA编码器

**文件**: `rsl_rl/algorithms/ppo_ts_depth.py`

在 `update` 方法中，教师PPO更新完成后，学生更新之前添加：

```python
# === Teacher PPO update 完成后 ===
# 更新EMA编码器
with torch.no_grad():
    for ema_param, param in zip(self.ema_privilege_encoder.parameters(), 
                                 self.actor_critic.privilege_encoder.parameters()):
        ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
```

#### Step 3: 使用EMA编码器生成学生目标

**文件**: `rsl_rl/algorithms/ppo_ts_depth.py`

修改学生更新部分（约第227-233行）：

```python
# 原代码:
# with torch.no_grad():
#     unpadded_student_privileged_obs = unpad_trajectories(student_privileged_obs_batch, masks_batch)
#     latent_targets = self.actor_critic.privilege_encoder(unpadded_student_privileged_obs)

# 修改为:
with torch.no_grad():
    unpadded_student_privileged_obs = unpad_trajectories(student_privileged_obs_batch, masks_batch)
    latent_targets = self.ema_privilege_encoder(unpadded_student_privileged_obs)
```

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `ema_decay` | 0.995 | 控制目标更新速度。0.995=慢更新（稳定），0.9=快更新（适应快） |

### 验证方法

1. 训练时观察latent reconstruction loss曲线
2. 期望：loss曲线更平滑，地形难度上升时不会突然跳升
3. 对比：对比修复前后loss的标准差

### 预期效果

- 消除因地形变化导致的目标分布突变
- latent reconstruction loss曲线更平滑
- 学生编码器学习更稳定
