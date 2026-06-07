# Fix 3: Gradual Hidden State Decay on Reset

## 问题描述

**严重程度**: MEDIUM

当地形难度上升时，机器人失败率增加，导致频繁重置。每次重置都会将GRU隐藏状态清零，丢失时序上下文。这使得学生编码器在复杂地形上无法产生一致的latent向量。

**代码位置**: `rsl_rl/modules/depth_history_encoder.py:120-128`

```python
def reset_hidden_states(self, dones=None):
    if self.hidden_states is None:
        return
    states = self.hidden_states if isinstance(self.hidden_states, (tuple, list)) else (self.hidden_states,)
    for hidden_state in states:
        hidden_state[..., dones, :] = 0.0
```

## 修复方案

使用渐进式隐藏状态衰减，而非直接清零。同时在loss计算中考虑重置状态。

### 方案A: 隐藏状态衰减（推荐）

**文件**: `rsl_rl/modules/depth_history_encoder.py`

修改 `reset_hidden_states` 方法：

```python
def reset_hidden_states(self, dones=None, decay_factor=0.5):
    """Reset hidden states with gradual decay instead of hard zeroing.
    
    Args:
        dones: Boolean mask of which environments have reset
        decay_factor: How much to decay (0=zero, 1=no decay). Default 0.5.
    """
    if self.hidden_states is None:
        return
    states = self.hidden_states if isinstance(self.hidden_states, (tuple, list)) else (self.hidden_states,)
    for hidden_state in states:
        # 渐进衰减而非直接清零
        hidden_state[..., dones, :] *= decay_factor
```

### 方案B: 添加重置掩码用于loss计算

**文件**: `rsl_rl/algorithms/ppo_ts_depth.py`

在学生更新中，根据重置状态调整loss权重：

```python
# 在学生更新循环中
# 获取重置掩码（哪些环境最近被重置过）
reset_mask = masks_batch.sum(dim=0) > 0  # 轨迹中有重置的环境

# 计算loss时，降低最近重置环境的权重
latent_reconstruction_loss = nn.functional.mse_loss(
    latent, latent_targets, reduction='none'
)

# 为重置环境降低权重
if reset_mask.any():
    # 最近重置的环境，loss权重降低50%
    weight = torch.ones_like(latent_reconstruction_loss)
    weight[reset_mask] = 0.5
    latent_reconstruction_loss = (latent_reconstruction_loss * weight).mean()
else:
    latent_reconstruction_loss = latent_reconstruction_loss.mean()
```

### 方案C: 隐藏状态预热

**文件**: `rsl_rl/modules/depth_history_encoder.py`

添加预热机制，在重置后逐渐恢复隐藏状态的使用：

```python
def __init__(self, ...):
    # ... 原有代码 ...
    self.reset_counter = None  # 跟踪每个环境重置后的步数
    
def forward(self, observation, depth_image, hidden_states=None, masks=None):
    # ... 原有代码 ...
    
    # 如果是实时模式（非batch），应用预热
    if not batch_mode and self.reset_counter is not None:
        # 重置后5步内，混合使用新旧隐藏状态
        warmup_steps = 5
        warmup_mask = (self.reset_counter < warmup_steps).float()
        if warmup_mask.any():
            # 对于预热中的环境，逐渐引入新状态
            self.hidden_states = (
                self.hidden_states[0] * (1 - warmup_mask * 0.5),
                self.hidden_states[1] * (1 - warmup_mask * 0.5)
            )
            self.reset_counter += 1
    
    return latent_output
```

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `decay_factor` | 0.5 | 隐藏状态保留比例 |
| `warmup_steps` | 5 | 预热步数 |

### 验证方法

1. 观察地形难度上升时loss的波动
2. 期望：loss在环境重置后不会突然跳升
3. 监控隐藏状态的范数变化

### 预期效果

- 环境重置后保持部分时序上下文
- 学生编码器在复杂地形上更稳定
- loss曲线更平滑
