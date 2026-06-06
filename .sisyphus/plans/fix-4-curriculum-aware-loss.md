# Fix 4: Curriculum-Aware Loss Weighting

## 问题描述

**严重程度**: MEDIUM

当前latent reconstruction loss对所有环境等权重处理。当地形难度上升时：
- 简单地形（平地）的loss快速下降
- 复杂地形（楼梯、间隙）的loss突然上升
- 复杂地形样本在mini-batch中占比增加，导致整体loss波动

**代码位置**: `rsl_rl/algorithms/ppo_ts_depth.py:232-235`

```python
latent_reconstruction_loss = nn.functional.mse_loss(
    latent, latent_targets)

loss = latent_reconstruction_loss
```

## 修复方案

根据地形难度和课程进度，动态调整loss权重。

### 方案A: 基于地形难度的loss加权（推荐）

**文件**: `rsl_rl/algorithms/ppo_ts_depth.py`

在学生更新中，根据环境的地形难度调整loss权重：

```python
# 在学生更新循环中
# 获取每个环境的地形难度（0-1范围）
# 假设地形难度可以通过 terrain_levels 获取
terrain_difficulty = self.storage.terrain_levels[:self.num_student] / self.num_terrain_rows

# 计算loss时，根据地形难度调整权重
latent_reconstruction_loss = nn.functional.mse_loss(
    latent, latent_targets, reduction='none'
)

# 对于高难度地形，降低loss权重（允许更多探索）
# 对于低难度地形，提高loss权重（精确匹配）
difficulty_weight = 1.0 - 0.5 * terrain_difficulty  # 高难度权重0.5，低难度权重1.0
latent_reconstruction_loss = (latent_reconstruction_loss * difficulty_weight).mean()
```

### 方案B: 基于课程变化率的自适应权重

**文件**: `rsl_rl/algorithms/ppo_ts_depth.py`

跟踪课程变化率，在变化快时降低权重：

```python
# 在 PPO_TSDepth.__init__ 中添加
self.prev_terrain_levels = None
self.curriculum_change_smoothing = 0.9

# 在 update 方法中
# 计算课程变化率
current_terrain_levels = self.storage.terrain_levels[:self.num_student].float().mean()
if self.prev_terrain_levels is not None:
    curriculum_change = abs(current_terrain_levels - self.prev_terrain_levels)
else:
    curriculum_change = 0.0

# 平滑变化率
if not hasattr(self, 'smoothed_curriculum_change'):
    self.smoothed_curriculum_change = curriculum_change
else:
    self.smoothed_curriculum_change = (
        self.curriculum_change_smoothing * self.smoothed_curriculum_change + 
        (1 - self.curriculum_change_smoothing) * curriculum_change
    )

# 根据变化率调整loss权重
loss_weight = 1.0 / (1.0 + self.smoothed_curriculum_change * 10)

# 应用到loss
latent_reconstruction_loss = latent_reconstruction_loss * loss_weight

self.prev_terrain_levels = current_terrain_levels
```

### 方案C: 渐进式loss权重预热

**文件**: `rsl_rl/algorithms/ppo_ts_depth.py`

在课程更新后的一段时间内，逐渐增加loss权重：

```python
# 在 PPO_TSDepth.__init__ 中添加
self.curriculum_update_step = 0
self.warmup_steps = 50  # 预热步数

# 在 update 方法中
# 检测课程是否最近更新
if self.storage.curriculum_updated:
    self.curriculum_update_step = 0
    self.storage.curriculum_updated = False

# 计算预热权重
if self.curriculum_update_step < self.warmup_steps:
    warmup_weight = self.curriculum_update_step / self.warmup_steps
else:
    warmup_weight = 1.0

self.curriculum_update_step += 1

# 应用到loss
latent_reconstruction_loss = latent_reconstruction_loss * warmup_weight
```

### 关键参数

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `difficulty_weight_range` | [0.5, 1.0] | 高难度到低难度的权重范围 |
| `curriculum_change_smoothing` | 0.9 | 变化率平滑系数 |
| `warmup_steps` | 50 | 预热步数 |

### 验证方法

1. 观察课程更新后的loss曲线
2. 期望：loss在课程更新后不会突然跳升
3. 监控loss权重变化

### 预期效果

- 课程更新时loss更平滑
- 复杂地形的学习更渐进
- 整体loss曲线更稳定
