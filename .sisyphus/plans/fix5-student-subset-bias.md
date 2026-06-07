# Fix 5: Student Environment Subset Bias Mitigation

## 问题描述

**严重程度**: LOW-MEDIUM

学生编码器只使用 `num_camera_envs=1500` 个环境（总环境数 `num_envs=4000`）。当课程更新时：
- 这1500个环境可能不能代表完整的地形难度分布
- 导致梯度估计有偏，增加loss波动

**代码位置**: `rsl_rl/runners/ts_depth_runner.py:91-96`

```python
self.storage_full = RolloutStorageTSDepth(
    self.env.num_envs, self.env.num_student, self.num_steps_per_env,
    [self.env.num_obs], [self.env.num_privileged_obs],
    self.env.depth_image_features_shape, [self.env.num_critic_obs],
    [self.env.num_actions], self.device)
```

## 修复方案

### 方案A: 增加学生环境数量（推荐）

**文件**: `legged_gym/envs/go2/go2_ts_depth/go2_ts_depth_config.py`

增加 `num_camera_envs` 的比例：

```python
# 原配置
num_camera_envs = 1500  # 37.5% 的环境

# 修改为
num_camera_envs = 3000  # 75% 的环境
```

**权衡**: 增加内存使用和计算开销。

### 方案B: 动态选择学生环境

**文件**: `rsl_rl/runners/ts_depth_runner.py`

根据地形难度分布，动态选择哪些环境使用深度相机：

```python
# 在 learn 方法中
# 每隔一定步数重新选择学生环境
if it % 100 == 0:
    # 获取地形难度分布
    terrain_levels = self.env.simulator.terrain_levels
    
    # 按难度分层采样，确保学生环境覆盖所有难度
    num_difficulty_bins = 5
    bin_size = self.env.num_envs // num_difficulty_bins
    student_per_bin = self.env.num_student // num_difficulty_bins
    
    student_indices = []
    for bin_idx in range(num_difficulty_bins):
        start_idx = bin_idx * bin_size
        end_idx = start_idx + bin_size
        bin_envs = torch.arange(start_idx, end_idx, device=self.device)
        
        # 从每个难度级别中随机选择
        selected = bin_envs[torch.randperm(len(bin_envs))[:student_per_bin]]
        student_indices.append(selected)
    
    self.student_env_indices = torch.cat(student_indices)
    
    # 更新存储
    self.storage_full.student_env_indices = self.student_env_indices
```

### 方案C: 存储中记录地形难度

**文件**: `rsl_rl/storage/rollout_storage_ts_depth.py`

在存储中添加地形难度信息，用于loss加权：

```python
class RolloutStorageTSDepth(RolloutStorage):
    def __init__(self, ...):
        # ... 原有代码 ...
        self.terrain_levels = torch.zeros(num_transitions_per_env, num_student, device=self.device)
        self.curriculum_updated = False
    
    def add_transitions(self, transition: Transition):
        # ... 原有代码 ...
        if hasattr(transition, 'terrain_levels') and transition.terrain_levels is not None:
            self.terrain_levels[self.step].copy_(transition.terrain_levels[:self.num_student])
    
    def clear(self):
        super().clear()
        # 保留地形难度信息（不清零）
```

**文件**: `rsl_rl/algorithms/ppo_ts_depth.py`

在学生更新中使用地形难度信息：

```python
# 获取地形难度
terrain_levels = self.storage.terrain_levels.mean(dim=0)  # 平均地形难度

# 用于loss加权
difficulty_weight = 1.0 / (1.0 + terrain_levels * 0.1)
latent_reconstruction_loss = latent_reconstruction_loss * difficulty_weight.mean()
```

### 关键参数

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `num_camera_envs` | 3000 | 增加学生环境数量 |
| `num_difficulty_bins` | 5 | 难度分层数量 |
| `resample_interval` | 100 | 重新采样间隔 |

### 验证方法

1. 观察学生环境的地形分布
2. 期望：学生环境覆盖所有难度级别
3. 监控梯度方差

### 预期效果

- 学生环境更好地代表整体分布
- 梯度估计更准确
- loss曲线更平滑
