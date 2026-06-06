# Latent Reconstruction Loss 不稳定性修复计划

## 问题概述

`go2_ts_depth` 任务在训练时，latent reconstruction loss 在地形难度上升后突然上升并波动很大。

## 问题源分析

| # | 问题 | 严重程度 | 影响 |
|---|------|----------|------|
| 1 | 目标分布突变（无EMA） | HIGH | 特权编码器输出随地形变化而突变 |
| 2 | 教师/学生学习率不匹配 | HIGH | 教师更新速度是学生的5倍 |
| 3 | 重置时隐藏状态清零 | MEDIUM | 环境重置后丢失时序上下文 |
| 4 | 无课程感知的loss加权 | MEDIUM | 复杂地形样本主导loss |
| 5 | 学生环境子集偏差 | LOW | 1500/4000环境不能代表整体分布 |

## 修复计划

### Fix 1: EMA Teacher (推荐优先实现)

**文件**: `rsl_rl/algorithms/ppo_ts_depth.py`

- 维护特权编码器的EMA副本
- 学生学习匹配EMA版本的输出
- 消除目标分布突变

### Fix 2: Teacher-Student LR Decoupling

**文件**: `rsl_rl/algorithms/ppo_ts_depth.py`

- 方案A: 学生训练期间冻结特权编码器 (推荐)
- 方案B: 特权编码器使用独立学习率
- 方案C: 特权编码器梯度裁剪加强

### Fix 3: Gradual Hidden State Decay

**文件**: `rsl_rl/modules/depth_history_encoder.py`

- 渐进衰减隐藏状态（而非清零）
- 添加重置掩码用于loss计算
- 隐藏状态预热机制

### Fix 4: Curriculum-Aware Loss Weighting

**文件**: `rsl_rl/algorithms/ppo_ts_depth.py`

- 基于地形难度的loss加权
- 基于课程变化率的自适应权重
- 渐进式loss权重预热

### Fix 5: Student Environment Subset Bias

**文件**: `rsl_rl/runners/ts_depth_runner.py`, `legged_gym/envs/go2/go2_ts_depth/go2_ts_depth_config.py`

- 增加学生环境数量
- 动态选择学生环境
- 存储中记录地形难度

## 实施建议

### 优先级

1. **Fix 1 + Fix 2**: 这两个问题最严重，建议首先实现
2. **Fix 3**: 简单修改，可以快速实现
3. **Fix 4**: 需要更多测试，建议在Fix 1+2之后实现
4. **Fix 5**: 影响较小，可以在其他fix验证后实现

### 验证方法

每个fix都应该：
1. 独立测试，观察loss曲线变化
2. 对比修复前后的loss标准差
3. 监控地形难度上升时的loss稳定性

### 预期效果

- Fix 1: 消除目标分布突变 → loss曲线更平滑
- Fix 2: 减少目标快速变化 → loss振荡减少
- Fix 3: 保持时序上下文 → 重置后loss不会跳升
- Fix 4: 平滑课程过渡 → 课程更新时loss更稳定
- Fix 5: 更好的分布覆盖 → 梯度估计更准确
