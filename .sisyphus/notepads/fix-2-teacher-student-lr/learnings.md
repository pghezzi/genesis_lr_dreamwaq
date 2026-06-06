## 2026-06-06 Task: fix-2-teacher-student-lr (方案B)

### 实现方案
采用方案B：特权编码器使用独立学习率

### 修改内容
**文件**: `rsl_rl/algorithms/ppo_ts_depth.py`

**`__init__` 改动**：
- 将 `privilege_encoder` 从 `teacher_params` 中移除
- 创建带不同学习率组的 `teacher_optimizer`：
  - teacher_params (actor, critic, std): `learning_rate` (1e-3)
  - privilege_encoder: `encoder_lr * 0.5` (1e-4)

**`update()` 改动**：
- KL adaptation 只更新 teacher_params 的学习率（param_groups[0]）

### 学习率对比
| 组件 | 原始 | 修改后 |
|------|------|--------|
| actor, critic, std | 1e-3 | 1e-3 (不变) |
| privilege_encoder | 1e-3 | 1e-4 (降低10倍) |
| depth_history_encoder | 2e-4 | 2e-4 (不变) |

### 预期效果
- 特权编码器更新速度降低，目标分布变化更平缓
- 学生编码器有更稳定的学习目标
- latent reconstruction loss 振荡减少
