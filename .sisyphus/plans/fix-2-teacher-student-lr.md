# Fix 2: Teacher-Student Learning Rate Decoupling

## 问题描述

**严重程度**: HIGH

教师策略更新速度（learning_rate=1e-3）比学生编码器（encoder_lr=2e-4）快5倍。教师PPO更新会修改特权编码器的权重，导致学生编码器面对一个快速移动的目标，产生振荡。

**代码位置**: `rsl_rl/algorithms/ppo_ts_depth.py:66-74`

```python
self.teacher_params = list(self.actor_critic.actor.parameters()) + \
                    list(self.actor_critic.privilege_encoder.parameters()) + \
                    list(self.actor_critic.critic.parameters()) + \
                    [self.actor_critic.std]
self.teacher_optimizer = optim.Adam(self.teacher_params, lr=learning_rate)

self.student_optimizer = optim.Adam(
    self.student_params, lr=self.encoder_lr)
```

## 修复方案

将特权编码器从教师优化器中分离，使用独立的学习率，或在学生训练阶段冻结特权编码器。

### 方案A: 学生训练期间冻结特权编码器（推荐）

#### 实现步骤

**文件**: `rsl_rl/algorithms/ppo_ts_depth.py`

在学生更新之前冻结特权编码器，更新之后解冻：

```python
# === 在学生更新循环之前 ===
# 冻结特权编码器
for param in self.actor_critic.privilege_encoder.parameters():
    param.requires_grad = False

# === 学生更新循环 ===
generator = self.storage.teacher_mini_batch_generator(...)
for ... in generator:
    # ... 学生更新代码 ...
    pass

# === 学生更新之后 ===
# 解冻特权编码器
for param in self.actor_critic.privilege_encoder.parameters():
    param.requires_grad = True
```

### 方案B: 特权编码器使用独立学习率

#### 实现步骤

**文件**: `rsl_rl/algorithms/ppo_ts_depth.py`

将特权编码器从教师参数中分离：

```python
# 修改 __init__ 中的参数分组
self.teacher_params = list(self.actor_critic.actor.parameters()) + \
                    list(self.actor_critic.critic.parameters()) + \
                    [self.actor_critic.std]
# 特权编码器使用独立的优化器
self.privilege_encoder_params = list(self.actor_critic.privilege_encoder.parameters())
self.privilege_encoder_optimizer = optim.Adam(
    self.privilege_encoder_params, lr=self.encoder_lr * 0.5)  # 比学生略快

# 或者将特权编码器加入教师优化器但使用不同的学习率组
self.teacher_optimizer = optim.Adam([
    {'params': list(self.actor_critic.actor.parameters()) + 
               list(self.actor_critic.critic.parameters()) + 
               [self.actor_critic.std], 'lr': learning_rate},
    {'params': list(self.actor_critic.privilege_encoder.parameters()), 
     'lr': self.encoder_lr * 0.5}
])
```

### 方案C: 特权编码器梯度裁剪加强

**文件**: `rsl_rl/algorithms/ppo_ts_depth.py`

对特权编码器使用更严格的梯度裁剪：

```python
# 在教师更新后，学生更新前
nn.utils.clip_grad_norm_(
    self.actor_critic.privilege_encoder.parameters(), 
    self.max_grad_norm * 0.5  # 更严格的裁剪
)
```

### 关键参数

| 参数 | 建议值 | 说明 |
|------|--------|------|
| 特权编码器学习率 | `encoder_lr * 0.5` | 比学生编码器慢 |
| 特权编码器梯度裁剪 | `max_grad_norm * 0.5` | 防止大幅更新 |

### 验证方法

1. 观察latent reconstruction loss曲线
2. 期望：loss振荡减少，收敛更稳定
3. 监控特权编码器权重变化幅度

### 预期效果

- 减少目标分布的快速变化
- 学生编码器有更稳定的学习目标
- loss曲线更平滑
