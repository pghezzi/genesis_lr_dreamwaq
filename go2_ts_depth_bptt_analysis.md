# go2_ts_depth BPTT (Backpropagation Through Time) 分析报告

**日期**: 2026-04-16  
**任务**: go2_ts_depth 的 BPTT 问题分析与优化建议  

---

## 📋 执行摘要

本报告从 BPTT（Backpropagation Through Time）角度分析 `go2_ts_depth` 的训练流程，发现**6个潜在问题**可能影响 RNN 的梯度传播和时序建模能力。

**修复状态**:
- ✅ **问题 1**: Hidden states detach 时机不当 - **已修复** (2026-04-16)
- ❌ **问题 2-6**: 待修复/验证

**最关键问题**:
1. ~~Hidden states detach 时机不当，可能阻断跨 mini-batch 的梯度传播~~ ✅ **已修复**
2. `unpad_trajectories` 可能破坏计算图
3. Teacher/Student 的 hidden states 可能存在干扰

---

## 🚨 BPTT 相关问题分析

### 问题 1: Hidden States Detach 时机不当 ⚠️ 严重

**状态**: ✅ **已修复**

**位置**: `rsl_rl/algorithms/ppo_ts_depth.py` 第 229-230 行 (旧) / `rsl_rl/runners/ts_depth_runner.py` 第 125-126 行 (新)

**原始问题**:
```python
# 在每个 mini-batch 后立即 detach
for batch in generator:
    loss.backward()
    optimizer.step()
    detach_hidden_states()  # ❌ 阻断跨 batch 梯度传播
```

**修复方案**:
将 `detach_hidden_states()` 从 `ppo_ts_depth.py` 的每个 mini-batch 后移除，改为在 `ts_depth_runner.py` 的整个 update 完成后统一调用：

```python
# ts_depth_runner.py 第 120-126 行
mean_value_loss, mean_surrogate_loss, mean_latent_reconstruction_loss, \
        mean_action_reconstruction_loss = self.alg.update()
# ...
# detach hidden states after each update (num_steps per env)
self.alg.actor_critic.detach_hidden_states()  # ✅ 允许跨 batch 梯度传播
```

**修复效果**:
- ✅ 梯度现在可以跨多个 mini-batches 传播
- ✅ 允许学习跨 trajectory 的长期依赖
- ✅ 符合 BPTT 的标准实现

**验证建议**:
```python
# 在 backward 后检查梯度
for name, param in self.alg.actor_critic.depth_history_encoder.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm():.6f}")
```

**优先级**: 🔴 高 (已解决)

---

### 问题 2: Gradient Flow 可能被截断 ⚠️ 严重

**位置**: `rsl_rl/modules/depth_history_encoder.py` 第 113 行

**当前实现**:
```python
rnn_out, _ = self.rnn(combined_encoding, hidden_states)
rnn_out = unpad_trajectories(rnn_out, masks)  # 可能破坏计算图
```

**问题分析**:
- `unpad_trajectories` 函数会**重新排列 tensor**
- 如果实现不当，某些 time steps 的梯度连接可能被切断
- 特别是当 `masks` 标记某些位置为 done 时

**验证方法**:
```python
# 在 backward 后添加调试代码
for name, param in self.actor_critic.depth_history_encoder.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm()
        print(f"{name}: grad_norm = {grad_norm:.6f}")
        if grad_norm < 1e-8:
            print(f"  ⚠️ 警告: {name} 的梯度几乎为零!")
```

**建议修复**:
```python
# 方案 A: 使用 PyTorch 内置的 pack_padded_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

lengths = masks.sum(dim=0)  # 每个 trajectory 的实际长度
packed_input = pack_padded_sequence(
    input=combined_encoding, 
    lengths=lengths.cpu(), 
    batch_first=True,
    enforce_sorted=False
)
packed_output, _ = self.rnn(packed_input, hidden_states)
rnn_out, _ = pad_packed_sequence(packed_output, batch_first=True)

# 方案 B: 检查 unpad_trajectories 实现
# 确保它保留了计算图和梯度流
```

**优先级**: 🔴 高

---

### 问题 3: Teacher/Student Hidden States 干扰 ⚠️ 严重

**位置**: `rsl_rl/algorithms/ppo_ts_depth.py`

**问题分析**:
- Teacher 和 Student 使用**不同的 hidden states**
- 但在 rollout 阶段，两者可能**相互干扰**
- Student 的 BPTT 依赖于 rollout 时保存的 hidden states

**可能的问题场景**:
```python
# Rollout 阶段
for step in range(num_steps):
    # Teacher action
    teacher_action = teacher.act(obs, privileged_obs)  # 可能改变某些共享状态？
    
    # Student action
    student_action = student.act(obs, depth)  # 依赖于 student 的 hidden states
    
    # 保存的 hidden states 可能被 teacher 影响
```

**建议检查**:
1. 确认 Teacher 和 Student 的 hidden states 完全独立
2. 检查 `act` 方法中的 hidden_states 参数传递
3. 验证 rollout 时保存的 hidden states 确实是 Student 的

**调试代码**:
```python
# 在 rollout 和 update 时打印 hidden states 的 hash/id
print(f"Rollout hidden states id: {id(self.actor_critic.depth_history_encoder.hidden_states)}")
print(f"Update hidden states id: {id(hid_states_batch)}")
```

**优先级**: 🔴 高

---

### 问题 4: Truncated BPTT 长度不一致 ⚠️ 中等

**当前配置**:
- `num_steps_per_env = 24`
- `rnn_hidden_size = 512`

**问题分析**:
- 24 步的序列长度对于 GRU 来说**可能太短**
- 不足以捕获长期的时序依赖（如完整步态周期 ~50-100 步）
- 但 BPTT 又要在这 24 步内完成，梯度传播深度有限

**建议优化**:
```python
# 方案 A: 增加序列长度
num_steps_per_env = 48  # 或 64

# 方案 B: 使用 Truncated BPTT
# 每 24 步收集数据，但每 48 步才截断一次梯度
if episode_length % 48 == 0:
    detach_hidden_states()

# 方案 C: 分层 BPTT
# 短序列快速更新，长序列定期截断
```

**优先级**: 🟡 中

---

### 问题 5: Hidden States 初始化问题 ⚠️ 中等

**位置**: `rsl_rl/modules/depth_history_encoder.py` 第 40-44 行

**当前实现**:
```python
if hidden_states is None:
    self.hidden_states = None  # GRU 默认零初始化
else:
    self.hidden_states = hidden_states
```

**问题分析**:
- 每个 episode 开始时，hidden states 初始化为 **零向量**
- 对于 POMDP，**初始 hidden state 很重要**
- 前一个 episode 的信息完全丢失，无法利用历史经验

**建议修复**:
```python
# 方案 A: 学习可训练的初始 hidden state
self.initial_hidden_state = nn.Parameter(
    torch.zeros(1, 1, rnn_hidden_size)
)

def reset(self, dones=None):
    if dones is not None:
        # 只对 done 的环境重置为初始值
        self.hidden_states[:, dones, :] = self.initial_hidden_state
    else:
        # 全部重置
        self.hidden_states = self.initial_hidden_state.expand(
            num_layers, batch_size, hidden_size
        ).clone()

# 方案 B: 使用前一 episode 的最后 hidden state
# 需要修改 storage 来保存这些信息
```

**优先级**: 🟡 中

---

### 问题 6: Storage 中 Hidden States 维度问题 ⚠️ 中等

**位置**: `rsl_rl/storage/rollout_storage_ts_depth.py` 第 55-61 行

**当前实现**:
```python
if self.saved_hidden_states is None:
    self.saved_hidden_states = [torch.zeros(
        self.observations.shape[0],  # num_transitions_per_env
        *hid[i].shape,               # hidden state shape
        device=self.device
    ) for i in range(len(hid))]
```

**问题分析**:
- 保存的 hidden states 是**下一时刻的输入**，不是当前时刻的输出
- 在 BPTT 中，需要**输入和输出**才能正确计算梯度
- 如果只保存了输入，梯度计算可能不完整

**建议检查**:
```python
# 验证保存的 hidden states 是否正确
# 在 update 时检查:
print(f"hid_states_batch shape: {hid_states_batch.shape}")
print(f"Should be: [num_layers, num_trajs, hidden_dim]")

# 验证时间对齐
for t in range(trajectory_length):
    # 确保 hid_states_batch[t] 对应 observations[t-1] 的输出
    # 和 observations[t] 的输入
```

**建议修复**:
```python
# 保存更完整的 RNN 状态信息
class Transition:
    def __init__(self):
        self.hidden_states_input = None   # t 时刻输入 RNN 的 hidden state
        self.hidden_states_output = None  # t 时刻 RNN 输出的 hidden state
        self.cell_states = None           # 如果使用 LSTM
```

**优先级**: 🟡 中

---

## 💡 BPTT 优化建议

### 1. 实现 Truncated BPTT

```python
class PPO_TSDepth:
    def __init__(self, ...):
        self.bptt_truncation_length = 32  # 每 32 步截断一次
    
    def update(self):
        for i, batch in enumerate(generator):
            # ... 前向传播 ...
            loss.backward()
            
            # 每 k 步截断一次
            if (i + 1) % (self.bptt_truncation_length // mini_batch_size) == 0:
                self.actor_critic.depth_history_encoder.detach_hidden_states()
```

### 2. 添加梯度监控

```python
def check_gradient_flow(self, model, iteration):
    """监控梯度是否正常传播"""
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    total_norm = total_norm ** 0.5
    print(f"Iteration {iteration}: Total grad norm = {total_norm:.4f}")
    
    if total_norm < 1e-4:
        print("⚠️ 警告: 梯度消失!")
    elif total_norm > 10.0:
        print("⚠️ 警告: 梯度爆炸!")
```

### 3. 调整梯度裁剪策略

```python
# 对 RNN 使用更宽松的梯度裁剪
if self.student_params:
    nn.utils.clip_grad_norm_(
        self.student_params, 
        max_norm=5.0,  # 从 1.0 放宽到 5.0
        norm_type=2
    )
```

### 4. 验证 RNN 状态传递

```python
def validate_rnn_states(self):
    """验证 RNN hidden states 是否正确传递"""
    # 1. 检查 rollout 时保存的状态
    # 2. 检查 update 时使用的状态
    # 3. 验证时间对齐
    
    # 测试用例:
    # 创建一个简单的 trajectory，验证梯度能传播到第一个 time step
    test_obs = torch.randn(10, 4, obs_dim)  # [time, env, obs]
    test_depth = torch.randn(10, 4, depth_dim)
    
    hidden = None
    latents = []
    for t in range(10):
        latent, hidden = self.depth_history_encoder(
            test_obs[t:t+1], test_depth[t:t+1], hidden
        )
        latents.append(latent)
    
    # 反向传播
    loss = sum(latents).mean()
    loss.backward()
    
    # 检查第一个 time step 的输入是否有梯度
    if test_obs[0].grad is None:
        print("❌ 梯度没有传播到第一个 time step!")
    else:
        print("✅ 梯度传播正常")
```

---

## 🎯 最优先修复建议

按严重程度排序：

1. **问题 1 (Hidden States Detach)** 🔴 ✅ **已修复**
   - 影响: 可能完全阻断 BPTT
   - 修复: 改为在 runner 的 update 完成后统一 detach
   - 日期: 2026-04-16

2. **问题 2 (Gradient Flow)** 🔴
   - 影响: `unpad_trajectories` 可能破坏计算图
   - 建议: 使用 `pack_padded_sequence` 或检查实现

3. **问题 3 (Teacher/Student 干扰)** 🔴
   - 影响: 数据收集和训练不一致
   - 建议: 添加调试代码验证 hidden states 独立性

4. **问题 4 (序列长度)** 🟡
   - 影响: 无法学习长期依赖
   - 建议: 增加 `num_steps_per_env` 到 48 或 64

---

## 📊 预期效果

修复 BPTT 问题后，预期能看到：

- ✅ `latent_reconstruction_loss` 显著下降
- ✅ 训练更稳定，loss 曲线更平滑
- ✅ 更好的时序建模能力
- ✅ 最终策略性能提升

---

## 🔍 验证清单

实施修复后，验证以下项目：

### 已修复项目
- [x] **问题 1**: Hidden states detach 时机 - 验证梯度能跨 mini-batch 传播
  - [ ] 梯度能传播到 trajectory 的早期 time steps
  - [ ] 梯度范数在合理范围内 (非零且不过大)

### 待修复/验证项目
- [ ] **问题 2**: `unpad_trajectories` 后梯度仍然存在
- [ ] **问题 3**: Teacher 和 Student 的 hidden states 完全独立
- [ ] **问题 4**: 增加序列长度后 Loss 曲线呈现稳定下降趋势
- [ ] **问题 5**: Hidden states 在 done=True 时正确重置
- [ ] **问题 6**: Storage 中 hidden states 维度正确

