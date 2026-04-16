# go2_ts_depth Training Issues Report

**Date**: 2026-04-15  
**Task**: go2_ts_depth (Teacher-Student with Depth Image)  
**Issue**: latent_reconstruction_loss remains high (expected ~0.01, actual higher)  

---

## Executive Summary

After deep analysis of the training pipeline, we identified **4 critical issues** causing the latent reconstruction loss to not converge properly.

**Most Critical Issue**: Teacher network and student encoder are trained on **different data batches**, causing inconsistent learning objectives.

---

## Issues Found

### Issue 1: Dual Generator Calls Cause Data Mismatch

**Status**: Not Fixed | **Priority**: CRITICAL

**Location**: rsl_rl/algorithms/ppo_ts_depth.py, lines 128-135 and 196-224

**Problem**: The update() method calls teacher_mini_batch_generator() twice:
1. First call for teacher update
2. Second call for student encoder training

Each call creates a new random permutation (indices), so teacher and student see completely different data batches.

**Impact**:
- Teacher and student train on mismatched data
- Latent reconstruction loss cannot converge
- Student encoder learns wrong targets

**Solution**: Use a single generator loop for both teacher and student updates.

---

### Issue 2: Hidden States Dimension Mismatch

**Status**: Not Fixed | **Priority**: HIGH

**Location**: rsl_rl/storage/rollout_storage_ts_depth.py, lines 30, 57-61, 126-130

**Problem**: 
- depth_image_features shape: [T, num_student, ...]
- saved_hidden_states shape: [T, num_layers, num_envs, hidden_dim]

Hidden states are saved for ALL environments but depth features only for num_student environments.

**Impact**:
- Hidden states may correspond to wrong environments
- RNN state confusion affects temporal modeling

**Solution**: Only save hidden states for num_student environments.

---

### Issue 3: Data Alignment Issue

**Status**: To Verify | **Priority**: MEDIUM

**Location**: rsl_rl/storage/rollout_storage_ts_depth.py, lines 70-88

**Problem**: Teacher uses randomly sampled data from ALL environments, while student uses only first num_student environments.

**Impact**:
- Teacher may not see student environments during training
- Inconsistent data distribution

**Solution**: Teacher should only use non-student environments (num_student:num_envs).

---

### Issue 4: Masks Computation Issue

**Status**: Low Priority | **Priority**: LOW

**Location**: rsl_rl/storage/rollout_storage_ts_depth.py, lines 112-115

**Problem**: Code forces last_was_done[0] = True, which may incorrectly split trajectories.

**Solution**: Remove the forced setting.

---

## TODO List

### High Priority (Must Fix)

- [x] **TODO-1** (COMPLETED): Fix hidden states dimension mismatch
  - File: rsl_rl/storage/rollout_storage_ts_depth.py
  - Action: Save hidden states only for num_student environments
  - Expected: RNN states properly aligned

### Medium Priority (Recommended)

- [x] **TODO-2** (COMPLETED): Replace MSE with L2 norm loss
  - File: rsl_rl/algorithms/ppo_ts_depth.py
  - Action: Change `nn.functional.mse_loss()` to `.norm(p=2, dim=1).mean()`
  - Reference: extreme-parkour uses L2 norm for better training stability
  - Code change:
    ```python
    # Before
    latent_reconstruction_loss = nn.functional.mse_loss(latent, latent_targets)
    
    # After
    latent_reconstruction_loss = (latent_targets.detach() - latent).norm(p=2, dim=1).mean()
    ```
  - Expected: More robust training, less sensitive to outliers

### Low Priority (Optional)

- [ ] **TODO-3**: Fix masks computation
  - File: rsl_rl/storage/rollout_storage_ts_depth.py
  - Action: Remove last_was_done[0] = True
  - Expected: More accurate trajectory splitting

### Validation Tasks

- [ ] **TODO-4**: Add debug output to verify data alignment
- [ ] **TODO-5**: Run small-scale overfitting test (1-4 envs)
- [ ] **TODO-6**: Compare training curves before/after fixes

---

## Fix Code Reference

### Fix 1: Single Generator Loop (ppo_ts_depth.py)

```python
def update(self):
    generator = self.storage.teacher_mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs)
    
    for (obs_batch, privileged_obs_batch, critic_obs_batch, actions_batch,
         target_values_batch, returns_batch, old_actions_log_prob_batch,
         advantages_batch, old_mu_batch, old_sigma_batch,
         student_obs_batch, student_privileged_obs_batch, depth_features_batch,
         hid_states_batch, masks_batch) in generator:
        
        # 1. Teacher update
        self.actor_critic.act(obs_batch, None, privileged_obs_batch, "teacher", None, None)
        # ... teacher update code ...
        
        # 2. Student encoder update (same batch!)
        latent = self.actor_critic.depth_history_encoder(
            student_obs_batch, depth_features_batch,
            hidden_states=hid_states_batch, masks=masks_batch)
        
        with torch.no_grad():
            unpadded = unpad_trajectories(student_privileged_obs_batch, masks_batch)
            latent_targets = self.actor_critic.privilege_encoder(unpadded)
        
        loss = nn.functional.mse_loss(latent, latent_targets)
        # ... student update code ...
```

### Fix 2: Save Only Student Hidden States (rollout_storage_ts_depth.py)

```python
def _save_hidden_states(self, hidden_states):
    if hidden_states is None or hidden_states == (None, None):
        return
    hid = hidden_states if isinstance(hidden_states, tuple) else (hidden_states,)
    
    # Only take first num_student environments
    hid_student = [h[:, :self.num_student, :] for h in hid]
    
    if self.saved_hidden_states is None:
        self.saved_hidden_states = [
            torch.zeros(self.observations.shape[0], *h.shape, device=self.device)
            for h in hid_student
        ]
    
    for i in range(len(hid_student)):
        self.saved_hidden_states[i][self.step].copy_(hid_student[i])
```

---

## Summary

The root cause of high latent reconstruction loss is that teacher and student are trained on different data batches (Issue 1). Fixing this should significantly improve convergence. Issue 2 (hidden states mismatch) may also contribute to training instability.

