import torch


class RL2ACAdaptiveCtrl:
    def __init__(self, num_envs, device="cuda", dtype=torch.float32):
        self.device = device
        self.dtype = dtype

        self.B = num_envs
        self.J = 12

        # Scalars (broadcasted)
        self.alpha = 20.0
        self.kappa = 1.2
        self.eta = 0.0001
        self.lambda_0 = 0.0
        self.k_0 = 3.0

        # State flags
        self.use_proactive_ctrl = False

        # Joint-space vectors: [B, J]
        self.phi = torch.zeros(self.B, self.J, device=device, dtype=dtype)
        self.s = torch.zeros_like(self.phi)
        self.tau = torch.zeros_like(self.phi)
        self.epsilon = torch.zeros_like(self.phi)

        self.q = torch.zeros_like(self.phi)
        self.qdot = torch.zeros_like(self.phi)

        self.q_ref = torch.zeros_like(self.phi)
        self.q_des = torch.zeros_like(self.phi)

        self.tau_des = torch.zeros_like(self.phi)
        self.tauDes_old = torch.zeros_like(self.phi)

        self.comp_old = torch.zeros_like(self.phi)
        self.comp = torch.zeros_like(self.phi)

        # Adaptive matrices: [B, J, J]
        self.Gamma = torch.eye(self.J, device=device, dtype=dtype).repeat(self.B, 1, 1)
        self.K = torch.zeros(self.B, self.J, self.J, device=device, dtype=dtype)

        # Numerical stability constants
        self.gamma_max_norm = 1e3
        self.phi_norm_max = 2.0
        self.min_lambda = 0.0
        self.dt_min = 1e-5

    # ------------------------------------------------------------------
    # State update (called every sim step)
    # ------------------------------------------------------------------

    def update_state(
        self,
        qpos,           # [B, nq]
        qvel,           # [B, nv]
        qfrc_actuator,  # [B, nv]
    ):
        qj = qpos
        qdj = qvel

        if self.use_proactive_ctrl:
            self.phi = self.q_des - self.q_ref
        else:
            self.phi = self.q_des - qj

        # Sliding variable
        self.s = qdj - self.alpha * (self.q_des - qj)

        # Torque tracking error
        self.tau = qfrc_actuator
        self.epsilon = (self.tauDes_old + self.comp) - self.tau

        # Log states
        self.q.copy_(qj)
        self.qdot.copy_(qdj)

        # ---- Stability: clamp ||phi||
        phi_norm = torch.norm(self.phi, dim=1, keepdim=True).clamp(min=1e-6)
        scale = torch.clamp(self.phi_norm_max / phi_norm, max=1.0)
        self.phi.mul_(scale)

    # ------------------------------------------------------------------
    # Command update
    # ------------------------------------------------------------------

    def update_cmd(self, q_ref, q_des, tau_cmd):
        self.q_ref.copy_(q_ref)
        self.q_des.copy_(q_des)

        self.tauDes_old.copy_(self.tau_des)
        self.tau_des.copy_(tau_cmd)

    # ------------------------------------------------------------------
    # Adaptive compensation update
    # ------------------------------------------------------------------

    def update_compensation(self, dt):
        self._update_forgetting_factor()
        self._update_gamma(dt)
        self._update_K(dt)

        self.comp_old.copy_(self.comp)
        self.comp = torch.einsum("bij,bj->bi", self.K, self.phi)

        return self.comp

    # ------------------------------------------------------------------
    # Internal updates
    # ------------------------------------------------------------------

    def _update_forgetting_factor(self):
        gamma_norm = torch.norm(self.Gamma, dim=(1, 2))
        lambda_val = self.lambda_0 * (1.0 - gamma_norm / self.k_0)
        self.lambda_val = torch.clamp(lambda_val, min=self.min_lambda)

    def _update_gamma(self, dt):
        # Elementwise equivalent of: Γ φ φᵀ Γ
        phi_outer = self.phi.unsqueeze(2) * self.phi.unsqueeze(1)  # [B,J,J]

        dGamma = (
            self.lambda_val[:, None, None] * self.Gamma
            - torch.bmm(self.Gamma, torch.bmm(phi_outer, self.Gamma))
        )

        self.Gamma += dt * dGamma

        # ---- Stability: norm clamp + symmetrize
        gamma_norm = torch.norm(self.Gamma, dim=(1, 2), keepdim=True).clamp(min=1e-6)
        scale = torch.clamp(self.gamma_max_norm / gamma_norm, max=1.0)
        self.Gamma.mul_(scale)

        self.Gamma = 0.5 * (self.Gamma + self.Gamma.transpose(1, 2))

    def _update_K(self, dt):
        # Equivalent to: -Γ φ (s + κ ε)ᵀ
        rhs = self.s + self.kappa * self.epsilon
        dK = -torch.einsum("bij,bj,bk->bik", self.Gamma, self.phi, rhs)
        dK -= self.eta * self.K

        self.K += dt * dK
