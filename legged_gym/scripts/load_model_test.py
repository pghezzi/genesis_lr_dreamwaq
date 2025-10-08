import torch
import numpy as np
import genesis as gs

gs.init(backend=gs.gpu)

scene = gs.Scene(
    sim_options = gs.options.SimOptions(
        dt = 0.01,
    ),
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (0, -3.5, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 30,
        max_FPS       = 60,
    ),
    show_viewer = True,
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)

go1 = = scene.add_entity(
    gs.morphs.URDF(
        pos = (6, 6, 1),
        file='/home/pablo/Documents/work/w/genesis_lr_dreamwaq/resources/robots/go1/urdf/go1.urdf'),
)


scene.build()




#self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
#                                    self.projected_gravity,
#                                    self.commands[:, :3] * self.commands_scale,
#                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
#                                    self.dof_vel * self.obs_scales.dof_vel,
#                                    self.actions
#                                    ),dim=-1)
# Load the exported model
model = torch.jit.load("estimator_1.pt")
model.eval()

# Simulated robot state (replace with real sensor input)
obs = np.array([0.0, 1.0, -0.5, ...], dtype=np.float32)
obs_tensor = torch.tensor(obs).unsqueeze(0)  # Add batch dim

# Run inference
with torch.no_grad():
    action = model(obs_tensor)

# Convert action to control signal for motors
action = action.squeeze(0).numpy()
# Send to robot actuator controller


for _ in range(self.cfg.control.decimation):  # use self-implemented pd controller
    self.torques = self._compute_torques(self.actions)
    torques = self.torques.squeeze()
    self.robot.control_dofs_force(torques, self.motors_dof_idx)

