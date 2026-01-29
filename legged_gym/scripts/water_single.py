import numpy as np
import torch
import genesis as gs
from itertools import combinations
import faulthandler
faulthandler.enable()


substeps = 10
particle_size = 0.011
liquid_mass = 1000
liquid_mu = 0.005
liquid_gamma = 0.010
go1_torso_height = 0.114

# # Cube Parameters
# outer_x = 2.0;  # X dimension
# outer_y = 2.0;  # Y dimension
# outer_z = 2.0;  # total outer height
# wall_thickness = .20
# bottom_thickness = 0.21
# stl_scale = 0.1

# Cube Parameters
outer_x = 0.20;  # X dimension
outer_y = 0.15;  # Y dimension
outer_z = 0.10;  # total outer height
wall_thickness = .015
bottom_thickness = 0.015
stl_scale = 2.0


liquid_init_buffer = 0.035 # (needs to be slightly bigger than particle size I suspect)
bucket_offset = (go1_torso_height/2.0)
#  ^^^ 
# (go1_torso_height/2.0)  - half-"thickness" of torso 
# (0.5*stl_scale*outer_z) - box is scaled from all sides, so half of the scaled height

lid_offset = (go1_torso_height/2.0) + (stl_scale*outer_z)

def random_yaw_quaternion(batch_size=1, yaw_range=torch.tensor([-3.14159, 3.14159]),
                          device="cpu", dtype=torch.float32):
    # Sample random yaw angles
    yaw = torch.rand(batch_size, device=device, dtype=dtype)
    yaw = yaw_range[0] + (yaw_range[1] - yaw_range[0]) * yaw

    # Compute quaternion for yaw rotation (roll=0, pitch=0)
    half_yaw = 0.5 * yaw
    cy = torch.cos(half_yaw)
    sy = torch.sin(half_yaw)

    # Quaternion in (w, x, y, z), yaw rotates around Z-axis
    quat = torch.zeros((batch_size, 4), device=device, dtype=dtype)
    quat[:, 0] = cy       # w
    quat[:, 3] = sy       # z

    return quat

def gs_inv_quat(quat):
    qw, qx, qy, qz = quat.unbind(-1)
    inv_quat = torch.stack([1.0 * qw, -qx, -qy, -qz], dim=-1)
    return inv_quat

def gs_transform_by_quat(pos, quat):
    qw, qx, qy, qz = quat.unbind(-1)
    rot_matrix = torch.stack(
        [
            1.0 - 2 * qy**2 - 2 * qz**2,
            2 * qx * qy - 2 * qz * qw,
            2 * qx * qz + 2 * qy * qw,
            2 * qx * qy + 2 * qz * qw,
            1 - 2 * qx**2 - 2 * qz**2,
            2 * qy * qz - 2 * qx * qw,
            2 * qx * qz - 2 * qy * qw,
            2 * qy * qz + 2 * qx * qw,
            1 - 2 * qx**2 - 2 * qy**2,
        ],
        dim=-1,
    ).reshape(*quat.shape[:-1], 3, 3)
    if pos.dim() == 3:
      rotated_pos = torch.matmul(rot_matrix[:, None, :], pos.unsqueeze(-1)).squeeze(-1)
    else:
      rotated_pos = torch.matmul(rot_matrix, pos.unsqueeze(-1)).squeeze(-1)

    return rotated_pos

# Borrowed these functions from gs_utils.py
def gs_quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([w, x, y, z], dim=-1).view(shape)

    return quat

go1_default = {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        }

go1_list = [
            'FR_hip_joint',
            'FR_thigh_joint',
            'FR_calf_joint',
            'FL_hip_joint',
            'FL_thigh_joint',
            'FL_calf_joint',
            'RR_hip_joint',
            'RR_thigh_joint',
            'RR_calf_joint',
            'RL_hip_joint',
            'RL_thigh_joint',
            'RL_calf_joint',
          ]

go1_links = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']

class LiquidOpts():

  """
  rho (float, optional) – The density (kg/m^3) the material tends to maintain in equilibrium (i.e., the “rest” or undeformed state). Default is 1000.
  stiffness (float, optional) – State stiffness (N/m^2). A material constant controlling how pressure increases with compression. Default is 50000.0.
  exponent (float, optional) – State exponent. Controls how nonlinearly pressure scales with density. Larger values mean stiffer response to compression. Default is 7.0.
  mu (float, optional) – The vscosity of the liquid. A measure of the internal friction of the fluid or material. Default is 0.005
  gamma (float, optional) – The surface tension of the liquid. Controls how strongly the material “clumps” together at boundaries. Default is 0.01
  sampler (str, optional) – Particle sampler (‘pbs’, ‘regular’, ‘random’). Default is ‘pbs’
  """
  def __init__(self, **kwargs):
    self.rho = 1000
    self.stiffness = 50000.0
    self.exponent = 7.0
    self.mu = 0.005
    self.gamma = 0.01
    self.sampler = 'pbs'

class Creator():
  def __init__(self):
    if not torch.cuda.is_available():
      self.device = torch.device('cpu')
    else:
      assert "cuda" in ["cpu", "cuda"]
      self.device = torch.device("cuda")
    gs.init(backend=gs.gpu)
    self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=0.002,
                substeps=substeps
            ),
            # viewer_options=gs.options.ViewerOptions(
            #     max_FPS=int(1 / 0.002 * 4),
            #     #camera_pos=np.array(self.cfg.viewer.pos),
            #     #camera_lookat=np.array(self.cfg.viewer.lookat),
            #     camera_fov=40,
            # ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            rigid_options=gs.options.RigidOptions(
                dt=0.002,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                enable_self_collision=True,
                batch_dofs_info=True,   # batch dof info for all envs
                batch_joints_info=True,
                batch_links_info=True,
                use_gjk_collision=True
            ),
            sph_options=gs.options.SPHOptions(
                #  lower_bound = (-1,-1,-1),
                #  upper_bound = (1,1,1),
                particle_size = particle_size,
            ),
            show_viewer=True,
        )

    self.cam = self.scene.add_camera(
       res    = (1280, 960),
       pos    = (1.0*stl_scale, 1.0*stl_scale , 0.5*stl_scale ),
       lookat = (0, 0, 0.1),
       fov    = 40,
       GUI    = True
   )

    self.plane = self.scene.add_entity(
                gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    self.rob_pos = (0, 0, 0.32)
    self.franka = None
    self.liquid = None
  
  def add_robot(self):
    self.franka = self.scene.add_entity(
        gs.morphs.URDF(
            pos = self.rob_pos,
            file='../../resources/robots/go1/urdf/go1.urdf',
            merge_fixed_links= True,
            links_to_keep=go1_links,
            ),
    )
  

  # For multiple robots, might need to be smarter and initialize the liquids and payloads at their "starting" positions
  #     on top of the robots...
  def add_box(self, n_envs):
    # Add the liquid container
    self.bucket = self.scene.add_entity(
      material=gs.materials.Rigid(
        gravity_compensation=1.0,
        ),
      morph=gs.morphs.Mesh(
          file="water_tank_proper_units_simple.stl",
          scale=(stl_scale, stl_scale, stl_scale),    # adjust scale if needed
          pos= (0, 0, bucket_offset),      # position
          quat=(1.0, 0.0, 0.0, 0.0), # no rotation; uses w, x, y, z quaternion
          decimate=False,
          convexify=False
      ),
      surface=gs.surfaces.Glass(opacity=0.6)
    )

    # Add a lid to the liquid container
    if self.franka:
        _lid_off =  (lid_offset - (go1_torso_height/2.0))* stl_scale  +  (go1_torso_height/2.0)
    else:
        _lid_off = lid_offset * stl_scale  - (go1_torso_height/2.0)
    self.lid = self.scene.add_entity(
      material=gs.materials.Rigid(
        gravity_compensation=1.0,
        ),
      morph=gs.morphs.Mesh(
          file="water_tank_lid.stl",
          scale=(stl_scale, stl_scale, 1),    # adjust scale if needed
          pos= (0, 0, _lid_off),      # position
          quat=(1.0, 0.0, 0.0, 0.0), # no rotation; uses w, x, y, z quaternion
          decimate=False,
          convexify=False
      ),
      
      surface=gs.surfaces.Glass(opacity=0.6)
    )

    # print(self.lid)

    # Calculate the scaled internal dimensions of the container
    #     ultimately we will want to randomly scale each axis within some pre-defined bounds (0.1 +/- 0.05?])
    scaled_width = outer_x * stl_scale - 2.0 * (stl_scale*wall_thickness)
    scaled_depth = outer_y * stl_scale - 2.0 * (stl_scale*wall_thickness)
    scaled_height = outer_z * stl_scale - stl_scale*bottom_thickness

    varied_buffers = np.random.uniform(0.005, 0.075, size=(n_envs,))

    # self.liquids = [
    #   self.scene.add_entity(
    #     material=gs.materials.SPH.Liquid(rho=liquid_mass, mu=liquid_mu, gamma=liquid_gamma),
    #     morph=gs.morphs.Box(pos=(self.rob_pos[0], self.rob_pos[1], self.rob_pos[2] + bucket_offset + 0.5*scaled_height), 
    #                       size=(scaled_width-varied_buffers[i],scaled_depth-varied_buffers[i],scaled_height-varied_buffers[i])),
    #     surface=gs.surfaces.Water(),
    # ) for i in range(n_envs)
    # ]
    if not self.franka:
        _bucket_offset = -0.2 
    else:
        _bucket_offset = bucket_offset +  0.5*scaled_height
    self.liquid = self.scene.add_entity(
        material=gs.materials.SPH.Liquid(rho=liquid_mass, mu=liquid_mu, gamma=liquid_gamma),
        morph=gs.morphs.Box(pos=(self.rob_pos[0], self.rob_pos[1], self.rob_pos[2] + _bucket_offset), 
                          size=(scaled_width-varied_buffers[0],scaled_depth-varied_buffers[0],scaled_height-varied_buffers[0])),
        surface=gs.surfaces.Water(),
    )

    #aprox 1 particle per 0.0001 m^3
    #0.000783458709716797/749
    
  
  def build(self, **kwargs):
    # Build the scene
    if kwargs.get("n_envs") is None:
      kwargs["n_envs"] = 1
    self.num_envs = kwargs["n_envs"]
    
    if self.bucket and self.franka:
      self.bucket.attach(self.franka, "base")
    
    if self.lid and self.franka:
      self.lid.attach(self.franka, "base")
    elif self.bucket and self.lid:
      print(self.bucket.links)
      self.lid.attach(self.bucket, "water_tank_proper_units_simple_stl_baselink")

    
    self.scene.build(**kwargs)
    
    # If the liquid and robot are added, 
    # then set their initial pose and cache some values for reset
    if self.liquid:
      self.liquid_init_pose = self.liquid.get_particles_pos()
    if self.franka:
        self.motors = [self.franka.get_joint(x).dof_start for x in go1_list]
        self.franka.set_dofs_position([[go1_default[i] for i in go1_list]], self.motors)
        self.franka_init_pos = self.franka.get_pos().detach().clone()
        self.franka_init_quat = self.franka.get_quat().detach().clone()
        self.franka_init_dof_pos = self.franka.get_dofs_position().detach().clone()

  def reset(self):

    # Random x/y offsets
    rand_pos_offset = 1.0*torch.rand_like(self.franka_init_pos)
    # Zeroout the random height offset
    rand_pos_offset[:, 2] = 0.0
    # New robot pose
    new_robot_pos = self.franka_init_pos + rand_pos_offset
    new_particle_pos_offset    = new_robot_pos.detach().clone()
    new_particle_pos_offset[:, 2] = 0.0 # no need to modify the height

    # Random_yaw offsets
    rand_yaw_offset = random_yaw_quaternion(
      batch_size=self.franka_init_quat.shape[0], 
      device=self.franka_init_pos.device
    ).squeeze()
    new_robot_quat = gs_quat_mul(rand_yaw_offset, self.franka_init_quat.squeeze())
    
    self.franka.set_dofs_position(self.franka_init_dof_pos)
    self.franka.set_quat(new_robot_quat)
    self.franka.set_pos(new_robot_pos)
    #self.bucket.set_pos(new_robot_pos + gs.tensor([0, 0, bucket_offset]))
    #self.lid.set_pos(new_robot_pos + gs.tensor([0, 0, lid_offset]))

    #rigid.delete_weld_constraint(link_lid, link_franka)
    #rigid.delete_weld_constraint(link_cube, link_franka)
    #
    #rigid.add_weld_constraint(link_cube, link_franka)
    #rigid.add_weld_constraint(link_lid, link_franka)
    
    self.franka.zero_all_dofs_velocity()
    
    # apply the yaw change to particle init positions THEN apply offset
    # active = torch.randint(0, len(self.liquids), (self.num_envs,1))
    self.liquid.set_particles_vel(0)
    new_particle_posistions = gs_transform_by_quat(self.liquid_init_pose, rand_yaw_offset)
    new_particle_posistions += new_particle_pos_offset[:, None, :]
    self.liquid.set_particles_pos(new_particle_posistions)


num_envs = 1  

with torch.no_grad():
  cass = Creator()
  cass.add_robot()
  cass.add_box(n_envs=num_envs)
  cass.build(n_envs=num_envs, env_spacing=(1.0, 1.0))


  cass.cam.start_recording()

  import time
  for _ in range(500):
    cass.scene.step()
    if cass.franka:
        cass.franka.set_dofs_position(cass.franka_init_dof_pos)

    cass.cam.render()
    
    
  #

  #variables can be changed at run time

  # for i in range(500)_create_envs
  #   cass.scene.step()
  #   cass.cam.render()

  from datetime import datetime

  cass.cam.stop_recording(save_to_filename=f'video_go2_large_box.mp4', fps=60)

  #https://github.com/Genesis-Embodied-AI/Genesis/blob/main/genesis/engine/entities/sph_entity.py
