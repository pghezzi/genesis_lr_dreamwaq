import numpy as np
import torch
import genesis as gs
from itertools import combinations

liquid_mass = 1000

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
    gs.init()
    self.particle_size = 0.02
    self.scene = gs.Scene(
      #rigid_options=gs.options.RigidOptions(
      #          #dt=0.005,
      #          constraint_solver=gs.constraint_solver.Newton,
      #          enable_collision=True,
      #          enable_joint_limit=True,
      #          enable_self_collision=True,
      #          batch_dofs_info=True,   # batch dof info for all envs
      #          batch_joints_info=True,
      #          batch_links_info=True,
      #      ),
      sim_options=gs.options.SimOptions(dt=4e-3, substeps=10),
      #sph_options=gs.options.SPHOptions(
      #  lower_bound = (-1,-1,-1),
      #  upper_bound = (1,1,1),
      #  particle_size = self.particle_size,
      #),
      vis_options = gs.options.VisOptions(
        visualize_sph_boundary = True,
      ),
      show_viewer = True,
    )

    self.cam = self.scene.add_camera(
      res    = (1280, 960),
      pos    = (3.5, 0.0, 2.5),
      lookat = (0, 0, 0.5),
      fov    = 30,
      GUI    = False
  )

    self.plane = self.scene.add_entity(morph=gs.morphs.Plane())
    self.panels = None
    self.franka = None
    self.liquid = None
  
  def add_robot(self):
    self.rob_pos = (0, 0, 0.5)
    self.franka = self.scene.add_entity(
        gs.morphs.URDF(
            pos = self.rob_pos,
            file='/home/pablo/Documents/work/w/genesis_lr_dreamwaq/resources/robots/go1/urdf/go1.urdf'),
    )
  

  def add_box(self, dim = (0.2, 0.2, 0.2), thickness = 1/32, number_of_particles = 100,):
    number_of_particles = max(1, number_of_particles)
    H, W, D = dim
    self.H = H
    panels = [
        gs.morphs.Box(pos=(0, -H/2 + thickness/2, 0),     size=(W, thickness, D)),
        gs.morphs.Box(pos=(0, +H/2 - thickness/2, 0),     size=(W, thickness, D)), 
        gs.morphs.Box(pos=(-W/2 + thickness/2, 0, 0),     size=(thickness, H - 2*thickness - 0.01, D- 2*thickness)),
        gs.morphs.Box(pos=(+W/2 - thickness/2, 0, 0),     size=(thickness, H - 2*thickness - 0.01, D- 2*thickness - 0.01)),
        gs.morphs.Box(pos=(0, 0, -D/2 + thickness/2),     size=(W - 2*thickness - 0.01, H - 2*thickness - 0.01, thickness)),
        gs.morphs.Box(pos=(0, 0, +D/2 - thickness/2),     size=(W - 2*thickness - 0.01, H - 2*thickness - 0.01, thickness)),
    ]

    #self.panels = [
    #    self.scene.add_entity(x,
    #      surface=gs.surfaces.Glass(opacity=0.2),
    #    ) for x in panels
    #]
    self.panels = ["m"]
    self.bucket = self.scene.add_entity(
      gs.morphs.Mesh(
          file="cube_2.stl",
          scale=(0.1, 0.1, 0.1),    # adjust scale if needed
          pos= [0, 0, self.H - 0.2],      # position
          quat=(1.0, 0.0, 0.0, 0.0) # no rotation; uses w, x, y, z quaternion
      ),
      surface=gs.surfaces.Glass(opacity=0.2)
    )
    water_dims  = (H - thickness,W - thickness, D - thickness)
    wd = np.cbrt(int(number_of_particles)) * self.particle_size
    wd = water_dims[0]
    #
    self.liquid = self.scene.add_entity(
      material=gs.materials.SPH.Liquid(rho=liquid_mass),
      morph=gs.morphs.Box(pos=(0, 0, self.rob_pos[2] + H - thickness + 0.02), size=(wd-0.05,wd-0.05,wd-0.05)),
      surface=gs.surfaces.Water( 
      ),
    )
    #aprox 1 particle per 0.0001 m^3
    #0.000783458709716797/749
    
  
  def build(self, **kwargs):
    self.scene.build(**kwargs)
    if self.panels and self.franka:
      self.panels = []
      rigid = self.scene.sim.rigid_solver
      base = self.franka.get_link("base")
      cube_link = []
      link_franka = np.array([base.idx], dtype=gs.np_int)
      pos = base.get_pos()
      if len(pos.shape) == 1:
        z_pos = pos[2]
      else:
        z_pos = pos[0, 2]
      for cube in self.panels:
          cube.set_pos(cube.get_pos() +  gs.tensor([0, 0, z_pos + self.H]))
          cube_link = cube.get_link("box_baselink")
          link_cube   = np.array([cube_link.idx],   dtype=gs.np_int)
          rigid.add_weld_constraint(link_cube, link_franka)
      
      self.bucket.set_pos(self.bucket.get_pos() +  gs.tensor([0, 0, z_pos + self.H + 0.05]))
      cube_link = self.bucket.get_link("cube_2_stl_baselink")
      link_cube   = np.array([cube_link.idx],   dtype=gs.np_int)
      rigid.add_weld_constraint(link_cube, link_franka)
    self.franka_init_pos = torch.zeros_like(
      self.franka.get_pos()
    )
    self.franka_init_quat = torch.zeros_like(
      self.franka.get_quat()
    )
    self.franka_init_vel = torch.zeros_like(
      self.franka.get_vel()
    )
    if self.liquid is not None:
      self.liquid_init_pos = torch.zeros_like(
        self.liquid.get_particles_pos()
      )
      self.liquid_init_pos[:] = self.liquid.get_particles_pos()

    self.franka_init_dof_pos =  torch.zeros_like(
      self.franka.get_dofs_position()
    )
    self.franka_init_dof_pos[:] = self.franka.get_dofs_position()
    self.franka_init_pos[:] = self.franka.get_pos()
    self.franka_init_quat[:] = self.franka.get_quat()

  def reset(self):
    print(f"reset to {self.franka_init_pos}")
    cass.scene.reset()
    self.franka.set_pos(self.franka.get_pos() + gs.tensor([5, 0, 1]))
    self.franka.zero_all_dofs_velocity()
    #self.liquid.set_particles_pos(self.liquid.get_particles_pos() + gs.tensor([1, 0, 0.5]))
    


cass = Creator()
cass.add_robot()
cass.add_box(dim=(0.2, 0.2, 0.2))



cass.build()


cass.cam.start_recording()

print(cass.liquid)

import time

for i in range(20):
  for _ in range(30):
    cass.scene.step()
    cass.cam.render()
    print(cass.bucket.get_pos())
    input()
    #input()
  
  #print("reset")
  #time.sleep(5)
  cass.reset()
  print(cass.liquid.get_particles_vel())
  
#cass.liquid.rho = 10_000

#variables can be changed at run time

for i in range(500):
  cass.scene.step()
  cass.cam.render()

from datetime import datetime

cass.cam.stop_recording(save_to_filename=f'video_liquid_{liquid_mass}_{datetime.now().timestamp()}.mp4', fps=60)

#https://github.com/Genesis-Embodied-AI/Genesis/blob/main/genesis/engine/entities/sph_entity.py
