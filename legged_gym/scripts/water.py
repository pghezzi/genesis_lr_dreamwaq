import numpy as np
import torch
import genesis as gs
from itertools import combinations

liquid_mass = 15_000

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
    self.particle_size = 0.05
    self.scene = gs.Scene(
      sim_options=gs.options.SimOptions(dt=4e-3, substeps=10),
      sph_options=gs.options.SPHOptions(
        lower_bound = (-1,-1,-1),
        upper_bound = (1,1,1),
        particle_size = self.particle_size,
      ),
      vis_options = gs.options.VisOptions(
        visualize_sph_boundary = True,
      ),
      show_viewer = False,
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
  
  def add_robot(self):
    self.rob_pos = (0, 0, 0.5)
    self.franka = self.scene.add_entity(
        gs.morphs.URDF(
            pos = self.rob_pos,
            file='/home/pablo/Documents/work/w/genesis_lr_dreamwaq/resources/robots/go1/urdf/go1.urdf'),
    )
  

  def add_box(self, dim = (0.2, 0.2, 0.2), thickness = 1/128, number_of_particles = 100,):
    number_of_particles = max(1, number_of_particles)
    H, W, D = dim
    self.H = H
    panels = [
        gs.morphs.Box(pos=(0, -H/2 + thickness/2, 0),     size=(W, thickness, D)),
        gs.morphs.Box(pos=(0, +H/2 - thickness/2, 0),     size=(W, thickness, D)), 
        gs.morphs.Box(pos=(-W/2 + thickness/2, 0, 0),     size=(thickness, H, D)),
        gs.morphs.Box(pos=(+W/2 - thickness/2, 0, 0),     size=(thickness, H, D)),
        gs.morphs.Box(pos=(0, 0, -D/2 + thickness/2),     size=(W, H, thickness)),
        gs.morphs.Box(pos=(0, 0, +D/2 - thickness/2),     size=(W, H, thickness)),
    ]

    self.panels = [
        self.scene.add_entity(x,
          surface=gs.surfaces.Glass(opacity=0.2),
        ) for x in panels
    ]

    water_dims  = (H - thickness,W - thickness, D - thickness)
    wd = np.cbrt(int(number_of_particles)) * self.particle_size
    wd = water_dims[0]
    self.liquid = self.scene.add_entity(
      material=gs.materials.SPH.Liquid(rho=liquid_mass),
      morph=gs.morphs.Box(pos=(0, 0, self.rob_pos[2] + H), size=(wd,wd,wd)),
      surface=gs.surfaces.Water( 
        vis_mode='recon'
      ),
    )
    #aprox 1 particle per 0.0001 m^3
    #0.000783458709716797/749
    
  
  def build(self, **kwargs):
    self.scene.build(**kwargs)
    if self.panels and self.franka:
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


cass = Creator()
cass.add_robot()
cass.add_box(dim=(0.1, 0.1, 0.1))
cass.build()

cass.cam.start_recording()
for i in range(500):
  cass.scene.step()
  cass.cam.render()

cass.cam.stop_recording(save_to_filename=f'video_liquid_{liquid_mass}.mp4', fps=60)
