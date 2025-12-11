import genesis as gs
from itertools import combinations
import numpy as np

gs.init()

scene = gs.Scene(
  sim_options=gs.options.SimOptions(dt=4e-3, substeps=10),
  sph_options=gs.options.SPHOptions(
     lower_bound = (-1,-1,-1),
     upper_bound = (1,1,3),
     particle_size = 0.01,
  ),
  vis_options = gs.options.VisOptions(
    visualize_sph_boundary = True,
  ),
  show_viewer = True,
)



plane = scene.add_entity(morph=gs.morphs.Plane())


franka = scene.add_entity(
    gs.morphs.URDF(
        pos = (0, 0, 1),
        file='/home/pablo/Documents/work/w/genesis_lr_dreamwaq/resources/robots/go1/urdf/go1.urdf'),
)


thickness = 0.05
W, H, D = 0.2, 0.2, 0.2

panels = [
    gs.morphs.Box(pos=(0, -H/2 + thickness/2, 0),     size=(W, thickness, D)),
    gs.morphs.Box(pos=(0, +H/2 - thickness/2, 0),     size=(W, thickness, D)), 
    gs.morphs.Box(pos=(-W/2 + thickness/2, 0, 0),     size=(thickness, H, D)),
    gs.morphs.Box(pos=(+W/2 - thickness/2, 0, 0),     size=(thickness, H, D)),
    gs.morphs.Box(pos=(0, 0, -D/2 + thickness/2),     size=(W, H, thickness)),
    #gs.morphs.Box(pos=(0, 0, +D/2 - thickness/2),     size=(W, H, thickness)),
]



panels = [
    scene.add_entity(x,
    surface=gs.surfaces.Glass(),
    ) for x in panels
]


liquid = scene.add_entity(
  material=gs.materials.SPH.Liquid(),
  morph=gs.morphs.Box(pos=(0, 0, 1.2), size=(0.1,0.1,0.1)),
  surface=gs.surfaces.Default(color=(0.4,0.8,1.0), vis_mode='particle'),
)


#cube = scene.add_entity(
#    gs.morphs.Box(pos=(0.0,0.0,0.65), size=(0.25,0.25,0.25)),
#)


B = 1
scene.build(n_envs=B, env_spacing=(1.0, 1.0))

rigid = scene.sim.rigid_solver
base = franka.get_link("base")
cube_link = []
link_franka = np.array([base.idx], dtype=gs.np_int)
for cube in panels:
    cube.set_pos(cube.get_pos() +  gs.tensor([0, 0, base.get_pos()[0, 2] + 0.2]))
    cube_link = cube.get_link("box_baselink")
    link_cube   = np.array([cube_link.idx],   dtype=gs.np_int)
    rigid.add_weld_constraint(link_cube, link_franka)

for i in range(1_000_000):
    scene.step()