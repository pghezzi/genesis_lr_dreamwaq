import numpy as np
import genesis as gs

gs.init(seed=0, backend=gs.gpu)

scene = gs.Scene(show_viewer=True,
)

franka = scene.add_entity(
    gs.morphs.URDF(
        pos = (6, 6, 1),
        file='/home/pablo/Documents/work/w/genesis_lr_dreamwaq/resources/robots/go1/urdf/go1.urdf'),
)
   

#scene.add_entity(
#                gs.morphs.Plane())

#terrain = scene.add_entity(
#    morph=gs.morphs.Terrain(
#        pos=(-10,-10, 0),
#        subterrain_size=(6.0, 6.0),
#        horizontal_scale=0.25,
#        #vertical_scale=0.005,
#    ),
#)

hf = np.load('height_field_raw.npy')

terrain = scene.add_entity(
    morph=gs.morphs.Terrain(
        height_field=hf,
        vertical_scale=-0.005,
    )
)

[['flat_terrain', 'random_uniform_terrain', 'stepping_stones_terrain'], ['pyramid_sloped_terrain', 'discrete_obstacles_terrain', 'wave_terrain'], ['random_uniform_terrain', 'pyramid_stairs_terrain', 'sloped_terrain']]
scene.build(n_envs=1, env_spacing=(6.0, 6.0))

for _ in range(10_000):
    scene.step()