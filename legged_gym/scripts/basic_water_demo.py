import genesis as gs

########################## init ##########################
gs.init()

########################## create a scene ##########################

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt       = 0.002,
        substeps = 5,
    ),
    sph_options=gs.options.SPHOptions(
        lower_bound   = (-0.5, -0.5, 0.0),
        upper_bound   = (0.5, 0.5, 1),
        particle_size = 0.012,
    ),
    vis_options=gs.options.VisOptions(
        visualize_sph_boundary = True,
    ),
    show_viewer = True,
)

########################## entities ##########################
plane = scene.add_entity(
    morph=gs.morphs.Plane(),
)

liquid = scene.add_entity(
    # viscous liquid
    material=gs.materials.SPH.Liquid(rho=850, mu=0.05, gamma=0.003),
    # material=gs.materials.SPH.Liquid(),
    morph=gs.morphs.Box(
        pos  = (0.0, 0.0, 0.65),
        size = (0.5, 0.5, 0.5),
    ),
    surface=gs.surfaces.Default(
        color    = (0.4, 0.8, 1.0),
        vis_mode = 'particle',
    ),
)

########################## build ##########################
scene.build()

horizon = 1000
for i in range(horizon):
    scene.step()