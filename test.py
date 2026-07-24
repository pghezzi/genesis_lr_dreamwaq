import genesis as gs

# Initialize Genesis
gs.init()

# Create scene
scene = gs.Scene(show_viewer=True)

# Add the obstacle course mesh
_gs_terrain = scene.add_entity(
    gs.morphs.Mesh(
        file="/home/pablo/Downloads/limbo_course_with_floor.stl",
        fixed=True
    )
)

# Build the scene
scene.build()

print(_gs_terrain.geoms[0].get_trimesh())

# Run
while True:
    scene.step()
