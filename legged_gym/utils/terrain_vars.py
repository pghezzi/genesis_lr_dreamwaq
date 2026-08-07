import os

TERRAIN_KEYS =[
  "slope",
  "random_uniform",
  "stairs",
  "upwards_stairs",
  "discrete_obstacles",
  "stepping_stones",
  "gap",
  "pit",
  "multiple_high_platforms",
  "high_platform_gaps",
  "center_platform"
  
]

TERRAIN_KEYS.append("plane")

TERRAIN_INDEX = {name: idx for idx, name in enumerate(TERRAIN_KEYS)}

def one_hot(idx):
    v = [0] * len(TERRAIN_KEYS)
    v[idx] = 1
    return v


def get_env_vars():
    terrain_name = os.environ.get("TERRAIN", "random_uniform").lower()
    finetune = os.environ.get("FINETUNE", "")

    if terrain_name == "baseline":
        terrain_list = [0] * len(TERRAIN_KEYS)
        terrain_list[TERRAIN_INDEX["slope"]] = 0.5
        terrain_list[TERRAIN_INDEX["random_uniform"]] = 0.5
        terrain_index = None
    
    elif terrain_name == "plane":
        terrain_list = [0] * len(TERRAIN_KEYS)
        terrain_index = -1

    elif terrain_name == "all_stairs":
        terrain_list = [0] * len(TERRAIN_KEYS)
        terrain_list[TERRAIN_INDEX["stairs"]] = 0.5
        terrain_list[TERRAIN_INDEX["upwards_stairs"]] = 0.5
        terrain_index = None

    elif terrain_name in TERRAIN_INDEX:
        terrain_index = TERRAIN_INDEX[terrain_name]
        terrain_list = one_hot(terrain_index)

    else:
        raise ValueError(
            f"Unknown TERRAIN '{terrain_name}'. "
            f"Valid options: {list(TERRAIN_INDEX)}"
        )

    return terrain_name, finetune, terrain_index, terrain_list