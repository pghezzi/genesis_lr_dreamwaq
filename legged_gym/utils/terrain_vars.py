terrain_list = [
  "slope",
  "random_uniform",
  "stairs",
  "upwards_stairs"
  "discrete_obstacles",
  "stepping_stones",
  "gap",
  "pit",
  "multiple_high_platforms"
]

TERRAIN_KEYS = terrain_list
TERRAIN_MAP = {
    name: [1 if i == idx else 0 for i in range(len(TERRAIN_KEYS))]
    for idx, name in enumerate(TERRAIN_KEYS)
}

import os

def get_env_vars():
  terrain_name = os.environ.get("TERRAIN", "random_uniform").lower()
  finetune = os.environ.get("FINETUNE", "")


  if terrain_name not in TERRAIN_MAP:
      raise ValueError(f"Unknown TERRAIN '{terrain_name}'. Valid options: {TERRAIN_KEYS}")

  terrain_index = TERRAIN_KEYS.index(terrain_name)
  terrain_list = TERRAIN_MAP[terrain_name]
  return terrain_name, finetune, terrain_index, terrain_list