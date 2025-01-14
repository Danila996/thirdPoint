import numpy as np
from PlacementEnv import PlacementEnv

def mask_fn(env: PlacementEnv) -> np.ndarray:
    return env.get_mask()