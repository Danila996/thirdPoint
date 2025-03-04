import numpy as np
from GridPlacementEnv import GridPlacementEnv
import matplotlib.pyplot as plt
# module_specs = {
#     "op1": {"size": (2, 4), "duration": 20, "dependencies": ["r1", "r2", "r3"]},
#     "op2": {"size": (2, 2), "duration": 20, "dependencies": ["r1", "r3"]},
#     "op3": {"size": (3, 3), "duration": 20, "dependencies": ["op1", "op2"]},
# }
# reagent_specs = {
#     "op1": {
#         "r1": {"cells": 2, "from": "external"},  # 2 cells from r1 for op1
#         "r2": {"cells": 2, "from": "external"},
#         "r3": {"cells": 2, "from": "external"}
#     },
#     "op2": {
#         "r1": {"cells": 2, "from": "external"},  # 2 cells from r1 for op2
#         "r3": {"cells": 2, "from": "external"}
#     },
#     "op3": {
#         "r4": {"cells": 3, "from": "op1"},
#         "r5": {"cells": 3, "from": "op2"},
#     }
# }
# start_point = {
#     "r1": (0, 0),
#     "r2": (5, 0),
#     "r3": (0, 5),
# }
module_specs = {
    "op1": {"size": (2, 3), "duration": 6, "dependencies": ["r1", "r2"], "generate": "r9"},
    "op2": {"size": (2, 3), "duration": 5, "dependencies": ["r3", "r4"], "generate": "r10"},
    "op3": {"size": (2, 3), "duration": 4, "dependencies": ["r5", "r6"], "generate": "r11"},
    "op4": {"size": (2, 3), "duration": 3, "dependencies": ["r7", "r8"], "generate": "r12"},
    "op5": {"size": (2, 3), "duration": 3, "dependencies": ["op1", "op2"], "generate": "r13"},  # 容量6
    "op6": {"size": (3, 4), "duration": 3, "dependencies": ["op3", "op4"], "generate": "r14"},
    "op7": {"size": (4, 5), "duration": 3, "dependencies": ["op5", "op6"], "generate": "r15"},
}
reagent_specs = {
    "op1": {
        "r1": {"cells": 3, "from": "external"},
        "r2": {"cells": 3, "from": "external"}
    },
    "op2": {
        "r3": {"cells": 2, "from": "external"},
        "r4": {"cells": 4, "from": "external"}
    },
    "op3": {
        "r5": {"cells": 4, "from": "external"},
        "r6": {"cells": 2, "from": "external"}
    },
    "op4": {
        "r7": {"cells": 3, "from": "external"},
        "r8": {"cells": 3, "from": "external"}
    },
    "op5": {
        "r9": {"cells": 2, "from": "op1"},   # op1 输出6，op5容量6，只接收2
        "r10": {"cells": 4, "from": "op2"}  # op2 输出6，op5容量6，只接收4
    },
    "op6": {
        "r11": {"cells": 6, "from": "op3"},
        "r12": {"cells": 6, "from": "op4"}
    },
    "op7": {
        "r13": {"cells": 12, "from": "op5"},
        "r14": {"cells": 8, "from": "op6"}
    }
}
start_point = {
    "r1": (0, 0),
    "r2": (2, 0),
    "r3": (0, 4),
    "r4": (0, 6),
    "r5": (5, 0),
    "r6": (7, 0),
    "r7": (9, 0),
    "r8": (8, 0)
}
grid_size = (15, 15)
env = GridPlacementEnv(grid_size=grid_size,
                       module_specs=module_specs,
                       reagent_specs=reagent_specs,
                       start_point=start_point)

# 假设你在测试过程中记录了每一步的状态和奖励
states = []
rewards = []
done = False


def reshape_grid(grid):
    """将一维 grid 转换为二维网格"""
    return np.reshape(grid, grid_size)


while not done:
    action = env.action_space.sample()
    next_state, reward, done, truncated, info = env.step(action)
    states.append(next_state)
    rewards.append(reward)
    # 转换 grid 为二维网格
    grid_2d = reshape_grid(next_state['grid'])
    grid_2d[0][0] = -1
    grid_2d[2][0] = -2
    grid_2d[0][4] = -3
    grid_2d[0][6] = -4
    grid_2d[5][0] = -5
    grid_2d[7][0] = -6
    grid_2d[9][0] = -7
    grid_2d[8][0] = -8

    print("Grid (2D):")
    print(grid_2d)
    print()

# 可视化奖励
plt.plot(rewards)
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.show()
