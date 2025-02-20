
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
    "op1": {"size": (2, 3), "duration": 1, "dependencies": ["r1", "r2"], "generate": "r9"},
    "op2": {"size": (2, 3), "duration": 1, "dependencies": ["r3", "r4"], "generate": "r10"},
    "op3": {"size": (2, 3), "duration": 1, "dependencies": ["r5", "r6"], "generate": "r11"},
    "op4": {"size": (2, 3), "duration": 1, "dependencies": ["r7", "r8"], "generate": "r12"},
    "op5": {"size": (2, 3), "duration": 2, "dependencies": ["op1", "op2"], "generate": "r13"},  # 容量6
    "op6": {"size": (3, 4), "duration": 2, "dependencies": ["op3", "op4"], "generate": "r14"},
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
    "r3": (0, 2),
    "r4": (2, 2),
    "r5": (4, 0),
    "r6": (6, 0),
    "r7": (4, 2),
    "r8": (6, 2)
}

env = GridPlacementEnv(grid_size=(10, 10),
                       module_specs=module_specs,
                       reagent_specs=reagent_specs,
                       start_point=start_point)

# 假设你在测试过程中记录了每一步的状态和奖励
states = []
rewards = []

done = False
while not done:
    action = env.action_space.sample()
    next_state, reward, done, truncated, info = env.step(action)
    states.append(next_state)
    rewards.append(reward)
    print(next_state)
    print()

# 可视化奖励
plt.plot(rewards)
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.show()
