
from GridPlacementEnv import GridPlacementEnv
import matplotlib.pyplot as plt

start_point = {
    "r1": (0, 0),
    "r2": (5, 0),
    "r3": (0, 5),
}
env = GridPlacementEnv(grid_size=(10, 10), module_specs={
    "op1": {"size": (2, 4), "duration": 20, "dependencies": ["r1", "r2", "r3"]},
    "op2": {"size": (2, 2), "duration": 20, "dependencies": ["r1", "r3"]},
    "op3": {"size": (3, 3), "duration": 20, "dependencies": ["op1", "op2"]},
}, reagent_specs={
    "op1": {
        "r1": {"cells": 2, "from": "external"},  # 2 cells from r1 for op1
        "r2": {"cells": 2, "from": "external"},
        "r3": {"cells": 2, "from": "external"}
    },
    "op2": {
        "r1": {"cells": 2, "from": "external"},  # 2 cells from r1 for op2
        "r3": {"cells": 2, "from": "external"}
    },
    "op3": {
        "r4": {"cells": 3, "from": "op1"},
        "r5": {"cells": 3, "from": "op2"},
    }
}, start_point=start_point)

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
