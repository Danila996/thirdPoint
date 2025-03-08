import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from stable_baselines3 import PPO
from GridRoutingEnv import GridRoutingEnv
from sb3_contrib import MaskablePPO

base_grid = np.zeros((10, 10), dtype=np.int32)
task1 = {
    "id": 1,
    "source_area": {(1, 1), (1, 2), (2, 1), (2, 2)},
    "target_area": {(1, 7), (2, 7), (1, 8)},
    "required_volume": 3,
}
# task2 = {
#     "id": 2,
#     "source_area": {(7, 1), (7, 2), (8, 1), (8, 2)},
#     "target_area": {(7, 7), (7, 8), (8, 7)},
#     "required_volume": 3,
# }
tasks = [task1]

# 创建环境实例
env = GridRoutingEnv(base_grid, tasks, max_steps=200)


model = MaskablePPO.load("ppo_grid_routing.zip")

obs, _ = env.reset()
done = False
total_reward = 0
step_count = 0
mask_action = []
while not done:
    mask_action = env.get_action_masks()
    print(f"mask_action: {mask_action}")

    action, _ = model.predict(obs, action_masks=mask_action)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    step_count += 1
    env.render()

print(f"🎯 测试完成：总奖励 = {total_reward:.2f}")
print(f"🎯 模型推断完成，共 {step_count} 步。")
