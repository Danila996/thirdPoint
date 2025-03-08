import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO

# from ActionMaskObservationWrapper import ActionMaskObservationWrapper
from GridRoutingEnv import GridRoutingEnv

base_grid = np.zeros((10, 10), dtype=np.int32)
task1 = {
    "id": 1,
    "source_area": {(1, 1), (1, 2), (2, 1), (2, 2)},
    "target_area": {(1, 7), (2, 7), (1, 8)},
    "required_volume": 3,
}
task2 = {
    "id": 2,
    "source_area": {(7, 1), (7, 2), (8, 1), (8, 2)},
    "target_area": {(7, 7), (7, 8), (8, 7)},
    "required_volume": 3,
}
tasks = [task1, task2]

# 创建环境实例
env = GridRoutingEnv(base_grid, tasks, max_steps=200)
env = ActionMasker(env, lambda env: env.get_action_masks())

env = Monitor(env)

check_env(env)

# 配置检查点回调，每 1000 步保存一次模型
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./models/', name_prefix='ppo_grid_routing')

tensorboard_log_dir = "./ppo_tensorboard_routing/"
# 使用 PPO 训练
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = MaskablePPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log_dir, device=device)

# 开始训练，例如 150,000 个时间步
model.learn(total_timesteps=150000, callback=checkpoint_callback)
model.save("ppo_grid_routing")

# 测试训练后的模型
obs, _ = env.reset()
done = False
while not done:
    action_masks = env.get_action_masks()
    action, _ = model.predict(obs, action_masks=action_masks)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
