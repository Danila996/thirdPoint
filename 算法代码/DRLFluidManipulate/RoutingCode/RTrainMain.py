from datetime import date
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO

from GridRoutingEnv import GridRoutingEnv
from CustomCombinedExtractor import CustomExtractor_routing
from mask_fn import get_mask

base_grid = np.zeros((10, 10), dtype=np.int32)
task1 = {
    "id": 1,
    "source_area": {(1, 1), (1, 2), (2, 1), (2, 2)},
    "target_area": {(1, 7), (2, 7), (1, 8)},
    "required_volume": 3,
}
tasks = [task1]


# 创建环境实例的函数，也定义在模块顶层
def make_env():
    def _init():
        env = GridRoutingEnv(base_grid, tasks, max_steps=200)
        env = ActionMasker(env, get_mask)
        env = Monitor(env, f'routing_ppo_log_{date.today()}')
        check_env(env)
        return env

    return _init


if __name__ == "__main__":
    num_envs = 40
    envs = [make_env() for _ in range(num_envs)]
    env = SubprocVecEnv(envs)

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='../models/', name_prefix='ppo_grid_routing')

    policy_kwargs = dict(
        features_extractor_class=CustomExtractor_routing,
    )

    tensorboard_log_dir = "./ppo_tensorboard_routing/"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = MaskablePPO("MultiInputPolicy",
                        env,
                        policy_kwargs=policy_kwargs,
                        verbose=1,
                        tensorboard_log=tensorboard_log_dir,
                        n_epochs=5,
                        n_steps=1024,
                        device=device)

    model.learn(total_timesteps=2000000, callback=checkpoint_callback)
    model.save("ppo_grid_routing_single2")
