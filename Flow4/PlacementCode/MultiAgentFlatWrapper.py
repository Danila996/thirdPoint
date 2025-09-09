# -------------------------------
# 自定义包装器：将多智能体环境的嵌套 dict 观测和 dict 动作平铺为单层 numpy 数组
# -------------------------------
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from abc import ABC


class MultiAgentFlatWrapper(gym.Env, ABC):
    def __init__(self, env):
        super(MultiAgentFlatWrapper, self).__init__()
        self.env = env
        self.agent_ids = env.agent_ids
        # 每个 agent 的观测为：grid (size: grid_size[0]*grid_size[1]) + module_id (1,)
        obs_dim = env.grid_size[0] * env.grid_size[1] + 1
        self.observation_space = spaces.Box(low=0, high=100, shape=(len(self.agent_ids) * obs_dim,), dtype=np.int32)
        # 每个 agent 的动作为 2 个离散数值
        single_action = env.single_agent_action_space.nvec  # 例如 [grid_size*2, max_reagent_assignments]
        self.action_space = spaces.MultiDiscrete(np.tile(single_action, len(self.agent_ids)))

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        flat_obs = self._flatten_obs(obs_dict)
        return flat_obs, info

    def step(self, action):
        # 将平铺的动作转换为 dict 格式
        actions = {}
        for i, agent in enumerate(self.agent_ids):
            actions[agent] = action[i * 2:(i + 1) * 2]
        obs_dict, rewards, dones, truncated, info = self.env.step(actions)
        flat_obs = self._flatten_obs(obs_dict)
        total_reward = sum(rewards.values())
        done = all(dones.values())
        return flat_obs, total_reward, done, truncated, info

    def _flatten_obs(self, obs_dict):
        flat_list = []
        for agent in self.agent_ids:
            obs = obs_dict[agent]
            flat_agent = np.concatenate([obs["grid"].astype(np.int32), obs["module_id"].astype(np.int32)])
            flat_list.append(flat_agent)
        return np.concatenate(flat_list)
