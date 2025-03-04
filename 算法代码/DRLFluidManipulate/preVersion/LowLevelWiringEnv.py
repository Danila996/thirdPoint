import numpy as np
import gymnasium as gym
from gymnasium import spaces
from abc import ABC


class LowLevelWiringEnv(gym.Env, ABC):
    """
    低层环境：针对单个设备的布线规划
    输入：设备的详细信息（包括内部试剂候选位置）和各试剂的起点（从高层环境传入）
    动作：对于设备中每个试剂，选择一个候选位置的索引
    奖励：计算所有试剂从起点到所选候选位置的曼哈顿距离平均值（取负值），距离越短奖励越高
    """

    def __init__(self, device, start_points):
        """
        device: dict，表示某设备的详细信息（来自高层环境 active_modules），必须包含 "reagent_positions"
        start_points: dict，映射试剂名称 -> 起点坐标，例如 {"r1": (x1,y1), "r2": (x2,y2), ...}
        """
        super(LowLevelWiringEnv, self).__init__()
        self.device = device
        self.start_points = start_points
        # 所有试剂名称
        self.reagents = list(device["reagent_positions"].keys())

        # 动作空间：对于每个试剂，选择候选位置的索引
        action_dims = [len(device["reagent_positions"][r]) for r in self.reagents]
        self.action_space = spaces.MultiDiscrete(action_dims)

        # 观测空间：这里简单构造每个试剂的观测为其候选位置列表和起点（用一个字典表示）
        obs_spaces = {}
        # 为了统一维度，我们可以求各试剂候选位置数的最大值
        self.max_candidates = max(len(device["reagent_positions"][r]) for r in self.reagents)
        for r in self.reagents:
            # 每个试剂的候选位置为一个二维数组 (max_candidates, 2) ，不足的部分填 0
            obs_spaces[r] = spaces.Dict({
                "candidates": spaces.Box(low=0, high=100, shape=(self.max_candidates, 2), dtype=np.int32),
                "start": spaces.Box(low=0, high=100, shape=(2,), dtype=np.int32)
            })
        self.observation_space = spaces.Dict(obs_spaces)

    def reset(self):
        # 构造每个试剂的观测，候选位置不足的部分填 0
        obs = {}
        for r in self.reagents:
            candidates = self.device["reagent_positions"][r]
            padded = np.zeros((self.max_candidates, 2), dtype=np.int32)
            padded[:len(candidates)] = np.array(candidates, dtype=np.int32)
            start = np.array(self.start_points.get(r, (0, 0)), dtype=np.int32)
            obs[r] = {"candidates": padded, "start": start}
        return obs, {}

    def step(self, action):
        """
        action: 一个数组，每个元素是一个索引，对应于每个试剂的候选位置选择
        """
        total_distance = 0.0
        for i, r in enumerate(self.reagents):
            idx = action[i]
            candidates = self.device["reagent_positions"][r]
            if idx >= len(candidates):
                idx = len(candidates) - 1
            chosen = candidates[idx]
            start = self.start_points.get(r, (0, 0))
            distance = abs(chosen[0] - start[0]) + abs(chosen[1] - start[1])
            total_distance += distance
        avg_distance = total_distance / len(self.reagents)
        reward = -avg_distance  # 距离越短，奖励越高（负距离惩罚）
        done = True  # 低层任务一次完成
        obs, _ = self.reset()  # 低层任务完成后重新生成观测（或可以直接返回当前 obs）
        return obs, reward, done, False, {}
