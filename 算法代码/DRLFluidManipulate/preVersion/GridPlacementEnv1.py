from abc import ABC
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GridPlacementEnv(gym.Env, ABC):
    def __init__(self, grid_size=(10, 10), module_specs=None, reagent_specs=None, start_point=None):
        super(GridPlacementEnv, self).__init__()
        self.grid_size = grid_size
        self.grid = np.zeros(grid_size, dtype=int)  # 0 indicates unoccupied
        self.module_queue = list(module_specs.keys()) if module_specs else []
        self.module_specs = module_specs if module_specs else {}
        self.reagent_specs = reagent_specs if reagent_specs else {}
        self.start_point = start_point if start_point else {}

        # Rewards
        self.reward_placement = 1.0
        self.reward_invalid = -1.0
        self.reward_proximity = 0.1
        self.reward_path_penalty = -0.05
        self.reward_completion = 0.5

        # Action and observation spaces
        self.action_space = spaces.Discrete(grid_size[0] * grid_size[1] * 2)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=1, shape=(grid_size[0], grid_size[1]), dtype=int),
            "current_module": spaces.Box(low=0, high=max(len(self.module_specs), 1), shape=(1,), dtype=int)
        })

        self.current_time = 0
        self.start_time = 0
        self.active_modules = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros(self.grid_size, dtype=int)
        self.current_time = 0
        self.start_time = 0
        self.active_modules = []
        self.module_queue = list(self.module_specs.keys())
        return self._get_observation(), {}

    def step(self, action):
        flat_idx, orientation = divmod(action, 2)
        row, col = divmod(flat_idx, self.grid_size[1])

        self._remove_expired_modules()  # 检查并释放过期模块

        if not self.module_queue:
            return self.grid.copy(), 0, True, False, {}

        module_id = self.module_queue[0]
        reward = 0

        if self._is_valid_placement(row, col, module_id, orientation):
            self._place_module(module_id, row, col, orientation)
            reward += self.reward_placement
            reward += self._calculate_proximity_reward(module_id, row, col)  # 计算父子组件的奖励
            reward += self._plan_reagent_paths(module_id)
            self.module_queue.pop(0)
            self.start_time += 1
            if not self.module_queue:
                reward += self.reward_completion
                return self._get_observation(), reward, True, False, {}
        else:
            reward += self.reward_invalid

        self.current_time += 1
        return self._get_observation(), reward, self._check_done(), self.current_time >= 100, {}

    # 移除已经执行完的组件
    def _remove_expired_modules(self):
        expired_modules = []
        for module in self.active_modules:
            if self.start_time - module["start_time"] >= module["duration"]:
                print(f"{module['id']} have be executed")
                expired_modules.append(module)

        for module in expired_modules:
            row, col, height, width = module["position"]
            for r in range(row, row + height):
                for c in range(col, col + width):
                    self.grid[r, c] = 0
            # self.active_modules.remove(module)

    def _get_observation(self):
        current_module = int(self.module_queue[0][2:]) if self.module_queue else 0
        return {
            "grid": self.grid.copy(),
            "current_module": np.array([current_module])
        }

    def _is_valid_placement(self, row, col, module_id, orientation):
        module_spec = self.module_specs[module_id]
        height, width = module_spec['size']
        if orientation == 1:
            height, width = width, height

        if row + height > self.grid_size[0] or col + width > self.grid_size[1]:
            return False

        for r in range(row, row + height):
            for c in range(col, col + width):
                if self.grid[r, c] != 0:
                    parent_id = next((dep for dep in module_spec.get('dependencies', [])
                                      if any(m['id'] == dep and self.grid[r, c] == int(dep[2:])
                                             for m in self.active_modules)), None)
                    if not parent_id:
                        return False
        return True

    def _place_module(self, module_id, row, col, orientation):
        module_spec = self.module_specs[module_id]
        height, width = module_spec['size']
        if orientation == 1:
            height, width = width, height

        for r in range(row, row + height):
            for c in range(col, col + width):
                self.grid[r, c] = int(module_id[2:])

        self.active_modules.append({
            "id": module_id,
            "position": (row, col, height, width),
            "dependencies": module_spec.get("dependencies", []),
            "start_time": self.start_time,  # 当前时间步
            "duration": module_spec.get("duration", 3)
        })

    # 奖励计算相关方法
    def _calculate_proximity_reward(self, module_id, row, col):
        module_spec = self.module_specs[module_id]
        dependencies = module_spec.get('dependencies', [])
        reward = 0

        for dep in dependencies:
            parent = next((m for m in self.active_modules if m['id'] == dep), None)
            if parent:
                parent_row, parent_col, parent_height, parent_width = parent['position']
                distance = abs(row - parent_row) + abs(col - parent_col)
                reward += self.reward_proximity / (distance + 1)
        return reward

    def _plan_reagent_paths(self, module_id):
        module = next(m for m in self.active_modules if m['id'] == module_id)
        row, col, height, width = module['position']
        total_penalty = 0

        for reagent, spec in self.reagent_specs.get(module_id, {}).items():
            start_point = self.start_point.get(reagent, (0, 0))
            path_length = self._calculate_reagent_distance(start_point, (row, col))
            total_penalty += path_length * self.reward_path_penalty
        return total_penalty

    def _calculate_reagent_distance(self, start, end):
        sx, sy = start
        ex, ey = end
        path_length = abs(sx - ex) + abs(sy - ey)

        # 检查路径冲突
        conflict_penalty = 0
        while sx != ex or sy != ey:
            if sx < ex:
                sx += 1
            elif sx > ex:
                sx -= 1
            elif sy < ey:
                sy += 1
            elif sy > ey:
                sy -= 1

            # 如果路径经过已占用网格，增加冲突惩罚
            if self.grid[sx, sy] != 0:
                conflict_penalty += 1

        return path_length + conflict_penalty

    def _check_done(self):
        return len(self.module_queue) == 0
