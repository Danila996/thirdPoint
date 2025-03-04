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
        self.module_output_points = {}

        # Rewards
        self.reward_placement = 1.0
        self.reward_invalid = -1.0
        self.reward_proximity = 0.5
        self.reward_path_penalty = -0.05
        self.reward_completion = 0.5
        self.reward_storage_efficiency = 0.2  # 试剂存放均匀性奖励

        # Action and observation spaces
        self.max_reagent_assignments = 4  # 允许的试剂存放模式数
        self.action_space = spaces.MultiDiscrete([
            grid_size[0] * grid_size[1] * 2,  # 组件的位置和方向
            self.max_reagent_assignments  # 试剂存放模式
        ])

        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=1, shape=(self.grid_size[0] * self.grid_size[1],), dtype=int),
            "current_module": spaces.Box(low=0, high=100, shape=(1,), dtype=int)
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
        flat_idx, orientation = divmod(action[0], 2)
        reagent_mode = action[1]  # 试剂存放模式
        row, col = divmod(flat_idx, self.grid_size[1])
        self._remove_expired_modules()  # 检查并释放过期模块
        if not self.module_queue:
            return self.grid.copy(), 0, True, False, {}

        module_id = self.module_queue[0]
        reward = 0

        if self._is_valid_placement(row, col, module_id, orientation):
            self._place_module(module_id, row, col, orientation, reagent_mode)
            reward += self.reward_placement
            print(f"After 'reward_placement' the reward is: {reward}")
            reward += self._calculate_proximity_reward(module_id, row, col)  # 计算父子组件的奖励
            print(f"After '_calculate_proximity_reward' the reward is: {reward}")
            reward += self._plan_reagent_paths(module_id)
            print(f"After '_plan_reagent_paths' the reward is: {reward}")

            # 获取当前组件
            module = next(m for m in self.active_modules if m["id"] == module_id)
            reward += self._calculate_storage_efficiency(module)  # 试剂存放奖励
            print(f"After '_calculate_storage_efficiency' the reward is: {reward}")

            self.module_queue.pop(0)
            self.start_time += 1
            if not self.module_queue:
                reward += self.reward_completion
                return self._get_observation(), reward, True, False, {}
        else:
            reward += self.reward_invalid
            print(f"After 'reward_invalid' the reward is: {reward}")

        print(f"After 'step' the reward is: {reward}")
        self.current_time += 1
        return self._get_observation(), reward, self._check_done(), self.current_time >= 100, {}

    # 移除已经执行完的组件
    def _remove_expired_modules(self):
        expired_modules = []
        for module in self.active_modules:
            if self.start_time - module["start_time"] >= module["duration"]:
                expired_modules.append(module)

        for module in expired_modules:
            row, col, height, width = module["position"]
            center = (row + height / 2, col + width / 2)

            for reagent in self.reagent_specs.get(module["id"], {}):
                if self.reagent_specs[module["id"]][reagent]["from"] == module["id"]:
                    self.start_point[reagent] = center  # 记录中间试剂的起始位置

            for r in range(row, row + height):
                for c in range(col, col + width):
                    self.grid[r, c] = 0
            # self.active_modules.remove(module)

    def _get_observation(self):
        # 将 grid 展平成 1D 向量
        flattened_grid = self.grid.flatten()
        current_module = np.array([int(self.module_queue[0][2:])]) if self.module_queue else np.array([0])

        # 返回字典格式的观测
        return {
            "grid": flattened_grid,
            "current_module": current_module
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

    def _place_module(self, module_id, row, col, orientation, reagent_mode):
        module_spec = self.module_specs[module_id]
        height, width = module_spec['size']
        if orientation == 1:
            height, width = width, height
        for r in range(row, row + height):
            for c in range(col, col + width):
                self.grid[r, c] = int(module_id[2:])
        module = {
            "id": module_id,
            "position": (row, col, height, width),
            "dependencies": module_spec.get("dependencies", []),
            "start_time": self.start_time,  # 当前时间步
            "duration": module_spec.get("duration", 3),
            "reagent_positions": {},
            "reagent_mode": reagent_mode,
        }
        module["reagent_positions"] = self._assign_reagent_distribution(module, reagent_mode)
        self.active_modules.append(module)

    def _assign_reagent_distribution(self, module, reagent_mode):
        """
        根据不同的 reagent_mode 在设备内部均匀分配试剂。
        reagent_mode	试剂存放方式
            0   	    均匀填充设备区域
            1   	    试剂集中在设备左上角
            2   	    试剂集中在设备右下角
            3   	    随机填充设备区域
        """
        row, col, height, width = module["position"]
        occupied_cells = [(r, c) for r in range(row, row + height) for c in range(col, col + width)]

        reagent_distribution = {}  # 存储试剂的具体位置
        assigned_cells = set()  # 记录已分配的网格

        # 获取所有试剂
        reagent_list = list(self.reagent_specs.get(module["id"], {}).items())
        if reagent_mode == 0:  # **均匀填充**
            index = 0
            for reagent, spec in reagent_list:
                reagent_distribution[reagent] = []
                required_cells = spec["cells"]
                while len(reagent_distribution[reagent]) < required_cells:
                    cell = occupied_cells[index % len(occupied_cells)]
                    # print(module, cell, assigned_cells, reagent_distribution[reagent],
                    #       len(reagent_distribution[reagent])
                    #       , required_cells)
                    if cell not in assigned_cells:
                        reagent_distribution[reagent].append(cell)
                        assigned_cells.add(cell)
                    index += 1
        elif reagent_mode == 1:  # **集中在左上角**
            available_cells = occupied_cells.copy()
            for reagent, spec in reagent_list:
                allocated = available_cells[:spec["cells"]]
                reagent_distribution[reagent] = allocated
                assigned_cells.update(allocated)
                available_cells = available_cells[spec["cells"]:]
        elif reagent_mode == 2:  # **集中在右下角**
            available_cells = occupied_cells.copy()
            for reagent, spec in reagent_list:
                allocated = available_cells[-spec["cells"]:]
                reagent_distribution[reagent] = allocated
                assigned_cells.update(allocated)
                available_cells = available_cells[:-spec["cells"]]
        else:  # **默认随机分布**
            np.random.shuffle(occupied_cells)
            index = 0
            for reagent, spec in reagent_list:
                reagent_distribution[reagent] = []
                required_cells = spec["cells"]
                while len(reagent_distribution[reagent]) < required_cells:
                    cell = occupied_cells[index % len(occupied_cells)]
                    # print(module, cell, assigned_cells, reagent_distribution[reagent],
                    #       len(reagent_distribution[reagent])
                    #       , required_cells)
                    if cell not in assigned_cells:
                        reagent_distribution[reagent].append(cell)
                        assigned_cells.add(cell)
                    index += 1

        return reagent_distribution

    # 奖励计算相关方法
    def _calculate_proximity_reward(self, module_id, row, col):
        """
            计算设备父子之间的重叠关系，并给予奖励。
        """
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
        """
        计算试剂流入设备的路径，目标是设备内部的最长路径单元，并给予惩罚。
        """
        module = next(m for m in self.active_modules if m['id'] == module_id)
        row, col, height, width = module['position']
        total_penalty = 0

        for reagent, spec in self.reagent_specs.get(module_id, {}).items():
            start_point = self.start_point.get(reagent, None)
            if start_point is None:
                print("start_point error")
                start_point = (row + height / 2, col + width / 2)

            # 获取试剂的所有可能存储位置
            candidate_positions = module["reagent_positions"].get(reagent, [])
            print(f"module:{module_id} have reagent {candidate_positions}")
            # 选择距离试剂源最近的目标位置
            max_path_length = -float('inf')

            for target in candidate_positions:
                path_length = self._calculate_reagent_distance(start_point, target)
                if path_length > max_path_length:
                    max_path_length = path_length

            # 计算最优路径的奖励/惩罚
            total_penalty += max_path_length * self.reward_path_penalty

        return total_penalty

    def _calculate_reagent_distance(self, start, end):
        sx, sy = start
        ex, ey = end
        path_length = abs(sx - ex) + abs(sy - ey)
        sx, sy = int(start[0]), int(start[1])
        ex, ey = int(end[0]), int(end[1])
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

    def _calculate_storage_efficiency(self, module):
        """
        计算试剂的存放是否有利于流体布线：
        - 如果试剂存放方式减少整体流体路径长度，则奖励
        - 如果试剂存放方式导致路径变长，则惩罚
        - 如果单元分配存在被包围的阻塞情况，则惩罚
        """
        row, col, height, width = module["position"]
        center = (row + height / 2, col + width / 2)
        grid = self.grid.copy()

        total_efficiency = 0
        total_positions = 0
        blockage_penalty = 0

        for reagent, positions in module["reagent_positions"].items():
            # 计算源头到中心点的路径
            start_point = self.start_point.get(reagent, (0, 0))
            dist_to_center = self._calculate_reagent_distance(start_point, center)
            for pos in positions:
                # 计算源头到最终位置的路径
                dist_to_final = self._calculate_reagent_distance(start_point, pos)

                # 计算路径差异，正差值意味着优化成功
                path_diff = dist_to_center - dist_to_final
                total_efficiency += path_diff
                total_positions += 1

                # 检查当前单元是否被包围
                if self._is_cell_surrounded(grid, module, reagent, pos):
                    blockage_penalty += 1  # 每个被包围的单元加1惩罚

        # 归一化：路径差值除以总数
        avg_efficiency = total_efficiency / max(total_positions, 1)

        # 奖励正向优化，惩罚逆向效果
        return avg_efficiency * self.reward_storage_efficiency - blockage_penalty * 0.2

    def _is_cell_surrounded(self, grid, module, reagent, cell):
        """
        判断某个目标单元是否被其他试剂包围。
        逻辑：检查其上下左右是否均为非空试剂单元。
        """
        r, c = cell
        rows, cols = len(grid), len(grid[0])

        # 定义四个方向
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        # 统计被占用的方向数量
        blocked_count = 0

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                # 若相邻单元为其他试剂，视为阻塞
                if grid[nr, nc] == module["id"] and (nr, nc) not in module["reagent_positions"][reagent].item():
                    blocked_count += 1

        # 若三个方向以上均被阻塞，视为被包围
        return blocked_count >= 3

    def _check_done(self):
        return len(self.module_queue) == 0
