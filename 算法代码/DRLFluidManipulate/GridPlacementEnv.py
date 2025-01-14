import numpy as np
import gym
from gym import spaces


class GridPlacementEnv(gym.Env):
    def __init__(self, grid_size=(10, 10), module_specs=None, reagent_specs=None, start_point=None):
        super(GridPlacementEnv, self).__init__()

        self.grid_size = grid_size
        self.grid = np.zeros(grid_size, dtype=int)  # 0 indicates unoccupied

        self.module_queue = list(module_specs.keys()) if module_specs else []
        self.module_specs = module_specs if module_specs else {}

        self.reagent_specs = reagent_specs if reagent_specs else {}
        self.start_point = start_point if start_point else {}

        #  动作空间用编号实现每个动作，10×10一共有100个动作，比如12表示在(1,2)的动作
        self.action_space = spaces.Discrete(grid_size[0] * grid_size[1]*2)
        #  观测空间使用low=0表示最小值为0，未占用；使用high=1表示最大值为1，占用
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(grid_size[0], grid_size[1]), dtype=int
        )

        self.current_time = 0
        self.active_modules = []

    def reset(self):
        self.grid = np.zeros(self.grid_size, dtype=int)
        self.current_time = 0
        self.active_modules = []
        return self.grid.copy()  # 使用copy()的原因，外部代码修改返回状态可能影响环境内部网络状态

    def step(self, action):
        if not self.module_queue:
            return self.grid.copy(), 0, True, {}  # 所有模块已放置完成，任务结束
        # 根据动作获取坐标
        flat_idx, orientation = divmod(action, 2)
        row, col = divmod(flat_idx, self.grid_size[1])

        module_id = self.module_queue[0]  # 取队列中的第一个模块
        print(module_id)

        if not self._is_valid_placement(row, col, module_id, orientation):  # 布局失败，返回状态和奖励
            return self.grid.copy(), -1, False, {}

        # 计算放置奖励
        reward = self._calculate_placement_reward(row, col, module_id, orientation)

        self._place_module(module_id, row, col, orientation)

        reward += self._plan_reagent_paths(module_id)

        self.current_time += 1
        self._remove_expired_modules()

        done = self._check_done()
        self.module_queue.pop(0)

        return self.grid.copy(), reward, done, {}

    def _is_parent_occupied(self, r, c, module_id):
        """
        判断给定位置 (r, c) 是否被父模块占用。
        """
        module_spec = self.module_specs[module_id]
        if 'dependencies' not in module_spec:
            return False  # 如果没有父模块依赖，直接返回 False

        parent_modules = [dep for dep in module_spec['dependencies'] if dep.startswith("op")]

        for parent_id in parent_modules:
            parent_module = next(m for m in self.active_modules if m['id'] == parent_id)
            parent_row, parent_col, parent_height, parent_width = parent_module['position']

            # 判断 (r, c) 是否在父模块的占用区域内
            if parent_row <= r < parent_row + parent_height and parent_col <= c < parent_col + parent_width:
                return True  # 该位置被父模块占用

        return False  # 该位置没有被父模块占用

    def _is_valid_placement(self, row, col, module_id, orientation):
        #  加一个父子关系组件

        module_spec = self.module_specs[module_id]
        height, width = module_spec['size']

        if orientation == 1:
            height, width = width, height

        if row + height > self.grid_size[0] or col + width > self.grid_size[1]:
            return False

        # 检查依赖是否满足
        if not self._can_place_module(module_id):
            return False  # 如果依赖未满足，返回 False

        for r in range(row, row + height):
            for c in range(col, col + width):
                if self.grid[r, c] != 0:
                    # 如果此位置已经被占用，检查是否是父子组件之间的重叠
                    if not self._is_parent_occupied(r, c, module_id):
                        return False

        return True

    def _can_place_module(self, module_id):
        module_spec = self.module_specs[module_id]
        dependencies = module_spec.get("dependencies", [])
        print(dependencies)
        print(self.active_modules)

        for dep in dependencies:
            # 如果依赖是操作，则检查对应模块是否已完成
            if dep.startswith("op"):
                if not any(m["id"] == dep for m in self.active_modules):
                    return False  # 依赖模块未完成
            # 如果依赖是试剂，跳过检查（已在路径规划中处理）
        return True

    def _place_module(self, module_id, row, col, orientation):
        module_spec = self.module_specs[module_id]
        height, width = module_spec['size']

        # Adjust size based on orientation
        if orientation == 1:  # Rotate 90 degrees
            height, width = width, height

        for r in range(row, row + height):
            for c in range(col, col + width):
                self.grid[r, c] = int(module_id[2:])

        self.active_modules.append({
            "id": module_id,
            "start_time": self.current_time,
            "duration": module_spec['duration'],
            "position": (row, col, height, width),
            "dependencies": module_spec.get('dependencies', []),
        })

    def _calculate_placement_reward(self, row, col, module_id, orientation):
        """
        计算模块放置后的奖励。
        奖励与父模块的相对位置和路径长度有关。
        """
        # 获取模块规格
        module_spec = self.module_specs[module_id]
        height, width = module_spec['size']

        # 如果模块方向被旋转，交换高度和宽度
        if orientation == 1:
            height, width = width, height

        # 计算奖励初始化值
        reward = 0

        # 如果模块有依赖的其他模块（如父模块），确保子组件放置在父模块附近
        if "dependencies" in module_spec:
            parent_modules = [dep for dep in module_spec['dependencies'] if dep.startswith("op")]

            for parent_id in parent_modules:
                parent_module = next(m for m in self.active_modules if m['id'] == parent_id)
                parent_row, parent_col, parent_height, parent_width = parent_module['position']

                # 计算子模块与父模块的相对位置，并给奖励
                reward += self._calculate_proximity_reward(row, col, height, width, parent_row, parent_col,
                                                           parent_height, parent_width)

        return reward

    def _calculate_proximity_reward(self, child_row, child_col, child_height, child_width, parent_row, parent_col,
                                    parent_height, parent_width):
        """
        计算子模块与父模块之间的距离奖励，距离越近奖励越高。
        """
        # 计算子模块和父模块的边界坐标
        child_bottom = child_row + child_height
        child_right = child_col + child_width
        parent_bottom = parent_row + parent_height
        parent_right = parent_col + parent_width

        # 计算子模块与父模块之间的距离
        # 如果子模块与父模块重叠或紧邻，距离为0（最佳情况）
        vertical_distance = max(0, max(parent_row - child_bottom, child_row - parent_bottom))
        horizontal_distance = max(0, max(parent_col - child_right, child_col - parent_right))

        # 计算总距离（曼哈顿距离）
        distance = vertical_distance + horizontal_distance

        # 计算奖励：距离越小，奖励越高
        reward = 1 / (distance + 1)  # 使用 +1 避免除以零的错误

        return reward

    def _plan_reagent_paths(self, module_id):
        module = next(m for m in self.active_modules if m['id'] == module_id)
        row, col, height, width = module['position']

        reagents = self.reagent_specs.get(module_id, {})
        total_path_length = 0

        for reagent, spec in reagents.items():
            if spec['from'] == "external":
                start_point = self.start_point[reagent]
            else:
                dependency_module = next(
                    m for m in self.active_modules if m['id'] == spec['from']
                )
                start_point = dependency_module['position'][:2]

            for _ in range(spec['cells']):  # 这个奖励也有点问题
                path_length = self._calculate_path_length(start_point, (row, col))
                total_path_length += path_length

        return -total_path_length

    def _calculate_path_length(self, start, end):
        return abs(start[0] - end[0]) + abs(start[1] - end[1])

    def _remove_expired_modules(self):
        expired = [m for m in self.active_modules if self.current_time - m['start_time'] >= m['duration']]
        for module in expired:
            # row, col, height, width = (module['position']["row"], module['position']["col"], module['position'][
            # "height"], module['position']["width"])
            row, col, height, width = module['position']
            for r in range(row, row + height):
                for c in range(col, col + width):
                    self.grid[r, c] = 0

        self.active_modules = [m for m in self.active_modules if m not in expired]

    def _check_done(self):
        return self.current_time >= 100