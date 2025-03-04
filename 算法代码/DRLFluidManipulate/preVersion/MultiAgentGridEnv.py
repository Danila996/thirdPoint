import numpy as np
import gymnasium as gym
from gymnasium import spaces
from abc import ABC


class MultiAgentGridEnv(gym.Env, ABC):
    def __init__(self, grid_size=(10, 10), module_specs=None, reagent_specs=None, start_point=None):
        super(MultiAgentGridEnv, self).__init__()
        self.grid_size = grid_size
        self.grid = np.zeros(grid_size, dtype=int)  # 0 表示空闲

        self.module_specs = module_specs if module_specs else {}
        self.reagent_specs = reagent_specs if reagent_specs else {}
        self.start_point = start_point if start_point else {}

        # 每个模块对应一个 agent，例如 "op1", "op2", ...
        self.agent_ids = list(self.module_specs.keys())
        # 存储已放置模块的信息（处于 running 或 finished 状态）
        self.active_modules = {}
        # 存储每个模块的状态："not_started", "ready", "running", "finished"
        self.module_status = {agent: "not_started" for agent in self.agent_ids}

        # 奖励参数
        self.reward_placement = 1.0
        self.reward_invalid = -2.0
        self.proximity_reward_alpha = 0.1
        self.proximity_reward_beta = 0.05
        self.proximity_reward_beta_pair = 0.5
        self.reward_proximity = 2.0  # 非依赖模块之间的距离奖励基准
        self.reward_path_penalty = -0.05  # wiring 路径长度惩罚
        self.reward_storage_efficiency = 0.2
        self.reward_blockage_penalty = 0.2
        self.reward_conflict_penalty = -0.1

        # 依赖顺序奖励参数
        self.reward_ordering_penalty = -5.0  # 如果依赖（父模块）未完成，给予较大负奖励
        self.reward_ordering_bonus = 2.0  # 若父子靠近，给予奖励
        self.ordering_distance_threshold = 3.0

        # 每个模块有自己的 duration（执行时间）
        # 定义每个 agent 的动作空间：第一个数字决定位置（平铺索引与旋转），第二个数字选择试剂分布模式
        self.max_reagent_assignments = 4
        self.single_agent_action_space = spaces.MultiDiscrete([
            grid_size[0] * grid_size[1] * 2,
            self.max_reagent_assignments
        ])

        # 观测空间：每个 agent 的观测为包含全局 grid（展平为一维向量）和 module_id（单个数值）
        self.single_agent_observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=100, shape=(grid_size[0] * grid_size[1],), dtype=int),
            "module_id": spaces.Box(low=0, high=100, shape=(1,), dtype=int)
        })
        # 多智能体环境的整体空间为 dict 格式
        self.action_space = spaces.Dict({agent: self.single_agent_action_space for agent in self.agent_ids})
        self.observation_space = spaces.Dict({agent: self.single_agent_observation_space for agent in self.agent_ids})

        self.current_iter = 1
        self.error_time = 0
        self.start_time = 0
        self.level = {}
        self.max_steps = 50  # 最大时间步数

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros(self.grid_size, dtype=int)
        self.current_iter = 1
        self.start_time = 0
        self.error_time = 0
        self.active_modules = {}
        self.module_status = {agent: "not_started" for agent in self.agent_ids}
        self.level = {agent: 0 for agent in self.agent_ids}
        self._update_ready_modules()
        observations = {agent: self._get_observation(agent) for agent in self.agent_ids}
        return observations, {}

    def _remove_expired_modules(self):
        expired = []
        for agent, module in self.active_modules.items():
            if self.current_iter - self.error_time - module['start_time'] > module['duration']:
                row, col, height, width = module['position']
                for r in range(row, row + height):
                    for c in range(col, col + width):
                        self.grid[r, c] = 0
                expired.append(agent)
            if module["generate"] not in self.start_point:
                row, col, height, width = module["position"]
                center = (row + height / 2, col + width / 2)
                self.start_point[module["generate"]] = center  # 记录中间试剂的起始位置

        for agent in expired:
            self.module_status[agent] = "finished"
            # del self.active_modules[agent]

    def _update_ready_modules(self):
        # 对于 "not_started" 状态的模块，检查依赖（外部依赖默认满足；对于 "op" 开头的依赖需 finished）
        for agent in self.agent_ids:
            if self.module_status[agent] != "not_started":
                continue
            deps = self.module_specs[agent].get("dependencies", [])
            ready = True
            for dep in deps:
                if dep.startswith("op"):
                    if self.module_status.get(dep, "not_started") != "finished":
                        ready = False
                        break
            if ready:
                self.module_status[agent] = "ready"

    def step(self, actions):
        # 1. 先释放已完成模块
        self._remove_expired_modules()
        # 2. 更新 ready 状态
        self._update_ready_modules()
        error_time = False
        rewards = {agent: 0.0 for agent in self.agent_ids}
        infos = {agent: {} for agent in self.agent_ids}

        # 3. 对于处于 "ready" 状态的模块，尝试放置
        for agent in self.agent_ids:
            if self.module_status[agent] != "ready":
                continue
            if agent not in actions:
                continue
            # 动作解析
            action = actions[agent]
            placement_val, reagent_mode = action
            flat_idx, orientation = divmod(placement_val, 2)
            row, col = divmod(flat_idx, self.grid_size[1])

            module_spec = self.module_specs[agent]
            height, width = module_spec['size']
            if orientation == 1:
                height, width = width, height

            if row + height > self.grid_size[0] or col + width > self.grid_size[1]:
                rewards[agent] += self.reward_invalid
                error_time = True
                continue

            new_cells = {(r, c) for r in range(row, row + height) for c in range(col, col + width)}
            conflict = False
            for (r, c) in new_cells:
                if self.grid[r, c] != 0:
                    existing_numeric = self.grid[r, c]
                    allowed = False
                    for placed_agent, module in self.active_modules.items():
                        if int(placed_agent[2:]) == existing_numeric:
                            if placed_agent in module_spec.get("dependencies", []):
                                allowed = True
                            break
                    if not allowed:
                        conflict = True
                        break
            if conflict:
                rewards[agent] += self.reward_invalid
                error_time = True
                continue

            dep_reward = self._dependency_ordering_reward(agent, row, col)
            rewards[agent] += dep_reward

            self._place_module(agent, row, col, orientation, reagent_mode)
            rewards[agent] += self.reward_placement
            # print(f"{agent} after reward_placement: {rewards[agent]}")
            rewards[agent] += self._calculate_proximity_reward(agent, row, col)
            # print(f"{agent} after _calculate_proximity_reward: {rewards[agent]}")

            wiring_penalty = self._plan_reagent_paths(agent)
            rewards[agent] += wiring_penalty
            # print(f"{agent} after wiring_penalty: {rewards[agent]}")
            storage_reward = self._calculate_storage_efficiency(agent)
            rewards[agent] += storage_reward
            # print(f"{agent} after storage_reward: {rewards[agent]}")
            # print(f"The position of {agent} is {row} , {col}, {orientation}")
            # print(f"agent:{agent}:{self.module_specs[agent]} statues is {self.module_status[agent]}")
            self.start_time += 1
        if error_time:
            self.error_time += 1
        # print(f"error_time: {self.error_time}")
        # print(f"real_time = current_iter({self.current_iter}) - error_time({self.error_time}): {self.current_iter - self.error_time}")


        # 4. 全局布线路径冲突检查
        # wiring_usage = {}
        # for agent, module in self.active_modules.items():
        #     if "wiring_paths" not in module:
        #         continue
        #     for reagent, path in module["wiring_paths"].items():
        #         for cell in path:
        #             wiring_usage.setdefault(cell, []).append((agent, reagent))
        # for cell, users in wiring_usage.items():
        #     if len(users) > 1:
        #         for agent, _ in users:
        #             rewards[agent] += self.reward_conflict_penalty

        # 5. 更新时间
        self.current_iter += 1
        all_done = all(self.module_status[agent] == "finished" for agent in self.agent_ids)
        terminated = all_done or (self.current_iter >= self.max_steps)
        dones = {agent: terminated for agent in self.agent_ids}
        observations = {agent: self._get_observation(agent) for agent in self.agent_ids}
        # 返回 5 个值：obs, reward, terminated, truncated, info
        return observations, rewards, dones, False, {}

    def _get_observation(self, agent):
        flattened_grid = self.grid.flatten()
        try:
            module_num = int(agent[2:]) if agent.startswith("op") else 0
        except:
            module_num = 0
        return {"grid": flattened_grid.copy(), "module_id": np.array([module_num], dtype=int)}

    def _dependency_ordering_reward(self, agent, row, col):
        module_spec = self.module_specs[agent]
        deps = module_spec.get("dependencies", [])
        reward = 0.0
        child_center = (row + module_spec['size'][0] / 2, col + module_spec['size'][1] / 2)
        for dep in deps:
            if dep.startswith("op"):
                if self.module_status.get(dep, "not_started") != "finished":
                    reward += self.reward_ordering_penalty
                else:
                    # 如有需要，可根据父模块位置奖励靠近程度
                    pass
        return reward

    def _place_module(self, agent, row, col, orientation, reagent_mode):
        module_spec = self.module_specs[agent]
        height, width = module_spec['size']
        if orientation == 1:
            height, width = width, height
        numeric_id = int(agent[2:]) if agent.startswith("op") else 0
        new_cells = [(r, c) for r in range(row, row + height) for c in range(col, col + width)]
        for (r, c) in new_cells:
            self.grid[r, c] = numeric_id
        module = {
            "id": agent,
            "position": (row, col, height, width),
            "dependencies": module_spec.get("dependencies", []),
            "generate": module_spec.get("generate", None),
            "start_time": self.level[agent],
            "duration": module_spec.get("duration", 3),
            "reagent_mode": reagent_mode,
            "reagent_positions": {},
            "wiring_paths": {}
        }
        module["reagent_positions"] = self._assign_reagent_distribution(module, reagent_mode)
        self.active_modules[agent] = module
        self.module_status[agent] = "running"

    def _assign_reagent_distribution(self, module, reagent_mode):
        row, col, height, width = module["position"]
        occupied_cells = [(r, c) for r in range(row, row + height) for c in range(col, col + width)]
        reagent_distribution = {}
        assigned_cells = set()
        reagent_list = list(self.reagent_specs.get(module["id"], {}).items())
        if reagent_mode == 0:  # 均匀分布
            index = 0
            for reagent, spec in reagent_list:
                reagent_distribution[reagent] = []
                required_cells = spec["cells"]
                while len(reagent_distribution[reagent]) < required_cells:
                    cell = occupied_cells[index % len(occupied_cells)]
                    if cell not in assigned_cells:
                        reagent_distribution[reagent].append(cell)
                        assigned_cells.add(cell)
                    index += 1
        elif reagent_mode == 1:  # 左上集中
            available_cells = occupied_cells.copy()
            for reagent, spec in reagent_list:
                allocated = available_cells[:spec["cells"]]
                reagent_distribution[reagent] = allocated
                assigned_cells.update(allocated)
                available_cells = available_cells[spec["cells"]:]
        elif reagent_mode == 2:  # 右下集中
            available_cells = occupied_cells.copy()
            for reagent, spec in reagent_list:
                allocated = available_cells[-spec["cells"]:]
                reagent_distribution[reagent] = allocated
                assigned_cells.update(allocated)
                available_cells = available_cells[:-spec["cells"]]
        else:  # 随机分布
            np.random.shuffle(occupied_cells)
            index = 0
            for reagent, spec in reagent_list:
                reagent_distribution[reagent] = []
                required_cells = spec["cells"]
                while len(reagent_distribution[reagent]) < required_cells:
                    cell = occupied_cells[index % len(occupied_cells)]
                    if cell not in assigned_cells:
                        reagent_distribution[reagent].append(cell)
                        assigned_cells.add(cell)
                    index += 1
        return reagent_distribution

    def _calculate_proximity_reward(self, agent, row, col):
        module_spec = self.module_specs[agent]
        deps = module_spec.get("dependencies", [])
        reward = 0.0
        # 子模块的边界：(top, left, bottom, right)
        child_box = (row, col, row + module_spec['size'][0], col + module_spec['size'][1])
        # 子模块中心
        child_center = (row + module_spec['size'][0] / 2, col + module_spec['size'][1] / 2)
        # 存放已放置的父组件中心
        parent_centers = []

        for dep in deps:
            if dep.startswith("op"):
                # 还需要计算重叠数量，然后适当调整奖励
                parent = self.active_modules.get(dep, None)
                if parent:
                    p_row, p_col, p_height, p_width = parent["position"]
                    parent_box = (p_row, p_col, p_row + p_height, p_col + p_width)
                    parent_center = (p_row + p_height / 2, p_col + p_width / 2)
                    parent_centers.append(parent_center)

                    # 计算重叠区域
                    overlap_width = max(0, min(child_box[3], parent_box[3]) - max(child_box[1], parent_box[1]))
                    overlap_height = max(0, min(child_box[2], parent_box[2]) - max(child_box[0], parent_box[0]))
                    overlap_area = overlap_width * overlap_height

                    # 计算中心距离（曼哈顿距离）
                    distance = abs(child_center[0] - parent_center[0]) + abs(child_center[1] - parent_center[1])

                    # 如果有重叠，则奖励重叠单元数；否则，根据距离给予惩罚
                    if overlap_area > 0:
                        reward += self.proximity_reward_alpha * overlap_area
                    else:
                        reward -= self.proximity_reward_beta * distance
                else:
                    # 如果依赖的父模块还未放置，则给予较大负奖励
                    reward += self.reward_ordering_penalty
        # 如果有多个父组件，鼓励它们彼此靠近
        if len(parent_centers) >= 2:
            pair_rewards = []
            for i in range(len(parent_centers)):
                for j in range(i + 1, len(parent_centers)):
                    d = abs(parent_centers[i][0] - parent_centers[j][0]) + abs(
                        parent_centers[i][1] - parent_centers[j][1])
                    pair_rewards.append(1.0 / (d + 1))  # 距离越近，奖励越高
            # 取平均
            if pair_rewards:
                reward += self.proximity_reward_beta_pair * (sum(pair_rewards) / len(pair_rewards))
        return reward

    def _plan_reagent_paths(self, agent):
        module = self.active_modules[agent]
        row, col, height, width = module["position"]
        total_penalty = 0.0
        wiring_paths = {}

        for reagent, spec in self.reagent_specs.get(module["id"], {}).items():
            total_length = 0.0
            start_point = self.start_point.get(reagent, None)
            if start_point is None:
                print("error start_point")
                start_point = (row + height / 2, col + width / 2)
                self.start_point[reagent] = start_point
            candidate_positions = module["reagent_positions"].get(reagent, [])
            max_path_length = -float('inf')
            chosen_path = []
            for target in candidate_positions:
                path, path_length = self._get_manhattan_path(start_point, target)
                # print(f"{module['id']} from {start_point} to {target} ({reagent}) is {path_length}")
                if path_length > max_path_length:
                    max_path_length = path_length
                    chosen_path = path
                total_length += path_length
            average_length = total_length / len(candidate_positions)
            # print(f"{reagent} average_length is {average_length}")
            wiring_paths[reagent] = chosen_path
            total_penalty += average_length * self.reward_path_penalty
            # print(f"{reagent} total_penalty is {total_penalty}")

        module["wiring_paths"] = wiring_paths
        return total_penalty

    def _get_manhattan_path(self, start, end):
        path = []
        sx, sy = int(round(start[0])), int(round(start[1]))
        ex, ey = int(round(end[0])), int(round(end[1]))
        x, y = sx, sy
        while x != ex:
            path.append((x, y))
            x += 1 if x < ex else -1
        while y != ey:
            path.append((x, y))
            y += 1 if y < ey else -1
        path.append((ex, ey))
        path_length = abs(sx - ex) + abs(sy - ey)
        return path, path_length

    def _calculate_storage_efficiency(self, agent):
        module = self.active_modules[agent]
        row, col, height, width = module["position"]
        center = (row + height / 2, col + width / 2)
        avg_efficiency = 0.0
        blockage_penalty = 0.0
        for reagent, positions in module["reagent_positions"].items():
            start_point = self.start_point.get(reagent, (0, 0))
            total_efficiency = 0.0
            total_positions = 0
            center_dist = abs(start_point[0] - center[0]) + abs(start_point[1] - center[1])
            # print(f"center_dist from {start_point}({reagent}) to {center}({module['id']}) is {center_dist}")
            for pos in positions:
                path, dist_to_final = self._get_manhattan_path(start_point, pos)
                # print(f"{module['id']} from {start_point} to {pos} ({reagent}) is {dist_to_final}")
                total_efficiency += center_dist - dist_to_final
                # print(f"{reagent} total_efficiency is {total_efficiency}")
                total_positions += 1
                if self._is_cell_surrounded(module, reagent, pos):
                    blockage_penalty += 1
            avg_efficiency += total_efficiency / max(total_positions, 1)
            # print(f"{reagent} avg_efficiency is {avg_efficiency}")
        return avg_efficiency * self.reward_storage_efficiency - blockage_penalty * self.reward_blockage_penalty

    def _is_cell_surrounded(self, module, reagent, cell):
        r, c = cell
        rows, cols = self.grid.shape
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        blocked_count = 0
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if self.grid[nr, nc] == (int(module["id"][2:]) if module["id"].startswith("op") else 0) and (
                nr, nc) not in module["reagent_positions"].get(reagent, []):
                    blocked_count += 1
        return blocked_count >= 3


if __name__ == "__main__":
    # 测试数据
    module_specs = {
        "op1": {"size": (2, 3), "duration": 4, "dependencies": ["r1", "r2"], "generate": "r9"},
        "op2": {"size": (2, 3), "duration": 4, "dependencies": ["r3", "r4"], "generate": "r10"},
        "op3": {"size": (2, 3), "duration": 4, "dependencies": ["r5", "r6"], "generate": "r11"},
        "op4": {"size": (2, 3), "duration": 4, "dependencies": ["r7", "r8"], "generate": "r12"},
        "op5": {"size": (2, 3), "duration": 2, "dependencies": ["op1", "op2"], "generate": "r13"},
        "op6": {"size": (3, 4), "duration": 2, "dependencies": ["op3", "op4"], "generate": "r14"},
        "op7": {"size": (4, 5), "duration": 1, "dependencies": ["op5", "op6"], "generate": "r15"}
    }
    reagent_specs = {
        "op1": {"r1": {"cells": 3, "from": "external"},
                "r2": {"cells": 3, "from": "external"}},
        "op2": {"r3": {"cells": 2, "from": "external"},
                "r4": {"cells": 4, "from": "external"}},
        "op3": {"r5": {"cells": 4, "from": "external"},
                "r6": {"cells": 2, "from": "external"}},
        "op4": {"r7": {"cells": 3, "from": "external"},
                "r8": {"cells": 3, "from": "external"}},
        "op5": {"r9": {"cells": 2, "from": "op1"},
                "r10": {"cells": 4, "from": "op2"}},
        "op6": {"r11": {"cells": 6, "from": "op3"},
                "r12": {"cells": 6, "from": "op4"}},
        "op7": {"r13": {"cells": 12, "from": "op5"},
                "r14": {"cells": 8, "from": "op6"}}
    }
    start_point = {
        "r1": (0, 0),
        "r2": (2, 0),
        "r3": (0, 4),
        "r4": (0, 6),
        "r5": (5, 0),
        "r6": (7, 0),
        "r7": (9, 0),
        "r8": (8, 0)
    }
    env = MultiAgentGridEnv(grid_size=(10, 10),
                            module_specs=module_specs,
                            reagent_specs=reagent_specs,
                            start_point=start_point)
    obs, _ = env.reset()
    # 随机采样动作，仅对 ready 状态模块有效
    actions = {}
    num_step = 1
    while num_step == 1 or all(value is False for key, value in dones.items()):
        print(f"num_step: {num_step}")
        for agent in env.agent_ids:
            actions[agent] = env.single_agent_action_space.sample()
        obs, rewards, dones, _, _ = env.step(actions)
        # print("Observations:", obs)
        print("Rewards:", rewards)
        print("Dones:", dones)
        num_step += 1
