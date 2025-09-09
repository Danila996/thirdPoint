from itertools import permutations
import os
import random
from abc import ABC
from collections import deque
from itertools import permutations
from typing import Dict, List, Tuple, Set

import gymnasium as gym
import numpy as np
from gymnasium import spaces

os.environ['PYTHONHASHSEED'] = '0'
random.seed(0)
np.random.seed(0)

from assignCellWithILP import assign_reagents_fast_ilp_with_warmstart, brute_assign_fill_single_solution2


def normalize_inputs(
        module_cells: Set[Tuple[int, int]],
        reserved_overlap: Dict[str, Set[Tuple[int, int]]],
        start_pos: Dict[str, Tuple[int, int]]
):
    """
    Shift all coordinates so that module_cells' min row/col becomes (0,0).
    Returns normalized module_cells, reserved_overlap, start_pos, and offset.
    """
    # Compute offsets
    min_r = min(r for r, c in module_cells)
    min_c = min(c for r, c in module_cells)

    # Normalize module_cells
    norm_module = {(r - min_r, c - min_c) for r, c in module_cells}

    norm_overlap = {}
    for r in sorted(reserved_overlap):
        cells = reserved_overlap[r]
        # 对每个 cell 应用相同的平移逻辑
        norm_overlap[r] = {
            (r0 - min_r, c0 - min_c)
            for r0, c0 in cells
        }

    # Normalize start positions
    norm_start = {
        r: (start_pos[r][0] - min_r, start_pos[r][1] - min_c)
        for r in sorted(start_pos)
    }

    return norm_module, norm_overlap, norm_start, (min_r, min_c)


def denormalize_solutions(
        solutions: List[Dict[str, List[Tuple[int, int]]]],
        offset: Tuple[int, int]
):
    """
    Convert solutions from normalized coordinates back to original frame.
    """
    min_r, min_c = offset
    denorm = [
        {
            r: [(r0 + min_r, c0 + min_c) for (r0, c0) in sol[r]]
            for r in sorted(sol)
        }
        for sol in solutions
    ]
    return denorm


def preprocess_overlap(
        reserved_overlap: Dict[str, Set[Tuple[int, int]]],
        r_specs: Dict[str, int],
        start_pos: Dict[str, Tuple[int, int]],
        module_area
) -> Dict[str, Set[Tuple[int, int]]]:
    """
    对每个 reagent r，若 overlap 点超过需求 k，
    1) 修剪到最小矩形 remaining；
    2) 从四个角落分别生成 k 点的候选；
    3) 测试剩余自由区是否连通，取第一个通过的方案。
    """
    new_overlap: Dict[str, Set[Tuple[int, int]]] = {}

    for r in sorted(reserved_overlap):
        pts = set(reserved_overlap[r])
        k = r_specs.get(r, 0)
        # 不超额，直接保留
        if len(pts) <= k:
            new_overlap[r] = pts
            continue

        remaining = set(pts)
        xs = sorted({x for x, _ in remaining})
        ys = sorted({y for _, y in remaining})

        all_module_pts = module_area

        # “行列升序扫描”：
        best_sel = set()
        for x in sorted(xs):
            for y in sorted(ys):
                sel = best_sel.copy()
                if (x, y) in remaining:
                    sel.add((x, y))
                    free = all_module_pts - sel
                    if is_connected(list(free)):
                        best_sel.add((x, y))
                    else:
                        break
                    if len(best_sel) == k:
                        break
            if len(best_sel) == k:
                break
        new_overlap[r] = best_sel

    return new_overlap


def is_connected(cells: List[Tuple[int, int]]) -> bool:
    """
    判断给定的点集在 4-邻下是否连通。
    cells: [(r,c), ...]
    返回 True/False
    """
    if not cells:
        return True
    # 用集合加速查找
    cell_set = set(cells)
    # BFS 从第一个点开始
    start = cells[0]
    seen = {start}
    dq = deque([start])
    while dq:
        r, c = dq.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (r + dr, c + dc)
            if nb in cell_set and nb not in seen:
                seen.add(nb)
                dq.append(nb)
    return len(seen) == len(cell_set)


def check_reagent_connectivity(
        sol: Dict[str, List[Tuple[int, int]]]
) -> bool:
    """
    对解 sol 中的每个试剂，返回其是否连通的布尔值字典。
    sol: { reagent_name: [(r,c), ...], ... }
    """
    for r, pts in sol.items():
        if not is_connected(pts):
            return False
    return True


class MultiAgentGridEnv(gym.Env, ABC):
    def __init__(self, grid_size=(10, 10), module_specs=None, reagent_specs=None, start_point=None, level=None):
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
        self.global_grid = [
            [[] for _ in range(10)]
            for _ in range(10)
        ]
        # 奖励参数
        self.reward_placement = 1.5
        self.reward_invalid = -2.0
        self.proximity_reward_alpha = 0.1
        self.proximity_reward_beta = 0.05
        self.proximity_reward_beta_pair = 0.5
        self.reward_proximity = 2.0  # 非依赖模块之间的距离奖励基准
        self.reward_path_penalty = -0.05  # wiring 路径长度惩罚
        self.reward_storage_efficiency = 0.2
        self.reward_blockage_penalty = 0.2
        self.reward_conflict_penalty = -0.1
        self.reward_assign_penalty = -3.0

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
        self.level = level
        self.cur_level = 0  # 当前的level
        self.max_steps = 50  # 最大时间步数

        self._just_placed_modules = []
        # 用来去重 start_points 的缓存
        self._seen_startpoint_sets = set()
        self.module_cands = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros(self.grid_size, dtype=int)
        self.current_iter = 1
        self.start_time = 0
        self.error_time = 0
        self.active_modules = {}
        self.module_status = {agent: "not_started" for agent in self.agent_ids}
        self.global_grid = [
            [[] for _ in range(10)]
            for _ in range(10)
        ]
        # self.level = {agent: 0 for agent in self.agent_ids}
        self.cur_level = 0
        self._update_ready_modules()
        self._just_placed_modules.clear()
        observations = {agent: self._get_observation(agent) for agent in self.agent_ids}
        return observations, {}

    def _remove_expired_modules(self):
        expired = []
        is_remove = False
        for agent, module in self.active_modules.items():
            # print(f"{agent}, self.module_status[agent]:{self.module_status[agent]}")
            if self.module_status[agent] != "running":
                continue
            #  就当做阶段去用，每个module的duration都为1，采用level来表示阶段
            # print(f"self.current_iter:{self.current_iter}, self.error_time:{self.error_time}
            # , self.level[agent]:{self.level[agent]}")
            if self.current_iter - self.error_time > self.level[agent] + 1:
                row, col, height, width = module['position']
                for r in range(row, row + height):
                    for c in range(col, col + width):
                        self.grid[r, c] = 0
                expired.append(agent)
                # 记录中间试剂的起始位置
                if module["generate"] not in self.start_point:
                    row, col, height, width = module["position"]
                    center = (row + height / 2, col + width / 2)
                    self.start_point[module["generate"]] = center
                is_remove = True
        if is_remove:
            self.cur_level += 1
        for agent in expired:
            self.module_status[agent] = "finished"
            # del self.active_modules[agent]

    def _update_ready_modules(self):
        # 对于 "not_started" 状态的模块，检查依赖（外部依赖默认满足；对于 "op" 开头的依赖需 finished）
        for agent in self.agent_ids:
            if self.module_status[agent] != "not_started":
                continue
            # print(self.level[agent], self.cur_level)
            if self.level[agent] == self.cur_level:
                self.module_status[agent] = "ready"
            # deps = self.module_specs[agent].get("dependencies", [])
            # ready = True
            # for dep in deps:
            #     if dep.startswith("op"):
            #         if self.module_status.get(dep, "not_started") != "finished":
            #             ready = False
            #             break
            # if ready:
            #     self.module_status[agent] = "ready"

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
            # print(f"{agent}:{self.module_status[agent]}")
            if self.module_status[agent] != "ready":
                continue
            if agent not in actions:
                continue
            # 动作解析
            action = actions[agent]
            placement_val, reagent_mode = action
            flat_idx, orientation = divmod(placement_val, 2)
            row, col = divmod(flat_idx, self.grid_size[1])
            # print(f"row:{row}, col:{col}")
            module_spec = self.module_specs[agent]
            height, width = module_spec['size']
            if orientation == 1:
                height, width = width, height

            if row + height > self.grid_size[0] or col + width > self.grid_size[1]:
                # print("place invalid")
                rewards[agent] += self.reward_invalid
                error_time = True
                continue

            new_cells = {(r, c) for r in range(row, row + height) for c in range(col, col + width)}
            conflict = False
            for (r, c) in new_cells:
                if self.global_grid[r][c]:
                    for cur_agent in self.global_grid[r][c]:
                        module = self.active_modules[cur_agent]
                        cur_level = module["level"]
                        if cur_level == self.level[agent] - 1:
                            if not cur_agent in module_spec.get("dependencies", []):
                                # print("conflict with pre_agent")
                                conflict = True
                                break
                if conflict:
                    break
                if self.grid[r, c] != 0:
                    existing_numeric = self.grid[r, c]
                    allowed = False
                    for placed_agent, module in self.active_modules.items():
                        if int(placed_agent[2:]) == existing_numeric:
                            #  bug
                            if placed_agent in module_spec.get("dependencies", []):
                                allowed = True
                            break
                    if not allowed:
                        conflict = True
                        break
            if conflict:
                # print("place conflict")
                rewards[agent] += self.reward_invalid
                error_time = True
                continue

            dep_reward = self._dependency_ordering_reward(agent, row, col)
            rewards[agent] += dep_reward

            self._place_module(agent, row, col, orientation, reagent_mode)
            if agent not in self._just_placed_modules:
                self._just_placed_modules.append(agent)
            rewards[agent] += self.reward_placement
            # print(f"{agent} after reward_placement: {rewards[agent]}")
            rewards[agent] += self._calculate_proximity_reward(agent, row, col)
            # print(f"{agent} after _calculate_proximity_reward: {rewards[agent]}")

            # wiring_penalty = self._plan_reagent_paths(agent)
            # rewards[agent] += wiring_penalty

            # print(f"{agent} after wiring_penalty: {rewards[agent]}")
            # storage_reward = self._calculate_storage_efficiency(agent)
            # rewards[agent] += storage_reward
            # print(f"{agent} after storage_reward: {rewards[agent]}")
            # print(f"The position of {agent} is {row} , {col}, {orientation}")
            # print(f"agent:{agent}:{self.module_specs[agent]} statues is {self.module_status[agent]}")
            self.start_time += 1
        if error_time:
            self.error_time += 1

        # 4. 检查：如果当前 level 上所有模块都已放（即它们都已进入 running 或 finished）
        pending = [
            a for a, lvl in self.level.items()
            if lvl == self.cur_level and self.module_status[a] == "ready"
        ]
        if not pending and self._just_placed_modules:
            # 这一 level 放置阶段结束，做全局分配
            success, fail_assign = self._assign_reagents_for_current_level(rewards=rewards)
            if not success:
                print(f"[Level {self.cur_level}] 全局单元分配失败！")
                # modules = [self.active_modules[a] for a in self._just_placed_modules]
                for module_id in fail_assign:
                    agent = module_id
                    module = self.active_modules[agent]
                    row, col, height, width = module['position']
                    for r in range(row, row + height):
                        for c in range(col, col + width):
                            self.grid[r, c] = 0

                    self.module_status[agent] = "ready"
                    rewards[agent] += self.reward_assign_penalty
                self.error_time += 1
            else:
                # 清空临时列表，准备下一个 level
                self._just_placed_modules.clear()
        # print(f"error_time: {self.error_time}")
        # print(f"real_time = current_iter({self.current_iter}) - error_time({self.error_time}): "
        #       f"{self.current_iter - self.error_time}")

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
            self.global_grid[r][c].append(agent)
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
            "wiring_paths": {},
            "level": self.level[agent]
        }
        # # 加一段新的逻辑：处理父模块生成的试剂
        if module["id"] in self.module_specs:
            for reagent, spec in self.reagent_specs.get(module["id"], {}).items():
                src_agent = spec["from"]
                if src_agent.startswith("op") and src_agent in self.active_modules:
                    parent_module = self.active_modules[src_agent]
                    pr, pc, ph, pw = parent_module["position"]
                    parent_pos = set([(r, c) for r in range(pr, pr + ph) for c in range(pc, pc + pw)])
                    child_pos = set([(r, c) for r in range(row, row + height) for c in range(col, col + width)])
                    overlap = parent_pos & child_pos
                    if overlap:
                        module["reagent_positions"].setdefault(reagent, [])
                        module["reagent_positions"][reagent].extend(list(overlap))
        # module["reagent_positions"] = self._assign_reagent_distribution(module, reagent_mode)
        #
        # if module["reagent_positions"] is None:
        #     # 分配失败，可做回退处理
        #     print(f"Cannot assign reagent cells for {agent}")
        #     exit()
        #     return
        self.active_modules[agent] = module
        self.module_status[agent] = "running"

    def _assign_reagents_for_current_level(self,
                                           rewards: Dict = {},
                                           beam_width: int = 5,
                                           top_k: int = 10) -> (bool, list):
        """
        在当前 level 批次所有模块放置完成后，做全局单元分配：
        1. 为每个模块生成 top_k 个候选分配方案及其路由惩罚（route_cost）。
        2. 用 Beam Search 在所有模块的候选方案组合上搜索，累加 route_cost。
        3. 对每个 beam 加上全局废液分布惩罚，选出总 cost 最小的方案。
        4. 写回各 module["reagent_positions"]，返回 True；若无可行解，返回 False。
        """
        # 1) 收集本 level 刚放置的模块
        modules = [
            self.active_modules[a]
            for a in sorted(self._just_placed_modules) if a not in self.module_cands
        ]
        # print(f"modules:{modules}")
        if not modules:
            return True, []

        # 2) 为每个 module 生成 top_k 个候选
        # module_cands = {}
        for module in modules:

            cands = []
            row, col, h, w = module["position"]
            point_num = h * w
            max_tries = point_num * point_num - point_num  # 最多试这么多 variant，防止死循环
            module_cells = {
                (r, c)
                for r in range(row, row + h)
                for c in range(col, col + w)
            }

            reserved_overlap = module["reagent_positions"]
            r_specs = {}
            # print(f"pre_reserved_overlap:{reserved_overlap}")
            for r, info in self.reagent_specs[module['id']].items():
                r_specs[r] = info["cells"]
            reserved_overlap = preprocess_overlap(reserved_overlap, r_specs, self.start_point, module_cells)
            # print(f"after_reserved_overlap:{reserved_overlap}")

            # print(
            #     f"row, col, h, w: {row, col, h, w} ,len(module_cells):{len(module_cells)}"
            #     f", reserved_overlap：{reserved_overlap}, module_cells:{module_cells}, r_specs:{r_specs}"
            #     f", start_pos：{self.start_point} ")
            norm_cells, norm_overlap, norm_start, offset = normalize_inputs(
                module_cells, reserved_overlap, self.start_point
            )
            # seen = set()  # 用来记录已经见过的分配 key
            # brute_assign_fill/2
            # assign_reagents_fast_ilp
            warm = brute_assign_fill_single_solution2(
                norm_cells,
                norm_overlap,
                reagent_specs=r_specs, start_pos=norm_start,
            )[0]  # 取列表中的第一个解字典
            # print(f"warm:{warm}")
            if not warm:
                # print(f"reserved_overlap: {reserved_overlap}")
                continue
            # assign_reagents_fast_ilp
            # assign_reagents_fast_ilp_with_warmstart
            sols = assign_reagents_fast_ilp_with_warmstart(
                norm_cells,
                norm_overlap,
                reagent_specs=r_specs, start_pos=norm_start,
                brute_solution=warm,
                num_variants=3
            )

            # print(f"ilp_sols:{sols}")
            valid = []
            for sol in sols:
                if not check_reagent_connectivity(sol):
                    # 检查是否连通
                    valid = []
                    break
                if all(len(sol[r]) == r_specs[r] for r in r_specs):
                    valid.append(sol)

            if not valid:
                # print("使用暴力枚举获得一个解")
                sols = [warm]
                # sols = brute_assign_fill_single_solution(
                #     norm_cells,
                #     norm_overlap,
                #     reagent_specs=r_specs, start_pos=norm_start,
                # )
            else:
                sols = valid
            # print(f"sols:{sols}")
            sols = denormalize_solutions(sols, offset)
            # print(f"module:{module['id']}, sols:{sols}")

            for idx, sol in enumerate(sols, 1):
                # for r, spec in r_specs.items():
                #     if r not in sol:
                #         print(f"警告：sol {sol} 中缺少键 {r}")
                #         break
                #     if len(sol[r]) != spec:
                #         print(f"键 {r} 的长度不符：got={len(sol[r])}, expected={spec}")
                #         break
                if all(len(sol[r]) == r_specs[r] for r in r_specs):
                    route_cost = self._plan_reagent_paths_for_candidate(module, sol)
                    cands.append({'cells': sol, 'route_cost': route_cost})
            # for variant in range(max_tries):
            #     cells = self._generate_candidate_cells(module, variant)
            #     # print(f"cells:{cells}")
            #     if cells is None:
            #         continue
            #     for cell in cells:
            #         key = tuple(
            #             (r, tuple(sorted(cpos)))
            #             for r, cpos in sorted(cell.items())
            #         )
            #         if key in seen:
            #             continue
            #         seen.add(key)
            #         route_cost = self._plan_reagent_paths_for_candidate(module, cell)
            #         cands.append({'cells': cell, 'route_cost': route_cost})
            #     if len(cands) >= top_k:
            #         break
            # 按 route_cost 排序，取前 top_k
            self.module_cands[module['id']] = sorted(
                cands,
                key=lambda x: x['route_cost'],
                reverse=True  # ← 越大越前面
            )[:top_k]
            # self._seen_startpoint_sets 是否需要置空？
        fail_assign = []
        for module in modules:
            if module['id'] not in self.module_cands:
                fail_assign.append(module['id'])
        # print(f"fail_assign:{fail_assign}")
        if len(fail_assign) != 0:
            fail_assign.sort()
            return False, fail_assign
        # print(f"module_cands:{self.module_cands}")
        # 3) Beam Search 组合
        #    beam: tuple(assign_map: Dict[module_id, cells], cumulative route cost)

        beams = [({}, 0.0, {})]
        for module_id in self._just_placed_modules:
            module = self.active_modules[module_id]
            new_beams = []
            for assign_map, cum_cost, cost_map in beams:
                for cand in self.module_cands[module['id']]:
                    new_map = assign_map.copy()
                    new_map[module['id']] = cand['cells']
                    new_cost = cum_cost + cand['route_cost']
                    new_cost_map = cost_map.copy()
                    new_cost_map[module['id']] = cand['route_cost']
                    new_beams.append((new_map, new_cost, new_cost_map))
            # 保留最优 beam_width 条
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

        # print(f"beams:{beams}")
        # 4) 在 beams 上加全局废液惩罚，选最优
        best_assign, best_cost, best_cost_map = None, -float('inf'), None
        for assign_map, route_sum, cost_map in beams:
            # waste_pen = self._compute_waste_penalty(assign_map)
            waste_pen = 0
            total = route_sum + waste_pen
            if total > best_cost:
                best_cost, best_assign, best_cost_map = total, assign_map, cost_map

        if best_assign is None:
            return False, []
        # print(f"best_assign:{best_assign}")
        # 5) 写回 assignment
        for module_id in self._just_placed_modules:
            module = self.active_modules[module_id]
            module['reagent_positions'] = best_assign[module_id]
            # print(f"rewards[module_id]:{rewards[module_id]}")
            rewards[module_id] += best_cost_map[module_id]
        return True, []

    def _generate_candidate_cells(self, module: Dict, variant: int):
        """
        忽略旧 overlap，只用模块自身的“九点”作为每个 reagent 的起点，
        并调用 _assign_reagent_distribution 去做真正分配。
        """
        row, col, h, w = module["position"]
        mod_id = int(module["id"][2:])
        area = h * w
        top_k = min(max(10, area), 20)  # 下限 10，上限 20
        # print(f"top_k:{top_k}")
        # 1) 本模块占用的所有格子
        module_cells = {
            (r, c)
            for r in range(row, row + h)
            for c in range(col, col + w)
            if self.grid[r, c] == mod_id
        }
        if not module_cells:
            return None

        # 2) 模块四角、四边中点、中心的“理想”锚点
        r0, c0 = row, col
        r1, c1 = row + h - 1, col + w - 1
        raw_anchors = [
            (r0, c0), (r0, c1), (r1, c0), (r1, c1),
            (r0, (c0 + c1) // 2), (r1, (c0 + c1) // 2),
            ((r0 + r1) // 2, c0), ((r0 + r1) // 2, c1),
            ((r0 + r1) // 2, (c0 + c1) // 2),
        ]

        # 3) 把每个“理想”点投影到最近的 module_cells 上
        anchors = []
        for ar, ac in raw_anchors:
            best = min(module_cells,
                       key=lambda x: abs(x[0] - ar) + abs(x[1] - ac))
            if best not in anchors:
                anchors.append(best)

        # 4) variant 控制循环位移，让不同训练轮次用不同顺序
        reagents = list(self.reagent_specs[module["id"]].keys())
        L = len(anchors)
        N = len(reagents)
        perms = list(permutations(anchors, N))

        idx = variant % len(perms)
        choice = perms[idx]

        # shift = variant % L
        # anchors = (anchors * ((top_k//L)+1))[shift:shift+L]
        # print(f"anchors:{anchors}")
        # step = L // N
        # 5) 为每个 reagent 分配一个 anchor
        start_points = {
            r: [choice[i]]
            for i, r in enumerate(reagents)
        }
        # print(f"start_points:{start_points}")
        # 6) 调用统一的分配函数返回最终方案
        return self._assign_reagent_distribution(module, start_points, variant)

    def _plan_reagent_paths_for_candidate(self,
                                          module: Dict,
                                          candidate_cells: Dict[str, List[Tuple[int, int]]]
                                          ) -> float:
        """
        将 module['reagent_positions'] 临时设为 candidate_cells，
        调用已有的 _plan_reagent_paths 计算 wiring_penalty，
        并恢复原状后返回该 penalty。
        """
        # 备份
        orig_positions = module.get("reagent_positions", {}).copy()
        orig_paths = module.get("wiring_paths", {}).copy()

        # 插入 candidate
        module["reagent_positions"] = candidate_cells
        wiring_penalty = self._plan_reagent_paths(module["id"])
        storage_reward = self._calculate_storage_efficiency(module["id"])
        base_cost = wiring_penalty + storage_reward  # 原来的 cost（注意 wiring_penalty 本身是负的）

        # —— 新增：重叠惩罚/奖励 ——
        overlap_bonus = 0.0
        # 对本模块的每个 reagent
        for reagent, cells in candidate_cells.items():
            spec = self.reagent_specs[module["id"]][reagent]
            src = spec.get("from")
            # 只有 “from=opX” 且该父模块已在 active_modules 里，才算重叠
            if src and src.startswith("op") and src in self.active_modules:
                row, col, h, w = self.active_modules[src]["position"]
                mod_id = int(src[2:])
                # 收集本模块的所有格子
                parent_cells = {
                    (r, c)
                    for r in range(row, row + h)
                    for c in range(col, col + w)
                }
                # parent_cells = set(self.active_modules[src]["reagent_positions"].get(reagent, []))
                if parent_cells and len(set(cells) & parent_cells) > 0:
                    # 算出 candidate 与 parent 的重叠数
                    overlap_cnt = len(set(cells) & parent_cells)
                    total_need = spec["cells"]
                    penalty_overlap = total_need - overlap_cnt
                    if penalty_overlap <= 0:
                        penalty_overlap = 0
                    # 惩罚权重，可以调节
                    beta = 0.1
                    # 缺少重叠的单元越多，惩罚越大；完全重叠则零惩罚
                    overlap_bonus += beta * penalty_overlap
                    # （或者 if 你想要“缺少”的惩罚：
                    #    missing = total_need - overlap_cnt
                    #    overlap_bonus -= beta * missing
                    # ）
        # 结束新增

        # 恢复
        module["reagent_positions"] = orig_positions
        module["wiring_paths"] = orig_paths
        overlap_bonus = 0
        # 注意：wiring_penalty 本身是负的，storage_reward 正；我们
        # 把 overlap_bonus 当作「额外奖励」，即最终 cost = base_cost - overlap_bonus
        return base_cost - overlap_bonus

    def _compute_waste_penalty(self,
                               assign_map: Dict[str, Dict[str, List[Tuple[int, int]]]]
                               ) -> float:
        """
        对当前 level 全部模块的分配 assign_map 做废液分布评估：
        统计所有分配单元的邻居密度方差，方差越大惩罚越高。
        """
        all_cells = []
        for cells in assign_map.values():
            for pos_list in cells.values():
                all_cells.extend(pos_list)

        if not all_cells:
            return 0.0

        cell_set = set(all_cells)
        counts = []
        for (r, c) in all_cells:
            cnt = sum(
                1
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]
                if (r + dr, c + dc) in cell_set
            )
            counts.append(cnt)

        return float(np.var(counts)) * self.reward_blockage_penalty

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


import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def build_ops_data(active_modules):
    ops_data = []
    for op_id, module in active_modules.items():
        r, c, h, w = (int(x) for x in module['position'])
        level = int(module['level'])
        reagent_positions = {
            reagent: [(int(rr), int(cc)) for rr, cc in cells]
            for reagent, cells in module['reagent_positions'].items()
        }
        ops_data.append({
            'id': op_id,
            'position': (r, c, h, w),
            'reagent_positions': reagent_positions,
            'level': level
        })
    return ops_data


def plot_all_levels_with_start(ops_data, start_point, grid_size=(10, 10)):
    """一次性并排画出所有 level，并标记 start_point"""
    levels = sorted({op['level'] for op in ops_data})
    n = len(levels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharex=True, sharey=True)
    if n == 1: axes = [axes]
    for ax, lvl in zip(axes, levels):
        ax.set_title(f'Level {lvl}')
        ax.set_xlim(0, grid_size[1])
        ax.set_ylim(0, grid_size[0])
        ax.set_xticks(range(grid_size[1] + 1))
        ax.set_yticks(range(grid_size[0] + 1))
        ax.grid(True);
        ax.set_aspect('equal')
        # 模块 + 试剂
        for op in ops_data:
            if op['level'] != lvl: continue
            r, c, h, w = op['position']
            ax.add_patch(Rectangle((c, r), w, h, fill=False, linewidth=2))
            ax.text(c + w / 2, r + h / 2, op['id'], ha='center', va='center')
            for reagent, cells in op['reagent_positions'].items():
                xs = [cc + 0.5 for rr, cc in cells]
                ys = [rr + 0.5 for rr, cc in cells]
                ax.scatter(xs, ys, s=50)
                for x, y in zip(xs, ys):
                    ax.text(x, y, reagent, ha='center', va='center', fontsize=6)
        # 起始点
        for reagent, (rr, cc) in start_point.items():
            x, y = cc + 0.5, rr + 0.5
            ax.scatter([x], [y], marker='*', s=100)
            ax.text(x + 0.1, y + 0.1, reagent, ha='left', va='bottom', fontsize=6)
        ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


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
    level = {
        "op1": 0,
        "op2": 0,
        "op3": 0,
        "op4": 0,
        "op5": 1,
        "op6": 1,
        "op7": 2,
    }
    env = MultiAgentGridEnv(grid_size=(10, 10),
                            module_specs=module_specs,
                            reagent_specs=reagent_specs,
                            start_point=start_point,
                            level=level)
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
    for v in env.active_modules.items():
        print(v)
    # ops_data = []
    # for key, module in env.active_modules.items():
    #     # 模块基本信息
    #     op_id = module['id']
    #     # 把 numpy.int64 转成 Python int
    #     r, c, h, w = (int(x) for x in module['position'])
    #     level = int(module['level'])
    #     # 转换试剂位置列表
    #     reagent_positions = {}
    #     for reagent, cells in module['reagent_positions'].items():
    #         # 每个 cell 也转换成 (int, int)
    #         reagent_positions[reagent] = [(int(row), int(col)) for row, col in cells]
    #     # 组装字典
    #     ops_data.append({
    #         'id': op_id,
    #         'position': (r, c, h, w),
    #         'reagent_positions': reagent_positions,
    #         'level': level
    #     })
    ops_data = build_ops_data(env.active_modules)
    # 2. 调用绘图
    plot_all_levels_with_start(ops_data, env.start_point)
