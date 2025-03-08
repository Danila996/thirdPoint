import numpy as np
import gymnasium as gym
from gymnasium import spaces
from abc import ABC
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers


def get_centroid(area):
    """计算区域（集合或列表）中所有单元的质心，返回 (row, col)"""
    arr = np.array(list(area))
    return np.mean(arr, axis=0)


def select_external_port(direction, candidate_type, grid, task):
    """
    根据动作方向和候选类型，从 grid 边界上选择一个空闲单元作为外部端口候选。
    candidate_type: "in" 表示外部输入端候选，"out" 表示外部输出端候选。
    选择时倾向于离相应区域（源区域或目标区域）质心较近；找不到则返回 None。
    """
    rows, cols = grid.shape
    centroid = get_centroid(task["source_area"]) if candidate_type == "in" else get_centroid(task["target_area"])
    candidates = []
    if direction == 0:  # 上边界
        r = 0
        for c in range(cols):
            if grid[r, c] == 0:
                candidates.append((r, c))
    elif direction == 1:  # 下边界
        r = rows - 1
        for c in range(cols):
            if grid[r, c] == 0:
                candidates.append((r, c))
    elif direction == 2:  # 左边界
        c = 0
        for r in range(rows):
            if grid[r, c] == 0:
                candidates.append((r, c))
    elif direction == 3:  # 右边界
        c = cols - 1
        for r in range(rows):
            if grid[r, c] == 0:
                candidates.append((r, c))
    else:
        return None

    if not candidates:
        return None

    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates - centroid, axis=1)
    idx = np.argmin(dists)
    return tuple(candidates[idx])


class GridRoutingMultiAgentEnv(ParallelEnv, ABC):
    """
    多任务布线环境（版本2）的多智能体版本。
    每个任务对应一个智能体，动作空间为 Discrete(4)（上/下/左/右），
    观测为字典，包含共享的 grid 状态以及该智能体的动作掩码。

    任务内部仍分为三个路由段：
      seg0：外部输入 -> 源区域（要求连续覆盖 source_area 至少达到 required_volume，否则判定失败）
      seg1：从 seg0 的尾部延伸至目标区域（要求连续覆盖 target_area，否则判定失败）
      seg2：从目标区域延伸至网格边界
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, base_grid, tasks, max_steps=500, reward_scale=1):
        # 多智能体：每个任务为一个智能体，名称为 "agent_0", "agent_1", …
        self.num_tasks = len(tasks)
        self.agents = ["agent_" + str(i) for i in range(self.num_tasks)]
        self.possible_agents = self.agents.copy()

        self.base_grid = base_grid.copy()  # 初始网格
        self.rows, self.cols = self.base_grid.shape
        self.tasks = tasks  # 每个任务必须包含 "id", "source_area", "target_area", "required_volume"
        self.routing_penalty = -5
        self.invalid_cover = -100
        self.reward_cover = 30
        # 为每个任务在 base_grid 上标记区域，并添加内部状态
        for task in self.tasks:
            for pos in task["source_area"]:
                self.base_grid[pos[0], pos[1]] = 1
            for pos in task["target_area"]:
                self.base_grid[pos[0], pos[1]] = 1
            # seg0：外部输入 -> 源区域，要求连续覆盖 source_area
            task["seg0"] = {"current_position": None, "route": [], "covered": []}
            # seg1：从 seg0 尾部延伸至目标区域，要求连续覆盖 target_area
            task["seg1"] = {"current_position": None, "A": None, "route": [], "covered": []}
            # seg2：从目标区域延伸至网格边界
            task["seg2"] = {"current_position": None, "B": None, "route": []}
            task["current_segment"] = 0  # 0: seg0, 1: seg1, 2: seg2
            task["completed"] = False

        # 整个环境动作空间：各智能体均为 Discrete(4)
        self.action_spaces = {agent: spaces.Discrete(4) for agent in self.agents}
        # 观测：此处采用字典形式，包括共享 grid 和智能体对应的动作掩码
        self.observation_spaces = {
            agent: spaces.Dict({
                "grid": spaces.Box(low=0, high=10, shape=(self.rows, self.cols), dtype=np.int32),
                "action_mask": spaces.Box(low=0, high=1, shape=(4,), dtype=bool)
            })
            for agent in self.agents
        }

        self.max_steps = max_steps
        self.current_step = 0
        self.grid = self.base_grid.copy()
        self.reward_scale = reward_scale

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.grid = self.base_grid.copy()
        for task in self.tasks:
            task["seg0"] = {"current_position": None, "route": [], "covered": []}
            task["seg1"] = {"current_position": None, "A": None, "route": [], "covered": []}
            task["seg2"] = {"current_position": None, "B": None, "route": []}
            task["current_segment"] = 0
            task["completed"] = False

        observations = {}
        masks = self.get_action_masks()
        for i, agent in enumerate(self.agents):
            observations[agent] = {"grid": self.grid.copy(), "action_mask": np.array(masks[i], dtype=bool)}
        return observations

    def get_action_masks(self):
        """返回一个列表，每个元素为对应任务（智能体）的 4 维布尔动作掩码"""
        masks = []
        drc = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        for task in self.tasks:
            # 如果任务已完成，则所有动作均非法
            if task["completed"]:
                masks.append([False, False, False, False])
                continue

            current_seg = task["current_segment"]
            seg_key = f"seg{current_seg}"
            pos = task[seg_key]["current_position"]
            # 未确定当前位置时允许所有方向
            if pos is None:
                masks.append([True, True, True, True])
                continue

            allowed = []
            for direction in range(4):
                delta = drc[direction]
                new_pos = (pos[0] + delta[0], pos[1] + delta[1])
                if not (0 <= new_pos[0] < self.rows and 0 <= new_pos[1] < self.cols):
                    allowed.append(False)
                    continue
                if not self._in_allowed(task, new_pos, current_seg):
                    allowed.append(False)
                    continue
                if self.grid[new_pos[0], new_pos[1]] == 2:
                    allowed.append(False)
                    continue
                allowed.append(True)
            # 对 seg0 与 seg1：若当前位置已在对应区域内，则尝试仅允许能连续留在该区域内的方向
            if current_seg in [0, 1]:
                region = task["source_area"] if current_seg == 0 else task["target_area"]
                if pos in region:
                    continuous = []
                    for direction in range(4):
                        delta = drc[direction]
                        new_pos = (pos[0] + delta[0], pos[1] + delta[1])
                        if (0 <= new_pos[0] < self.rows and 0 <= new_pos[1] < self.cols and
                                self._in_allowed(task, new_pos, current_seg) and
                                self.grid[new_pos[0], new_pos[1]] != 2 and
                                new_pos in region):
                            continuous.append(True)
                        else:
                            continuous.append(False)
                    if any(continuous):
                        masks.append(continuous)
                    else:
                        masks.append(allowed)
                else:
                    masks.append(allowed)
            else:
                masks.append(allowed)
        return masks

    def _in_allowed(self, task, pos, seg):
        """
        判断 pos 是否在当前任务 seg 允许通行的区域：
          - 对于 seg0：允许空闲（base_grid==0）或在 source_area 中；
          - 对于 seg1：允许空闲或在 target_area 中；
          - 对于 seg2：仅允许空闲。
        同时，其他任务的 source_area 和 target_area（若不属于当前任务允许区域）视为障碍。
        """
        r, c = pos
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False
        if self.grid[r, c] == 2:
            return False
        for other_task in self.tasks:
            if other_task["id"] != task["id"]:
                if pos in other_task.get("source_area", set()) or pos in other_task.get("target_area", set()):
                    if pos not in task.get("source_area", set()) and pos not in task.get("target_area", set()):
                        return False
        base_val = self.base_grid[r, c]
        if base_val == 1:
            if seg == 0 and pos in task["source_area"]:
                return True
            if seg == 1 and pos in task["target_area"]:
                return True
            return False
        return True

    def _has_legal_moves(self, task):
        """判断当前任务是否至少有一个合法动作"""
        seg = task["current_segment"]
        seg_key = f"seg{seg}"
        pos = task[seg_key]["current_position"]
        if pos is None:
            return True
        drc = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for delta in drc:
            new_pos = (pos[0] + delta[0], pos[1] + delta[1])
            if 0 <= new_pos[0] < self.rows and 0 <= new_pos[1] < self.cols:
                if self._in_allowed(task, new_pos, seg) and self.grid[new_pos[0], new_pos[1]] != 2:
                    return True
        return False

    def _check_deadlock(self):
        """如果所有任务都无合法动作，则认为发生死锁"""
        for task in self.tasks:
            if not task["completed"] and self._has_legal_moves(task):
                return False
        return True

    def _all_done(self):
        return all(task["completed"] for task in self.tasks)

    def step(self, actions):
        """
        参数 actions 是一个字典，键为 agent 名称，值为动作（0,1,2,3 表示方向）。
        顺序处理各智能体动作，更新共享 grid 与各任务状态。
        返回：(observations, rewards, dones, infos)
        """
        rewards = {agent: 0 for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # 依次处理每个智能体（任务）
        for i, agent in enumerate(self.agents):
            # 若该任务已完成，则跳过动作处理
            task = self.tasks[i]
            if task["completed"]:
                continue
            direction = actions[agent]
            seg = task["current_segment"]
            seg_key = f"seg{seg}"
            seg_state = task[seg_key]

            # 外部端选择阶段（seg0 或 seg2）且尚未确定起点时
            if seg in [0, 2] and seg_state["current_position"] is None:
                candidate_type = "in" if seg == 0 else "out"
                pos = select_external_port(direction, candidate_type, self.grid, task)
                if pos is None:
                    infos[agent] = {"error": "未找到合适外部端"}
                    rewards[agent] += self.routing_penalty * self.reward_scale
                    continue
                seg_state["current_position"] = pos
                seg_state["route"].append(pos)
                r, c = pos
                self.grid[r, c] = 2
                rewards[agent] -= 1
                # seg0：若候选点已在 source_area 内，则记录覆盖并给予奖励
                if seg == 0 and pos in task["source_area"]:
                    if pos not in seg_state.get("covered", []):
                        seg_state.setdefault("covered", []).append(pos)
                        rewards[agent] += self.reward_cover
                    if len(seg_state["covered"]) >= task["required_volume"]:
                        # seg0 完成，切换至 seg1
                        task["seg0"]["head"] = seg_state["covered"][0]
                        task["seg0"]["tail"] = pos
                        task["seg1"]["A"] = pos
                        task["seg1"]["current_position"] = pos
                        task["seg1"]["route"].append(pos)
                        task["current_segment"] = 1
                        rewards[agent] += self.reward_cover
                # seg2：若候选点已在网格边界，则任务完成
                if seg == 2 and (pos[0] in [0, self.rows - 1] or pos[1] in [0, self.cols - 1]):
                    task["seg2"]["external_port"] = pos
                    task["completed"] = True
                    rewards[agent] += 100
                    vol = len({p for p in task["seg0"]["route"] if p in task["source_area"]})
                    diff = abs(vol - task["required_volume"])
                    rewards[agent] -= diff * 5
                continue  # 外部端选择后，本 step 不再延伸

            # 已有起点后，进行延伸
            cur_pos = seg_state["current_position"]
            drc = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
            if direction not in drc:
                infos[agent] = {"error": "无效方向"}
                rewards[agent] += self.routing_penalty * self.reward_scale
                continue
            delta = drc[direction]
            new_pos = (cur_pos[0] + delta[0], cur_pos[1] + delta[1])
            if not (0 <= new_pos[0] < self.rows and 0 <= new_pos[1] < self.cols):
                infos[agent] = {"error": "越界"}
                rewards[agent] += self.routing_penalty * self.reward_scale
                continue
            if not self._in_allowed(task, new_pos, seg):
                infos[agent] = {"error": "目标单元不允许或被占用"}
                rewards[agent] += self.routing_penalty * self.reward_scale
                continue
            if self.grid[new_pos[0], new_pos[1]] == 2:
                infos[agent] = {"error": "单元已被占用"}
                rewards[agent] += self.routing_penalty * self.reward_scale
                continue

            # 更新当前段状态
            seg_state["current_position"] = new_pos
            seg_state["route"].append(new_pos)
            self.grid[new_pos[0], new_pos[1]] = 2
            rewards[agent] -= 1

            # 针对不同段分别处理
            if seg == 0:
                # seg0：要求连续覆盖 source_area
                if new_pos in task["source_area"]:
                    if new_pos not in seg_state.get("covered", []):
                        seg_state.setdefault("covered", []).append(new_pos)
                        rewards[agent] += self.reward_cover
                else:
                    # 如果当前位置在 source_area 内，离开后且覆盖数不足，判定连续性中断
                    if cur_pos in task["source_area"]:
                        if len(seg_state.get("covered", [])) < task["required_volume"]:
                            return (self.grid.copy(),
                                    self.invalid_cover * self.reward_scale,
                                    {agent: True for agent in self.agents},
                                    {agent: {"error": "seg0连续覆盖中断"} for agent in self.agents})
                if len(seg_state.get("covered", [])) >= task["required_volume"]:
                    task["seg0"]["head"] = seg_state["covered"][0]
                    task["seg0"]["tail"] = new_pos
                    task["seg1"]["A"] = new_pos
                    task["seg1"]["current_position"] = new_pos
                    task["seg1"]["route"].append(new_pos)
                    task["current_segment"] = 1
                    rewards[agent] += self.reward_cover
            elif seg == 1:
                # seg1：要求连续覆盖 target_area
                if new_pos in task["target_area"]:
                    if new_pos not in seg_state.get("covered", []):
                        seg_state.setdefault("covered", []).append(new_pos)
                        rewards[agent] += self.reward_cover
                else:
                    if cur_pos in task["target_area"]:
                        # 若未能覆盖 target_area（连续性中断）则直接终止
                        return (self.grid.copy(),
                                self.invalid_cover * self.reward_scale,
                                {agent: True for agent in self.agents},
                                {agent: {"error": "seg1连续覆盖中断"} for agent in self.agents})
                # 当 target_area 全部被连续覆盖后，完成 seg1
                if set(seg_state.get("covered", [])) == set(task["target_area"]):
                    task["current_segment"] = 2
            elif seg == 2:
                # seg2：延伸至网格边界后完成任务
                if new_pos[0] in [0, self.rows - 1] or new_pos[1] in [0, self.cols - 1]:
                    task["seg2"]["external_port"] = new_pos
                    task["completed"] = True
                    rewards[agent] += 100
                    vol = len({p for p in task["seg0"]["route"] if p in task["source_area"]})
                    diff = abs(vol - task["required_volume"])
                    rewards[agent] -= diff * 5

        self.current_step += 1
        # 计算各智能体结束标志及整体结束标志
        dones = {agent: self.tasks[i]["completed"] for i, agent in enumerate(self.agents)}
        dones["__all__"] = (self._all_done() or
                            (self.current_step >= self.max_steps) or
                            self._check_deadlock())

        # 构建新的观测（附带动作掩码）
        masks = self.get_action_masks()
        observations = {}
        for i, agent in enumerate(self.agents):
            observations[agent] = {"grid": self.grid.copy(), "action_mask": np.array(masks[i], dtype=bool)}

        return observations, rewards, dones, infos

    def render(self, mode="human"):
        """简单打印 grid 状态；S/T 分别表示 source/target 区域，A 表示当前所在点"""
        symbol_map = {0: ".", 1: "#", 2: "*"}
        grid_disp = np.full((self.rows, self.cols), "", dtype=object)
        for r in range(self.rows):
            for c in range(self.cols):
                val = self.grid[r, c]
                grid_disp[r, c] = symbol_map.get(val, str(val))
        for task in self.tasks:
            for pos in task["source_area"]:
                r, c = pos
                if self.base_grid[r, c] == 1 and self.grid[r, c] == 1:
                    grid_disp[r, c] = "S"
            for pos in task["target_area"]:
                r, c = pos
                if self.base_grid[r, c] == 1 and self.grid[r, c] == 1:
                    grid_disp[r, c] = "T"
            seg = task["current_segment"]
            pos = task[f"seg{seg}"]["current_position"]
            if pos is not None:
                grid_disp[pos[0], pos[1]] = "A"
        for r in range(self.rows):
            print(" ".join(grid_disp[r]))

    def close(self):
        pass


def env():
    """
    环境构造函数，返回经过 PettingZoo 封装的多智能体环境示例。
    此示例中定义了两个任务，分别对应两个智能体。
    """
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
    env_instance = GridRoutingMultiAgentEnv(base_grid, tasks, max_steps=200)
    # 可根据需要添加 PettingZoo 常用包装器
    env_instance = wrappers.OrderEnforcingWrapper(env_instance)
    return env_instance


if __name__ == "__main__":
    # 测试多智能体环境
    env_instance = env()
    observations = env_instance.reset()
    print("初始状态:")
    for agent, obs in observations.items():
        print(f"{agent} 的初始观测：")
        print(obs["grid"])
        print("动作掩码：", obs["action_mask"])
    done = False
    while True:
        actions = {}
        # 此处每个智能体随机采样动作
        for agent in env_instance.agents:
            actions[agent] = env_instance.action_spaces[agent].sample()
        observations, rewards, dones, infos = env_instance.step(actions)
        for agent in env_instance.agents:
            print(f"{agent} 动作: {actions[agent]}, 奖励: {rewards[agent]}, done: {dones[agent]}, info: {infos[agent]}")
        env_instance.render()
        if dones["__all__"]:
            print("所有任务完成或达到最大步数。")
            break
