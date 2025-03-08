import numpy as np
import gymnasium as gym
from gymnasium import spaces
from abc import ABC


def get_centroid(area):
    """计算区域（集合或列表）中所有单元的质心，返回 (row, col)"""
    arr = np.array(list(area))
    return np.mean(arr, axis=0)


def select_external_port(direction, candidate_type, grid, task):
    """
    根据动作方向和候选类型，从grid边界上选择一个空闲单元作为外部端口候选
    candidate_type: "in" 表示外部输入端候选，"out" 表示外部输出端候选
    选择时倾向于离相应区域（源区域或目标区域）质心较近
    如果找不到合适单元，返回 None
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


class GridRoutingEnv(gym.Env, ABC):
    """
    多任务布线环境（版本2）

    输入参数：
      - base_grid: np.array，二维网格状态，0: 空闲, 1: 设备占用（非当前任务允许区域）
      - tasks: 每个任务为字典，要求包含：
            "id": 任务编号
            "source_area": 集合或列表表示源区域单元
            "target_area": 集合或列表表示目标区域单元（可不连续，但应与 required_volume 对应）
            "required_volume": 整数，要求选取的液体单元数
      注意：外部输入/输出端口由智能体在路由过程中动态选取。

    每个任务内部包含三个路由段：
      seg0: 外部输入 -> 源区域（首次进入源区域时记录为 A，同时作为试剂的“尾部”）
      seg1: 从 A 延伸到目标区域（首次进入目标区域时记录为 B，同时记录试剂“头部”）
      seg2: 从 B 延伸至外部输出（必须到达网格边界）

    动作格式：(task_index, direction)
         task_index: 指明对哪个任务进行延伸
         direction: 0: 上, 1: 下, 2: 左, 3: 右
    """

    def __init__(self, base_grid, tasks, max_steps=500, reward_scale=1):
        super(GridRoutingEnv, self).__init__()
        self.base_grid = base_grid.copy()  # 原始网格
        self.rows, self.cols = self.base_grid.shape
        # 注意：此处不预先将任务区域标记到 base_grid 中
        self.tasks = tasks  # 每个任务包含 source_area, target_area, required_volume
        self.num_tasks = len(self.tasks)
        self.routing_penalty = -5
        self.invalid_cover = -100
        self.reward_cover = 30
        # 为每个任务添加置网格以及增加内部状态
        for task in self.tasks:
            for pos in task["source_area"]:
                self.base_grid[pos[0], pos[1]] = 1
            for pos in task["target_area"]:
                self.base_grid[pos[0], pos[1]] = 1
            # seg0: 外部输入 -> 源区域
            task["seg0"] = {"current_position": None, "route": [], "covered": []}
            # seg1: 源区域 (A) -> 目标区域
            task["seg1"] = {"current_position": None, "A": None, "route": [], "covered": []}
            # seg2: 目标区域 (B) -> 外部输出
            task["seg2"] = {"current_position": None, "B": None, "route": []}
            task["current_segment"] = 0  # 0: seg0, 1: seg1, 2: seg2
            task["completed"] = False
            task["has_cover"] = []

        self.action_space = spaces.Discrete(self.num_tasks * 4)
        self.observation_space = spaces.Box(low=0, high=10, shape=(self.rows, self.cols), dtype=np.int32)

        self.max_steps = max_steps
        self.current_step = 0
        self.grid = self.base_grid.copy()
        self.reward_scale = reward_scale  # 奖励归一化系数

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.grid = self.base_grid.copy()
        for task in self.tasks:
            task["seg0"] = {"current_position": None, "route": [], "covered": []}
            task["seg1"] = {"current_position": None, "A": None, "route": [], "covered": []}
            task["seg2"] = {"current_position": None, "B": None, "route": []}
            task["current_segment"] = 0
            task["completed"] = False
        return self.grid.copy(), {}

    def _in_allowed(self, task, pos, seg):
        """
        判断 pos 是否在当前任务 seg 允许通行的区域
        对于 seg0：允许空闲 (base_grid==0) 或 属于 source_area
        对于 seg1：允许空闲或属于 target_area（也允许在 source_area 内延伸，但希望尽快进入 target_area）
        对于 seg2：仅允许空闲
        此外，将其他任务的 source_area 与 target_area 视为障碍（除非该单元同时也在当前任务允许区域中）。
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
        # print(r, c, task["target_area"], pos in task
        #       ["target_area"], seg)
        if base_val == 1:
            if seg == 0 and pos in task["source_area"]:
                return True
            if seg == 1 and pos in task["target_area"]:
                return True
            return False
        return True

    def _has_legal_moves(self, task):
        """检查某个任务从当前状态是否至少存在一个合法动作"""
        seg = task["current_segment"]
        if seg == 0:
            allowed_seg = 0
            pos = task["seg0"]["current_position"]
        elif seg == 1:
            allowed_seg = 1
            pos = task["seg1"]["current_position"]
        elif seg == 2:
            allowed_seg = 2
            pos = task["seg2"]["current_position"]
        else:
            return False
        if pos is None:
            return True
        drc = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for delta in drc:
            new_pos = (pos[0] + delta[0], pos[1] + delta[1])
            if 0 <= new_pos[0] < self.rows and 0 <= new_pos[1] < self.cols:
                if self._in_allowed(task, new_pos, allowed_seg) and self.grid[new_pos[0], new_pos[1]] != 2:
                    return True
        return False

    def _check_deadlock(self):
        """如果所有任务都无合法动作，则认为死锁，返回 True"""
        for task in self.tasks:
            if not task["completed"] and self._has_legal_moves(task):
                return False
        return True

    def _all_done(self):
        return all(task["completed"] for task in self.tasks)

    def get_action_masks(self):
        """
        计算当前环境中每个任务的合法动作掩码，返回形状为 (num_tasks, 4) 的布尔数组。
        对于每个任务：
          - 如果任务已完成，则返回 [False, False, False, False]；
          - 如果当前任务处于外部端尚未选定阶段（seg0或seg2 且 current_position 为 None），返回全 True；
          - 否则，从任务当前所在 segment 的当前位置出发，对四个方向判断：
                1. 是否在网格内；
                2. 是否允许进入（调用 _in_allowed）；
                3. 是否未被占用（grid 值不为2）。
        额外要求：对于 seg0（source_area）和 seg1（target_area），如果当前位置已在相应区域内，
        则优先只允许那些能连续进入该区域的动作；如果至少有一个方向能连续进入，则仅允许这些方向，
        否则维持原来的合法动作判断。
        """
        masks = np.zeros(self.num_tasks * 4, dtype=bool)
        drc = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        task_n = 0
        for task in self.tasks:
            # 如果任务已经完成，则全 False
            base = task_n * 4
            if task["completed"]:
                masks[base:base + 4] = False
                continue

            current_seg = task["current_segment"]
            seg_key = f"seg{current_seg}"
            pos = task[seg_key]["current_position"]
            # 如果当前位置未确定，则允许所有动作
            if pos is None:
                masks[base:base + 4] = True
                continue

            # 先计算通用合法动作（不考虑连续性的限制）
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
            # 如果当前 segment 为 seg0 或 seg1，需要检查连续覆盖的限制
            if current_seg in [0, 1]:
                # 对 seg0，区域为 source_area；对 seg1，区域为 target_area
                region = task["source_area"] if current_seg == 0 else task["target_area"]
                is_over = True if current_seg == 1 or (current_seg == 0 and
                                                       len(task["source_area"]) < task["required_volume"]) else False
                # 如果当前位置已经在区域内，则检查四个方向中能否保持在区域内
                if pos in region and is_over:
                    continuous = []
                    for direction in range(4):
                        delta = drc[direction]
                        new_pos = (pos[0] + delta[0], pos[1] + delta[1])
                        # 新位置要在区域内，并且满足基本合法条件
                        if (0 <= new_pos[0] < self.rows and 0 <= new_pos[1] < self.cols and
                                self._in_allowed(task, new_pos, current_seg) and
                                self.grid[new_pos[0], new_pos[1]] != 2 and
                                new_pos in region):
                            continuous.append(True)
                        else:
                            continuous.append(False)
                    # 如果至少有一个方向能连续进入区域，则仅允许那些方向
                    if any(continuous):
                        branch_mask = continuous
                    else:
                        branch_mask = allowed
                else:
                    branch_mask = allowed
            else:
                masks.append(allowed)
            masks[base:base + 4] = branch_mask
            task_n += 1
        return masks

    def step(self, action):
        """
        执行动作：
          action: (task_index, direction)
        分段说明：
          seg0：外部输入 -> 源区域
                 在 seg0 阶段，agent 必须连续覆盖 source_area 内的单元，直到覆盖数量达到 required_volume。
                 如果 agent 在未达到要求时离开 source_area，则立即判定为失败并给予重罚终止 episode。
          seg1：从 seg0 的尾部开始延伸至目标区域
                 在 seg1 阶段，agent 必须连续在 target_area 内行走，直到将 target_area 内所有单元均覆盖。
                 如果 agent 在未全部覆盖 target_area 时离开目标区域，则立即终止该 episode。
          seg2：从 target_area 的末端延伸至网格边界，逻辑保持不变。
        返回值为 (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1
        task_idx = action // 4
        direction = action % 4

        reward = 0
        info = {}

        # 基础判断：任务编号、任务完成状态等
        if task_idx < 0 or task_idx >= self.num_tasks:
            return self.grid.copy(), self.routing_penalty * self.reward_scale, False, False, {"error": "无效任务编号"}
        task = self.tasks[task_idx]
        if task["completed"]:
            return self.grid.copy(), -1 * self.reward_scale, False, False, {"info": "任务已完成"}

        seg = task["current_segment"]
        if seg == 0:
            seg_state = task["seg0"]
            allowed_seg = 0
        elif seg == 1:
            seg_state = task["seg1"]
            allowed_seg = 1
        elif seg == 2:
            seg_state = task["seg2"]
            allowed_seg = 2
        else:
            return self.grid.copy(), self.routing_penalty * self.reward_scale, False, False, {"error": "未知路由段"}

        # 若外部端尚未确定，则使用原有逻辑选择候选点（不改变）
        if seg in [0, 2] and seg_state["current_position"] is None:
            candidate_type = "in" if seg == 0 else "out"
            pos = select_external_port(direction, candidate_type, self.grid, task)
            if pos is None:
                return self.grid.copy(), self.routing_penalty * self.reward_scale, False, False, {"error": "未找到合适外部端"}
            seg_state["current_position"] = pos
            seg_state["route"].append(pos)
            r, c = pos
            self.grid[r, c] = 2
            reward -= 1
            # seg0 初始时，如果候选点已经在 source_area，直接记录覆盖
            if seg == 0 and pos in task["source_area"]:
                if "covered" not in seg_state:
                    seg_state["covered"] = []
                seg_state["covered"].append(pos)
                reward += self.reward_cover
                if len(seg_state["covered"]) >= task["required_volume"]:
                    task["seg0"]["head"] = seg_state["covered"][0]
                    task["seg0"]["tail"] = pos
                    # seg0 完成，切换到 seg1
                    task["seg1"]["A"] = pos
                    task["seg1"]["current_position"] = pos
                    task["seg1"]["route"].append(pos)
                    task["current_segment"] = 1
                    reward += self.reward_cover
            if seg == 2 and (pos[0] in [0, self.rows - 1] or pos[1] in [0, self.cols - 1]):
                task["seg2"]["external_port"] = pos
                task["completed"] = True
                reward += 100
                vol = len({p for p in task["seg0"]["route"] if p in task["source_area"]})
                diff = abs(vol - task["required_volume"])
                reward -= diff * 5
            return self.grid.copy(), reward * self.reward_scale, self._all_done(), False, info

        # 已选定起点后，延伸当前段
        cur_pos = seg_state["current_position"]
        drc = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        if direction not in drc:
            return self.grid.copy(), self.routing_penalty * self.reward_scale, False, False, {"error": "无效方向"}
        delta = drc[direction]
        new_pos = (cur_pos[0] + delta[0], cur_pos[1] + delta[1])
        if not (0 <= new_pos[0] < self.rows and 0 <= new_pos[1] < self.cols):
            return self.grid.copy(), self.routing_penalty * self.reward_scale, False, False, {"error": "越界"}
        if not self._in_allowed(task, new_pos, allowed_seg):
            return self.grid.copy(), self.routing_penalty * self.reward_scale, False, False, {"error": "目标单元不允许或被占用"}
        if self.grid[new_pos[0], new_pos[1]] == 2:
            return self.grid.copy(), self.routing_penalty * self.reward_scale, False, False, {"error": "单元已被占用"}

        seg_state["current_position"] = new_pos
        seg_state["route"].append(new_pos)
        self.grid[new_pos[0], new_pos[1]] = 2
        reward -= 1

        # 分段逻辑处理
        if seg == 0:
            # seg0：要求连续覆盖 source_area
            if new_pos in task["source_area"]:
                # 检查进入区域和区域内时的覆盖情况
                if "covered" not in seg_state:
                    seg_state["covered"] = []
                # 如果是连续覆盖（即上一个位置也在 source_area）或第一次进入区域，则记录；否则认为连续性中断
                if cur_pos in task["source_area"] or len(seg_state["covered"]) == 0:
                    if new_pos not in seg_state["covered"]:
                        seg_state["covered"].append(new_pos)
                        reward += self.reward_cover
            else:
                # 检查离开区域时的覆盖情况
                if cur_pos in task["source_area"]:
                    if len(seg_state["covered"]) < task["required_volume"]:
                        # 连续性中断，直接终止并给予重罚
                        return self.grid.copy(), self.invalid_cover * self.reward_scale, True, False, {
                            "error": "seg0连续覆盖中断"}
            if len(seg_state["covered"]) >= task["required_volume"]:
                task["seg0"]["head"] = seg_state["covered"][0]
                task["seg0"]["tail"] = new_pos
                task["seg1"]["A"] = new_pos
                task["seg1"]["current_position"] = new_pos
                task["seg1"]["route"].append(new_pos)
                task["current_segment"] = 1
                reward += self.reward_cover
        elif seg == 1:
            # seg1：要求连续覆盖 target_area
            if new_pos in task["target_area"]:
                # 同seg0的情况
                if "covered" not in task["seg1"]:
                    seg_state["covered"] = []
                # 如果连续，则记录新位置
                if cur_pos in task["target_area"] and len(seg_state["covered"]) == 0:
                    seg_state["covered"].append(new_pos)
                    reward += self.reward_cover
            else:
                if cur_pos in task["target_area"]:
                    if seg_state["covered"] != task["target_area"]:
                        # 如果不连续，则认为连续覆盖中断
                        return self.grid.copy(), self.invalid_cover * self.reward_scale, True, False, {
                            "error": "seg1连续覆盖中断"}
            # 当目标区域全部被连续覆盖后，完成 seg1
            if seg_state["covered"] == task["target_area"]:
                task["current_segment"] = 2
        elif seg == 2:
            # seg2：与之前保持不变，延伸到网格边界完成任务
            if new_pos[0] in [0, self.rows - 1] or new_pos[1] in [0, self.cols - 1]:
                task["seg2"]["external_port"] = new_pos
                task["completed"] = True
                reward += 100
                vol = len({p for p in task["seg0"]["route"] if p in task["source_area"]})
                diff = abs(vol - task["required_volume"])
                reward -= diff * 5

        if self._check_deadlock():
            return self.grid.copy(), -100 * self.reward_scale, True, False, {"deadlock": True}
        else:
            done = self._all_done() or (self.current_step >= self.max_steps)
            return self.grid.copy(), reward * self.reward_scale, done, False, info

    def render(self, mode="human"):
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


if __name__ == "__main__":
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

    env = GridRoutingEnv(base_grid, tasks, max_steps=200)
    state, _ = env.reset()
    print("初始状态：")
    env.render()

    done = False
    while not done:
        act = env.action_space.sample()
        s, r, d, truncated, info = env.step(act)
        direction = act
        print_info = ""
        if direction == 0:
            print_info = "上"
        elif direction == 1:
            print_info = "下"
        elif direction == 2:
            print_info = "左"
        elif direction == 3:
            print_info = "右"
        mask_action = env.get_action_masks()
        print(f"mask_action: {mask_action}")
        print(f"动作: {act}：{print_info}, 奖励: {r}, done: {d}, info: {info}")
        env.render()
        if d:
            print("所有任务完成或达到最大步数。")
            break
