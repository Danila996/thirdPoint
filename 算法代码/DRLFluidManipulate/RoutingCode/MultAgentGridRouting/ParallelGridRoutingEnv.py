import functools
import random
from abc import ABC
import numpy as np

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers, agent_selector
from gymnasium import spaces
from pettingzoo.utils.conversions import parallel_wrapper_fn
# util function
from generate_random_task import generate_random_grid, generate_random_task
from routing_util import select_external_port, _can_cover_all


def env(**kwargs):
    env = raw_env(**kwargs)
    # env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = ParallelGridRoutingEnv(render_mode=render_mode)
    return env


class ParallelGridRoutingEnv(AECEnv, ABC):
    metadata = {"render.modes": ["human"], "name": "parallel_grid_routing_v0"}

    def __init__(self, max_steps=500, reward_scale=0.1, max_tasks_num=3, render_mode="human"):
        AECEnv.__init__(self)
        # 创建原始单环境
        self._agent_selector = None
        self.observations = None
        self.state = None
        self.truncations = None
        self.terminations = None
        # def tasks and grid
        self.base_grid = None
        self.rows = 0
        self.cols = 0
        self.max_tasks_num = max_tasks_num
        self.tasks = []
        self.grid = None

        # def agent:
        self.possible_agents = ["agent_" + str(i) for i in range(self.max_tasks_num)]
        self.agents = None
        self.active_agents = None
        self.inactive_agents = None
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self.max_steps = max_steps
        self.current_step = 0
        self.render_mode = render_mode

        # rewards param
        self._cumulative_rewards = None
        self.reward_scale = reward_scale  # 奖励归一化系数
        self.num_tasks = 1
        self.routing_penalty = -5
        self.invalid_cover = -30
        self.reward_cover = 10
        self.reward_completed = 50

    def action_space(self, agent):
        return spaces.Discrete(4)

    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return spaces.Dict({
            "grid": spaces.Box(low=0, high=1, shape=(self.rows, self.cols), dtype=float),
            "agent_positions": spaces.Box(low=0, high=1, shape=(self.rows, self.cols), dtype=float),
            "action_mask": spaces.Box(low=0, high=1, shape=(4,), dtype=bool),
        })

    def observe(self, agent):
        self.observations = self._get_observation()
        # print(self.observations)
        # assert self.observations[agent], print(self.observations, agent)
        # print(agent, self.observations[agent])
        return self.observations[agent]

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        # 调用原环境的 reset
        self.rows = 10
        self.cols = 10
        self.base_grid = generate_random_grid((self.rows, self.cols))
        self.grid = self.base_grid
        self.rows, self.cols = self.base_grid.shape
        # 任务类型
        # 随机生成任务数量：例如最大任务数为 7，则任务数为1~7
        task_type = random.randint(0, 1)
        task_types = ["double", "single"]
        max_tasks = random.randint(1, self.max_tasks_num)
        occupied = set()
        self.tasks = [generate_random_task(i, (self.rows, self.cols), occupied) for i in range(max_tasks)]
        self.num_tasks = len(self.tasks)
        cur_task = 1
        for task in self.tasks:
            task["task_type"] = task_types[task_type]
            if task["task_type"] == "double":
                for pos in task["source_area"]:
                    self.base_grid[pos[0], pos[1]] = cur_task * 0.1 + 0.01
            for pos in task["target_area"]:
                self.base_grid[pos[0], pos[1]] = cur_task * 0.1 + 0.02
            task["seg0"] = {"current_position": None, "route": [], "covered": []}
            task["seg1"] = {"current_position": None, "A": None, "route": [], "covered": []}
            task["seg2"] = {"current_position": None, "B": None, "route": []}
            in_pos = select_external_port("in", self.base_grid.copy(), task)
            # out_pos = select_external_port("out", self.base_grid.copy(), task)
            if task["task_type"] == "double":
                task["current_segment"] = 0
                task["seg0"]["current_position"] = in_pos
                self.grid[in_pos[0], in_pos[1]] = 0.01
                if task["current_segment"] == 0 and in_pos in task["source_area"]:
                    if "covered" not in task["seg0"]:
                        task["seg0"]["covered"] = []
                    task["seg0"]["covered"].append(in_pos)
                    if len(task["seg0"]["covered"]) >= task["required_volume"]:
                        task["seg0"]["head"] = task["seg0"]["covered"][0]
                        task["seg0"]["tail"] = in_pos
                        # seg0 完成，切换到 seg1
                        task["seg1"]["A"] = in_pos
                        task["seg1"]["current_position"] = in_pos
                        task["seg1"]["route"].append(in_pos)
                        task["current_segment"] = 1
            else:
                task["current_segment"] = 1
                task["seg1"]["current_position"] = in_pos
                self.grid[in_pos[0], in_pos[1]] = 0.01
                if task["current_segment"] == 1 and in_pos in task["target_area"]:
                    if "covered" not in task["seg1"]:
                        task["seg1"]["covered"] = []
                    task["seg1"]["covered"].append(in_pos)
                    if len(task["seg1"]["covered"]) >= task["required_volume"]:
                        task["seg1"]["head"] = task["seg0"]["covered"][0]
                        task["seg1"]["tail"] = in_pos
                        # seg0 完成，切换到 seg1
                        task["seg2"]["B"] = in_pos
                        task["seg2"]["current_position"] = in_pos
                        task["seg2"]["route"].append(in_pos)
                        task["current_segment"] = 2
            task["completed"] = False
            task["cur_task"] = cur_task
            cur_task += 1

        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
        # self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.rewards = {a: 0.0 for a in self.possible_agents}
        self.infos = {a: {} for a in self.possible_agents}
        # self.observation_space = {agent: {
        #     "grid": spaces.Box(low=0, high=1, shape=(self.rows, self.cols), dtype=float),
        #     "agent_positions": spaces.Box(low=0, high=1, shape=(self.rows, self.cols), dtype=float),
        #     "action_mask": spaces.Box(low=0, high=1, shape=(4,), dtype=bool),
        # } for agent in self.possible_agents}
        # self.action_spaces = {agent: spaces.Discrete(4) for agent in self.possible_agents}
        self.agents = ["agent_" + str(task["id"]) for task in self.tasks]
        self.state = {a: None for a in self.possible_agents}
        self.observations = {a: None for a in self.possible_agents}
        self.active_agents = ["agent_" + str(task["id"]) for task in self.tasks]
        self.inactive_agents = [a for a in self.possible_agents if a not in self.active_agents]
        self.terminations = {a: False if a in self.active_agents else True for a in self.possible_agents}
        self.truncations = {a: False for a in self.possible_agents}
        self.current_step = 0

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

    def step(self, action):
        """
        actions: 字典，key 为 agent id，value 为动作（0,1,2,3）
        这里我们对每个 agent 依次执行动作（注意：由于原环境的 step() 内部是顺序执行的，
        多 agent 的动作会依次更新共享 grid 状态，这可能存在动作冲突问题）
        """
        if (
                self.truncations[self.agent_selection]
                or self.terminations[self.agent_selection]
        ):
            pre_agent_selection = self.agent_selection
            # print(f"pre_agent_selection:{self.agent_selection}", self.agents)
            pre_index = self.agents.index(pre_agent_selection)
            self._was_dead_step(action)
            # print(f"self.agent_selection:{self.agent_selection}", self.agents)
            if self.agent_selection == pre_agent_selection:
                if len(self.agents) != 0:
                    if pre_index >= len(self.agents):
                        self.agent_selection = self.agents[0]
                    else:
                        self.agent_selection = self.agents[pre_index]
                        # self.agent_selection = self._agent_selector.next()
                else:
                    self.agent_selection = None
            # print(f"self.agent_selection:{self.agent_selection}", self.agents)
            return

        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0
        # stores action of current agent
        self.state[self.agent_selection] = action

        if self._agent_selector.is_last():
            # 为每个 agent，构造组合动作：
            # 内部逻辑存在问题，后面看情况修改
            for agent in self.agents:
                task_idx = self.agent_name_mapping[agent]

                task = self.tasks[task_idx]
                if not self._has_legal_moves(task):
                    self.terminations[agent] = True
                    self.rewards[agent] = self.invalid_cover * self.reward_scale
                    continue
                direction = self.state[agent]
                self.rewards[agent] = 0
                info = {}
                target_area = {}

                seg = task["current_segment"]
                if seg == 0:
                    seg_state = task["seg0"]
                    target_area = task["source_area"]
                    allowed_seg = 0
                elif seg == 1:
                    seg_state = task["seg1"]
                    target_area = task["target_area"]
                    allowed_seg = 1
                elif seg == 2:
                    seg_state = task["seg2"]
                    allowed_seg = 2

                # 若外部端尚未确定，则选择候选点（后续端口这边肯定还是要让他自己选择，避免布线冲突）
                if seg in [0, 2] and seg_state["current_position"] is None:
                    candidate_type = "in" if seg == 0 else "out"
                    pos = select_external_port(candidate_type, self.grid, task)
                    if pos is None:
                        self.rewards[agent] = self.routing_penalty * self.reward_scale
                        continue
                        # return self._get_observation(), self.routing_penalty * self.reward_scale, False, False, {
                        #     "error": "未找到合适外部端"}
                    seg_state["current_position"] = pos
                    seg_state["route"].append(pos)
                    r, c = pos
                    self.grid[r, c] = 0.01
                    self.rewards[agent] -= 1

                    if seg == 2 and (pos[0] in [0, self.rows - 1] or pos[1] in [0, self.cols - 1]):
                        task["seg2"]["external_port"] = pos
                        task["completed"] = True
                        self.rewards[agent] += self.reward_completed * self.reward_scale
                        self.terminations[agent] = True
                    continue
                    # return self._get_observation(), reward * self.reward_scale, self._all_done(), False, info

                # 已选定起点后，延伸当前段
                cur_pos = seg_state["current_position"]
                if cur_pos in target_area and cur_pos not in seg_state["covered"]:
                    if "covered" not in seg_state:
                        seg_state["covered"] = []
                    seg_state["covered"].append(cur_pos)
                    self.rewards[agent] += self.reward_cover
                    if len(seg_state["covered"]) >= task["required_volume"]:
                        task["seg0"]["head"] = seg_state["covered"][0]
                        task["seg0"]["tail"] = cur_pos
                        # seg0 完成，切换到 seg1
                        task["seg1"]["A"] = cur_pos
                        task["seg1"]["current_position"] = cur_pos
                        task["seg1"]["route"].append(cur_pos)
                        task["current_segment"] = 1
                        self.rewards[agent] += self.reward_cover
                drc = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
                delta = drc[direction]
                new_pos = (cur_pos[0] + delta[0], cur_pos[1] + delta[1])
                if not (0 <= new_pos[0] < self.rows and 0 <= new_pos[1] < self.cols):
                    self.rewards[agent] += self.routing_penalty * self.reward_scale
                    # infos[agent] = {"error": "越界"}
                    continue
                if not self._in_allowed(task, new_pos, allowed_seg):
                    self.rewards[agent] += self.routing_penalty * self.reward_scale
                    # infos[agent] = {"error": "目标单元不允许或被占用"}
                    continue
                self.rewards[agent] -= 1 * self.reward_scale

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
                                self.rewards[agent] += self.reward_cover * self.reward_scale
                    else:
                        # 检查离开区域时的覆盖情况
                        if cur_pos in task["source_area"]:
                            if len(seg_state["covered"]) < task["required_volume"]:
                                # 连续性中断，直接终止并给予重罚
                                mask_action = self.get_action_masks(task)
                                print(mask_action, seg_state["covered"])
                                print(f"{seg:}mask_action invalid", seg_state["covered"], task["source_area"],
                                      task["required_volume"])
                                self.render()
                                self.rewards[agent] = self.invalid_cover * self.reward_scale
                                continue
                                # return self._get_observation(), self.invalid_cover * self.reward_scale, True, False, {
                                #     "error": "seg0连续覆盖中断"}
                    if len(seg_state["covered"]) >= task["required_volume"]:
                        task["seg0"]["head"] = seg_state["covered"][0]
                        task["seg0"]["tail"] = new_pos
                        task["seg1"]["A"] = new_pos
                        task["seg1"]["current_position"] = new_pos
                        task["seg1"]["route"].append(new_pos)
                        task["current_segment"] = 1
                        self.rewards[agent] += self.reward_cover * self.reward_scale
                elif seg == 1:
                    # seg1：要求连续覆盖 target_area
                    if new_pos in task["target_area"]:
                        # 同seg0的情况
                        if "covered" not in task["seg1"]:
                            seg_state["covered"] = []
                        # 如果连续，则记录新位置
                        if cur_pos in task["target_area"] or len(seg_state["covered"]) == 0:
                            seg_state["covered"].append(new_pos)
                            self.rewards[agent] += self.reward_cover * self.reward_scale
                        else:
                            seg_state["covered"].append(new_pos)
                        if not _can_cover_all(new_pos, task["target_area"]) and cur_pos not in task["target_area"]:
                            print(f"{seg}mask_action {new_pos}", seg_state["covered"], task["target_area"])
                            self.render()
                            mask_action = self.get_action_masks(task)
                            print(mask_action)
                            # if not any(mask_action):
                            self.rewards[agent] += self.invalid_cover * self.reward_scale
                            continue
                            # return self._get_observation(), self.invalid_cover * self.reward_scale, True, True, {
                            #     "deadlock": True}
                    else:
                        if cur_pos in task["target_area"]:
                            if set(seg_state["covered"]) != task["target_area"]:
                                # 如果不连续，则认为连续覆盖中断
                                print(f"{seg}mask_action invalid", seg_state["covered"], task["target_area"])
                                mask_action = self.get_action_masks(task)
                                print(mask_action)
                                self.render()
                                self.rewards[agent] = self.invalid_cover * self.reward_scale
                                continue
                                # return self._get_observation(), self.invalid_cover * self.reward_scale, True, False, {
                                #     "error": "seg1连续覆盖中断"}
                    # 当目标区域全部被连续覆盖后，完成 seg1
                    if set(seg_state["covered"]) == task["target_area"]:
                        task["current_segment"] = 2
                elif seg == 2:
                    # seg2：延伸到网格边界完成任务
                    if new_pos[0] in [0, self.rows - 1] or new_pos[1] in [0, self.cols - 1]:
                        task["seg2"]["external_port"] = new_pos
                        task["completed"] = True
                        self.terminations[agent] = True
                        self.rewards[agent] += self.reward_completed * self.reward_scale
                seg_state["current_position"] = new_pos
                seg_state["route"].append(new_pos)
                self.grid[new_pos[0], new_pos[1]] = 0.01
                # infos[agent] = info
                self.current_step += 1
                if self.current_step >= self.max_steps:
                    self.truncations[agent] = True
                    self.rewards[agent] = self.invalid_cover * self.reward_scale
                # return self._get_observation(), self.invalid_cover * self.reward_scale, True, True, {"deadlock": True}
            self._get_observation()
        else:
            self._clear_rewards()

        # 选择下一个智能体
        self.agent_selection = self._agent_selector.next()
        # 将当前step的所有奖励累加到_cumulative_rewards中：Adds .rewards to ._cumulative_rewards,由下面方法实现
        # for agent, reward in self.rewards.items():
        #     self._cumulative_rewards[agent] += reward
        self._accumulate_rewards()

    def _get_observation(self):
        self.observations = {}
        for agent in self.possible_agents:
            # print(agent, self.terminations.get(agent, True))
            if self.terminations.get(agent, True):
                # 对于 terminated 的 agent，返回固定的（默认）观测

                self.observations[agent] = {
                    "grid": self.grid.copy(),
                    "agent_positions": np.zeros((self.rows, self.cols), dtype=float),
                    "action_mask": np.array([False, False, False, False], dtype=bool)
                }
            else:
                agent_pos = np.zeros((self.rows, self.cols), dtype=float)
                agent_id = int(agent.split('_')[-1])
                task = self.tasks[agent_id]
                seg = task.get("current_segment", 0)
                pos = task.get(f"seg{seg}", {}).get("current_position", None)
                if pos is not None:
                    agent_pos[pos[0], pos[1]] = task["cur_task"] * 0.1
                self.observations[agent] = {
                    "grid": self.grid.copy(),
                    "agent_positions": agent_pos,
                    "action_mask": self.get_action_masks(task)
                }
        return self.observations

    def _in_allowed(self, task, pos, seg):
        """
        判断 pos 是否在当前任务 seg 允许通行的区域
        对于 seg0：允许空闲 (base_grid==0) 或 属于 source_area
        对于 seg1：允许空闲或属于 target_area（也允许在 source_area 内延伸，但希望尽快进入 target_area）
        对于 seg2：仅允许空闲
        此外，将其他任务的 source_area 与 target_area 视为障碍（除非该单元同时也在当前任务允许区域中）。
        """
        r, c = pos
        cur_task = task["cur_task"]
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False
        if self.grid[r, c] == 0:
            return True
        if self.grid[r, c] == 0.01:
            return False
        for other_task in self.tasks:
            if other_task["id"] != task["id"]:
                if pos in other_task.get("source_area", set()) or pos in other_task.get("target_area", set()):
                    if pos not in task.get("source_area", set()) and pos not in task.get("target_area", set()):
                        return False
        base_val = self.base_grid[r, c]
        if base_val in [cur_task * 0.1 + 0.01, cur_task * 0.1 + 0.02]:
            if seg == 0 and pos in task["source_area"]:
                return True
            if seg == 1 and pos in task["target_area"]:
                return True
            return False
        return True

    def _has_legal_moves(self, task):
        """检查某个任务从当前状态是否至少存在一个合法动作"""
        cur_action_mask = self.get_action_masks(task)
        if any(cur_action_mask):
            return True
        return False

    def _check_deadlock(self):
        """如果所有任务都无合法动作，则认为死锁，返回 True"""
        for task in self.tasks:
            if not task["completed"] and self._has_legal_moves(task):
                return False
        return True

    def get_total_action_masks(self):
        masks = np.zeros(self.num_tasks * 4, dtype=bool)
        task_n = 0
        for task in self.tasks:
            base = task_n * 4
            masks[base:base + 4] = self.get_action_masks(task)
            task_n += 1
        return masks

    def get_action_masks(self, task):
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
        masks = np.zeros(4, dtype=bool)
        drc = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

        # 如果任务已经完成，则全 False
        if task["completed"]:
            masks[:] = True
            return masks
        current_seg = task["current_segment"]
        seg_key = f"seg{current_seg}"
        pos = task[seg_key]["current_position"]
        # 如果当前位置未确定，则允许所有动作
        if pos is None:
            masks[:] = True
            return masks
        # 先计算通用合法动作（不考虑连续性的限制）
        allowed = []
        for direction in range(4):
            delta = drc[direction]
            new_pos = (pos[0] + delta[0], pos[1] + delta[1])
            if not self._in_allowed(task, new_pos, current_seg):
                allowed.append(False)
            else:
                allowed.append(True)
        # 如果当前 segment 为 seg0 或 seg1，需要检查连续覆盖的限制
        if current_seg in [0, 1]:
            # 对 seg0，区域为 source_area；对 seg1，区域为 target_area
            region = task["source_area"] if current_seg == 0 else task["target_area"]
            is_over = True if current_seg == 1 or (current_seg == 0 and
                                                   len(task[seg_key]["covered"]) < task[
                                                       "required_volume"]) else False
            # 如果当前位置已经在区域内，则检查四个方向中能否保持在区域内
            # print(current_seg, region, is_over, pos)
            if pos in region and is_over:
                continuous = []
                for direction in range(4):
                    delta = drc[direction]
                    new_pos = (pos[0] + delta[0], pos[1] + delta[1])
                    # 新位置要在区域内，并且满足基本合法条件
                    if (self._in_allowed(task, new_pos, current_seg) and
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
                if current_seg == 1 and pos not in region:
                    for direction in range(4):
                        if allowed[direction]:
                            delta = drc[direction]
                            new_pos = (pos[0] + delta[0], pos[1] + delta[1])
                            if new_pos in task["target_area"]:
                                if not _can_cover_all(new_pos, task["target_area"]):
                                    allowed[direction] = False
                branch_mask = allowed
        else:
            branch_mask = allowed
        masks[:] = branch_mask
        return masks

    def render(self, mode="human"):
        symbol_map = {0: ".", 1: "#", 0.01: "*"}
        grid_disp = np.full((self.rows, self.cols), "", dtype=object)
        for r in range(self.rows):
            for c in range(self.cols):
                val = self.grid[r, c]
                grid_disp[r, c] = symbol_map.get(val, str(val))
        for task in self.tasks:
            cur_task = task["cur_task"]
            for pos in task["source_area"]:
                r, c = pos
                # print(r, c, type(self.base_grid[r, c]), type(self.grid[r, c]), type(cur_task + 0.1))
                if self.base_grid[r, c] == cur_task * 0.1 + 0.01 and self.grid[r, c] == cur_task * 0.1 + 0.01:
                    grid_disp[r, c] = f"{cur_task}"
            for pos in task["target_area"]:
                r, c = pos
                if self.base_grid[r, c] == cur_task * 0.1 + 0.02 and self.grid[r, c] == cur_task * 0.1 + 0.02:
                    grid_disp[r, c] = f"{cur_task + len(self.tasks)}"
            seg = task["current_segment"]
            pos = task[f"seg{seg}"]["current_position"]
            if pos is not None:
                grid_disp[pos[0], pos[1]] = "A"
        for r in range(self.rows):
            print(" ".join(grid_disp[r]))
        print()


if __name__ == "__main__":
    env = env(render_mode="human")
    env.reset(seed=42)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            action = env.action_space(agent).sample()
        if action is not None:
            print_info = ""
            direction = action
            if direction == 0:
                print_info = "上"
            elif direction == 1:
                print_info = "下"
            elif direction == 2:
                print_info = "左"
            elif direction == 3:
                print_info = "右"
            mask_action = env.get_action_masks(env.tasks[env.agent_name_mapping[agent]])
            print(f"mask_action: {mask_action}")
            print(
                f"{agent} 动作:{direction}({print_info}), 奖励: {reward}, termination: {termination}"
                f", truncation: {truncation}, seg: {env.tasks[int(agent[6:])]['current_segment']}")
        env.step(action)
    env.render()
    env.close()
    # print("Multi-agent测试结束")
    # print("各agent总奖励：", total_rewards)
    # print("步数：", step_count)
