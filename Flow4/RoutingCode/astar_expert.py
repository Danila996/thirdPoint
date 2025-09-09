# astar_expert.py
import heapq
import numpy as np
import random

from getPairPort2 import select_full_route
import re

def a_star_single(grid: np.ndarray, start: tuple, goals: set) -> list:
    """
    单段 A*：从 start 到 goals 中任意一点
    返回路径列表，找不到返回空列表。
    """
    print(start, goals)

    def h(pos):
        return min(abs(pos[0] - g[0]) + abs(pos[1] - g[1]) for g in goals)

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    rows, cols = grid.shape

    open_heap = []
    heapq.heappush(open_heap, (h(start), 0, start, None))
    came_from = {}
    g_score = {start: 0}
    closed = set()

    while open_heap:
        f, g, current, parent = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        came_from[current] = parent

        if current in goals:
            # 回溯
            path = []
            p = current
            while p is not None:
                path.append(p)
                p = came_from[p]
            return path[::-1]

        for d in neighbors:
            nr, nc = current[0] + d[0], current[1] + d[1]
            nb = (nr, nc)
            if not (0 <= nr < rows and 0 <= nc < cols): continue
            if grid[nr, nc] != 0: continue

            tentative_g = g + 1
            if tentative_g < g_score.get(nb, float('inf')):
                g_score[nb] = tentative_g
                heapq.heappush(open_heap, (tentative_g + h(nb), tentative_g, nb, current))

    return []


def a_star_two_phase(grid: np.ndarray,
                     start: tuple,
                     source_area: set,
                     target_area: set) -> list:
    """
    两段 A*：
      1) start -> source_area
      2) source_reached -> target_area
    返回：完整拼接的路径列表；若第一段未到达 source_area，则返回 []。
    """
    # 1) 第一段
    path1 = a_star_single(grid, start, source_area)
    if not path1:
        return []  # 无法到达第一个区域

    # 2) 从第一段终点继续到第二个区域
    mid = path1[-1]
    path2 = a_star_single(grid, mid, target_area)
    if not path2:
        # 找不到第二段路径时，也可以只返回到第一区域的路径
        return path1

    # 拼接：去掉第二段的起点重复
    return path1 + path2[1:]


def expert_action(env, cagent_id):
    """给定 env 和 agent_id，返回一个动作 0/1/2/3"""
    str_agent = cagent_id
    if not isinstance(cagent_id, int):
        match = re.search(r'\d+$', cagent_id)
        if match:
            cagent_id = int(match.group())
    agent_id = cagent_id
    
    task = env.tasks[agent_id]
    seg = task["current_segment"]
    pos = task[f"seg{seg}"]["current_position"]
    source_set = set(env.tasks[agent_id]["source_area"])
    target_set = set(env.tasks[agent_id]["target_area"])
    # target = (task["source_area"] if seg == 0 else task["target_area"])
    path = task["expert_route"]
    cur_step = task["expert_step"]
    # path = a_star_two_phase(env.base_grid, pos, source_set, target_set)
    if not path:
        # 从 mask 里找合法 action，再随机选一个
        mask = env.get_action_masks(task)  # 返回形如 [True, False, True, True]
        valid_actions = [i for i, m in enumerate(mask) if m]
        if not valid_actions:
            return None  # 连随机都不行就返回 None
        return [int(np.random.choice(valid_actions))], None
    # print(path, pos)
    try:
        actual_step = path.index(pos)
    except ValueError:
        print(f"path{path}, pos{pos}")
        # disp = np.full((10, 10), '·', dtype=str)
        # for spos in source_set: disp[spos] = 'S'
        # for tpos in target_set: disp[tpos] = 'T'
        # disp[pos] = 'A'
        # for row in disp:
        #     print(' '.join(row))
        print("Position not on path:", pos)
        return None
    # print(actual_step, cur_step, pos)
    # print(cagent_id, actual_step, cur_step, pos)
    # print(f"path:{path}, task['route']:{task['route']},pos:{pos}")
    if actual_step != cur_step-1:
        print(actual_step, cur_step, pos)
        # print(path, task["route"])
        print(env.observe(str_agent))
        disp = np.full((10, 10), '·', dtype=str)
        for spos in source_set: disp[spos] = 'S'
        for tpos in target_set: disp[tpos] = 'T'
        disp[pos] = 'A'
        for row in disp:
            print(' '.join(row))
        print("error step")
        exit()
    drc = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}
    if cur_step >= len(path):
        delta = (-1, 0)
    else:
        next_pt = path[cur_step]
        task["expert_step"] += 1
        delta = (next_pt[0] - pos[0], next_pt[1] - pos[1])
    return [drc[delta]], None


def route_action(base_grid, source_area, target_area, pos):
    source_set = source_area
    target_set = target_area
    # target = (task["source_area"] if seg == 0 else task["target_area"])
    path = a_star_two_phase(base_grid, pos, source_set, target_set)
    print(f"path:{path}")
    next_pt = path[1]
    drc = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}
    delta = (next_pt[0] - pos[0], next_pt[1] - pos[1])
    return drc[delta]

