import heapq
import numpy as np
import random
from collections import deque
from typing import List, Tuple, Set, Optional
# from generate_random_task import generate_random_task


# ------------------------------------------------------------
# 基础工具：曼哈顿距离 & 找边界空口
# ------------------------------------------------------------
def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def find_boundary_ports(grid: np.ndarray) -> List[Tuple[int, int]]:
    """
    返回所有空闲的边界单元坐标（上、下、左、右边界中值为0的格子）。
    """
    R, C = grid.shape
    ports = []
    # 上下边界
    for c in range(C):
        if grid[0, c] == 0:           ports.append((0, c))
        if grid[R - 1, c] == 0:       ports.append((R - 1, c))
    # 左右边界
    for r in range(R):
        if grid[r, 0] == 0:           ports.append((r, 0))
        if grid[r, C - 1] == 0:       ports.append((r, C - 1))
    return list(set(ports))


# ------------------------------------------------------------
# BFS 求网格最短路径（障碍： grid[r,c] != 0）
# ------------------------------------------------------------
def bfs_shortest(grid: np.ndarray,
                 visited_area: Set,
                 start: Tuple[int, int],
                 goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    在 grid 中（0可走，非0障碍）用 BFS 找从 start 到 goal 的最短路径坐标序列。
    如果找不到，返回 []。
    """
    R, C = grid.shape
    if start == goal:
        return [start]
    # 四个方向
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    visited = set([start])

    visited.update(visited_area)
    queue = deque([(start, None)])  # 存 (当前坐标, parent_index)
    came_from = {start: None}

    while queue:
        cur, _ = queue.popleft()
        if cur == goal:
            # 回溯路径
            path = []
            p = cur
            while p is not None:
                path.append(p)
                p = came_from[p]
            return path[::-1]

        for dr, dc in dirs:
            nr, nc = cur[0] + dr, cur[1] + dc
            nxt = (nr, nc)
            if not (0 <= nr < R and 0 <= nc < C):
                continue
            if grid[nr, nc] != 0:
                continue
            if nxt in visited:
                continue
            visited.add(nxt)
            came_from[nxt] = cur
            queue.append((nxt, cur))
    return []


# ------------------------------------------------------------
# 枚举“简单路径”段：给定连通区域 area，找长度恰好为 length 的所有路径
# ------------------------------------------------------------
def enumerate_paths(area: Set[Tuple[int, int]], length: int):
    """
    在给定的连通区域 area（任意 set of (r,c)）中，枚举所有简单路径，且路径长度恰好为 length。
    返回生成器，遍历每条 path（list of cells）。
    """
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    area = set(area)

    def dfs(path):
        if len(path) == length:
            yield list(path)
            return
        for dr, dc in dirs:
            nxt = (path[-1][0] + dr, path[-1][1] + dc)
            if nxt in area and nxt not in path:
                path.append(nxt)
                yield from dfs(path)
                path.pop()

    for start in area:
        yield from dfs([start])


# ------------------------------------------------------------
# 给定一条“区域内固定顺序”路径 path，比如 src_path 或 tgt_path，
# 枚举所有边界口 I,O 配对，返回最优 (I, path[0], path[-1], O, cost_full)
# cost_full = manhattan(I, path[0]) + manhattan(path[-1], O)
# ------------------------------------------------------------
def getObstacle(tasks, task) -> Set:
    visited_area = set()
    if tasks is None:
        return visited_area
    for other_task in tasks:
        if other_task["id"] != task["id"]:
            for pos in other_task.get("source_area", set()):
                visited_area.add(pos)
            for pos in other_task.get("target_area", set()):
                visited_area.add(pos)
    return visited_area


# ------------------------------------------------------------
# 给定一条“区域内固定顺序”路径 path，比如 src_path 或 tgt_path，
# 枚举所有边界口 I,O 配对，返回最优 (I, path[0], path[-1], O, cost_full)
# cost_full = manhattan(I, path[0]) + manhattan(path[-1], O)
# ------------------------------------------------------------
def best_ports_for_path(grid: np.ndarray, path: List[Tuple[int, int]], obstacle_area, port_type):
    ports = find_boundary_ports(grid)
    best = None
    best_cost = np.inf
    best_path = []
    start_cell = path[0]
    end_cell = path[-1]
    if port_type:
        obstacle_area.remove(start_cell)
        for I in ports:
            # 1) 检查 I→start_cell 真正可达，并拿到 BFS 路径长度 d1
            path_I_i = bfs_shortest(grid, obstacle_area, I, start_cell)
            if not path_I_i:
                continue
            d = len(path_I_i) - 1
            if d < best_cost:
                best_cost = d
                best = (I, start_cell, end_cell, None, d)
                best_path = path_I_i
        return best_path, best
    else:
        for O in ports:
            do = manhattan(end_cell, O)
            if do < best_cost:
                best_cost = do
                best = (None, start_cell, end_cell, O, do)
        return None, best  # None if ports 为空


def find_nearest_boundary_path(grid, visited_area, start):
    """
    从 start 出发，做 BFS，直到遇到某个空闲的边界格点。
    返回 (Q, path)，path 是从 start 到 Q 的最短路径列表；
    如果完全找不到边界，就返回 (None, [])。
    """
    R, C = grid.shape
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # 先把所有真正的障碍标记出来
    blocked = set(visited_area)
    # BFS 队列：存 (cell, parent)
    from collections import deque
    queue = deque([(start, None)])
    came_from = {start: None}
    visited = {start} | blocked

    while queue:
        cell, _ = queue.popleft()
        r, c = cell
        # 判断是不是边界
        if (r == 0 or r == R - 1 or c == 0 or c == C - 1) and grid[r, c] == 0:
            # 找到最近的边界 Q = cell
            Q = cell
            # 回溯出完整路径
            path = []
            p = cell
            while p is not None:
                path.append(p)
                p = came_from[p]
            return Q, list(reversed(path))
        # 否则继续扩展
        for dr, dc in dirs:
            nxt = (r + dr, c + dc)
            nr, nc = nxt
            if not (0 <= nr < R and 0 <= nc < C):
                continue
            if grid[nr, nc] != 0:
                continue
            if nxt in visited:
                continue
            visited.add(nxt)
            came_from[nxt] = cell
            queue.append((nxt, cell))

    # 全都搜完了也没到边界
    return None, []


# ------------------------------------------------------------
# 改进版select_full_route：不仅返回端口坐标还返回完整坐标序列 full_path
# ------------------------------------------------------------
def select_full_route(grid: np.ndarray,
                      source_area: Set[Tuple[int, int]],
                      target_area: Set[Tuple[int, int]],
                      required_volume: int,
                      tasks: List,
                      task: dict
                      ) -> Optional[Tuple[
    Tuple[int, int],  # I
    Tuple[int, int],  # i = src_path[0]
    Tuple[int, int],  # o = src_path[-1]
    Tuple[int, int],  # t = tgt_path[0]
    Tuple[int, int],  # T_internal = tgt_path[-1]
    Tuple[int, int],  # Q
    float,  # total_cost
    List[Tuple[int, int]]  # full_path = [ ... ]
]]:
    """
    返回 (I, i, o, t, T_internal, Q, total_cost, full_path)，
    full_path 是从 I→…→i →(src_path)→o →…→t →(tgt_path)→T_internal →…→Q 的坐标列表。
    如果找不到任何合法解，则返回 None。
    计算时按照词典序优先 (cost_i, cost_o, cost_rest)。
    """
    best_global = None
    best_key = None  # (cost_i, cost_o, cost_rest)
    obstacle_area = getObstacle(tasks, task)

    if not source_area:
        # 这是对于单目标的情况，只有一个target_area
        best_global = None
        best_key = None
        obstacle_area = getObstacle(tasks, task)

        for tgt_path in enumerate_paths(target_area, len(target_area)):
            visited_area = obstacle_area.copy()
            visited_area.update(target_area)

            # 目标路径入口
            path_I_t, res_entry = best_ports_for_path(grid, tgt_path, visited_area, True)
            if res_entry is None:
                continue
            I, t, T_internal, Q, cost_tgt_full = res_entry

            if len(path_I_t) == 0:
                continue

            for pos in path_I_t:
                visited_area.add(pos)

            # 尾部路径：T_internal → Q
            visited_area.add(t)
            Q, path_T_Q = find_nearest_boundary_path(grid, visited_area, T_internal)
            # if len(path_T_Q) == 0:
            #     continue

            cost_entry = len(path_I_t) - 1
            steps_tgt = len(tgt_path) - 1

            key = cost_entry + steps_tgt

            if best_key is None or key < best_key:
                best_key = key
                total_cost = key
                full = []
                full.extend(path_I_t)
                full.extend(tgt_path[1:])
                if Q is None or not path_T_Q:
                    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    R, C = grid.shape
                    for dr, dc in dirs:  # 先尝试四邻域
                        cand = (T_internal[0] + dr, T_internal[1] + dc)
                        if 0 <= cand[0] < R and 0 <= cand[1] < C and grid[cand] == 0:
                            Q = cand
                            path_T_Q = [T_internal, cand]
                            break
                    else:
                        # 实在找不到就把 Q 设为 T_internal（至少不为空）
                        Q = T_internal
                        path_T_Q = [T_internal]
                full.extend(path_T_Q[1:])
                best_global = (I, t, t, t, T_internal, Q, float(total_cost), full)

        return best_global

    # 先找所有 src_path
    for src_path in enumerate_paths(source_area, required_volume):
        src_visited_area = obstacle_area.copy()
        # 先找段0最优端口 (I, i, o, O_src, cost_src_full)
        temp = target_area.copy()
        src_visited_area.update(temp)
        src_visited_area.update(source_area)

        path_I_i, res_src = best_ports_for_path(grid, src_path, src_visited_area, True)
        if res_src is None:
            continue
        I, i, o, O_src, cost_src_full = res_src
        # 预计算 段0 内部走路步数 = len(src_path)-1
        steps_src = len(src_path) - 1
        # 计算 I → i 的实际路径
        # temp = target_area.copy()
        # visited_area.update(temp)
        # visited_area.update(source_area)
        # visited_area.remove(i)
        # path_I_i = bfs_shortest(grid, visited_area, I, i)
        if len(path_I_i) == 0:
            continue
        for pos in path_I_i:
            src_visited_area.add(pos)
        cost_i = len(path_I_i) + steps_src  # I→i + 区内步数
        # cost_o = manhattan(o, O_src)  # o→O_src

        # 再枚举目标区路径 tgt_path（长度 = |target_area|）
        for tgt_path in enumerate_paths(target_area, len(target_area)):
            visited_area = src_visited_area.copy()  # 阻塞重新设置
            _, res_tgt = best_ports_for_path(grid, tgt_path, visited_area, False)
            if res_tgt is None:
                continue
            _, t, T_internal, Q, cost_tgt_full = res_tgt

            steps_tgt = len(tgt_path) - 1  # 区内步数

            # 计算 o → t 最短路径代价值和实际路径
            visited_area.add(I)
            visited_area.remove(t)
            path_o_t = bfs_shortest(grid, visited_area, o, t)
            if len(path_o_t) == 0:
                # print(f"path_I_i：{path_I_i} make it hard to find path_o_t: o:{o} to t:{t}")
                # R, C = grid.shape
                # obstacle = np.full((R, C), '·', dtype=str)
                # for pos in visited_area:
                #     obstacle[pos] = '*'
                # disp = np.full((R, C), '·', dtype=str)
                # for cell in source_area:           disp[cell] = 'S'
                # for cell in target_area:           disp[cell] = 'T'
                # disp[I] = 'I'
                # disp[i] = 'i'
                # disp[o] = 'o'
                # disp[t] = 'x'
                # disp[T_internal] = 't'
                # disp[Q] = 'Q'
                # for row in disp:
                #     print(' '.join(row))
                # for row in obstacle:
                #     print(' '.join(row))
                continue
            for pos in path_o_t:
                visited_area.add(pos)
            cost_mid = len(path_o_t) - 1  # o 到 t 的步数

            # 计算 T_internal → Q 最短路径代价值和实际路径
            visited_area.add(t)
            # path_T_Q = bfs_shortest(grid, visited_area, T_internal, Q)
            Q, path_T_Q = find_nearest_boundary_path(grid, visited_area, T_internal)
            # if Q is None:
                # print(f"it hard to find path_T_Q: T_internal:{T_internal} to Q:{Q}")
                # continue
            cost_T_Q = len(path_T_Q) - 1

            # 段1边界匹配 cost 已包含在 cost_tgt_full = manhattan(tgt_path[-1], Q_internal)
            # total 步数 = segment 内步数 + BFS 中间步数 + BFS 末端步数
            cost_rest = cost_mid + steps_tgt + cost_tgt_full

            # 总成本
            key = cost_rest + cost_i
            # 这个比较还是有点问题的，先比较cost_i再比较cost_rest，试着反过来看看
            if best_key is None or key < best_key:
                best_key = key
                total_cost = cost_i + cost_rest

                # 现在把各段具体路径都算出来，拼成 full_path
                # 1) boundary I → i

                # print(I, i)
                # print(f"path_I_i:{path_I_i}")
                # for pos in path_I_i:
                #     visited_area.add(pos)
                # 2) src_path 本身
                path_src = src_path[:]  # already a list
                # 3) o → t
                # 注意，path_o_t 的第一点就是 o, 我们和 src_path[-1] 重复了
                # 因此取 path_o_t[1:]
                path_o_t_trunc = path_o_t[1:]
                # 4) tgt_path 本身
                path_tgt = tgt_path[:]
                # 5) T_internal → Q
                if Q is None or not path_T_Q:
                    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    R, C = grid.shape
                    for dr, dc in dirs:  # 先尝试四邻域
                        cand = (T_internal[0] + dr, T_internal[1] + dc)
                        if 0 <= cand[0] < R and 0 <= cand[1] < C and grid[cand] == 0:
                            Q = cand
                            path_T_Q = [T_internal, cand]
                            break
                    else:
                        # 实在找不到就把 Q 设为 T_internal（至少不为空）
                        Q = T_internal
                        path_T_Q = [T_internal]
                path_T_Q_trunc = path_T_Q[1:]

                # 把它们拼起来：
                #   path_I_i + path_src[1:] （去掉 i 重复） + path_o_t_trunc + path_tgt[1:] + path_T_Q_trunc
                full = []
                # I→i
                full.extend(path_I_i)
                # print(f"path_I_i_mini:{path_I_i}")
                # src_path：从 i 开始，但 i 已经在 path_I_i[-1]，所以去掉第一个元素
                full.extend(path_src[1:])
                # o→t：path_o_t_trunc
                full.extend(path_o_t_trunc)
                # tgt_path：从 t 开始，但 t 是 path_o_t[-1], 所以上面的 path_o_t_trunc 已经包含 t；这里去掉第一个元素
                full.extend(path_tgt[1:])
                # T_internal→Q
                full.extend(path_T_Q_trunc)

                best_global = (I, i, o, t, T_internal, Q, float(total_cost), full)

    return best_global


# ------------------------------------------------------------
# 随机测试示例
# ------------------------------------------------------------

#
# if __name__ == "__main__":
#     # 演示一个随机测试
#     R, C = 10, 10
#     grid = np.zeros((R, C), dtype=float)
#     # 随机插一些障碍
#     for _ in range(int(R * C * 0.2)):
#         r = random.randrange(R)
#         c = random.randrange(C)
#         grid[r, c] = 0
#     occupied = set()
#     task = generate_random_task(1, (R, C), occupied)
#     # source_area = {(7, 4), (6, 3), (6, 4), (7, 3)}
#     # target_area = {(4, 5), (5, 5), (3, 5)}
#     source_area = task["source_area"]
#     target_area = task["target_area"]
#     # 放两个随机连通区域
#     # forbidden = set(zip(*np.where(grid != 0)))
#     # source_area = random_connected_area((R, C), size=4, forbidden=forbidden)
#     # forbidden |= source_area
#     # target_area = random_connected_area((R, C), size=4, forbidden=forbidden)
#
#     # 要覆盖的体积
#     required_volume = len(target_area)
#     tasks = []
#     task = {}
#     res = select_full_route(grid, source_area, target_area, required_volume, tasks, task)
#     # 可视化为字符网格
#     char_grid = [['·' for _ in range(C)] for _ in range(R)]
#     for r in range(R):
#         for c in range(C):
#             if grid[r, c] != 0:
#                 char_grid[r][c] = '#'
#     for (r, c) in source_area:
#         char_grid[r][c] = 'S'
#     for (r, c) in target_area:
#         char_grid[r][c] = 'T'
#     if res is None:
#         # 可视化为字符网格
#         for r in range(R):
#             print(" ".join(char_grid[r]))
#         print("未找到合法路径！")
#     else:
#         I, i, o, t, T_internal, Q, total_cost, full_path = res
#         print("选到的入口/出口：")
#         print(f"  I = {I},  i(src入口) = {i},  o(src出口) = {o}")
#         print(f"  t = {t},  T_internal(目标内部) = {T_internal},  Q = {Q}")
#         print("总代价：", total_cost)
#         print("完整坐标序列 full_path (共 %d 步)：" % len(full_path))
#         print(full_path)
#         print(f"source_area:{source_area}, target_area:{target_area}")
#         for idx, (r, c) in enumerate(full_path):
#             # 标注首尾
#             if (r, c) == I:
#                 char_grid[r][c] = 'I'
#             elif (r, c) == Q:
#                 char_grid[r][c] = 'Q'
#             elif (r, c) == i:
#                 char_grid[r][c] = 'i'
#             elif (r, c) == o:
#                 char_grid[r][c] = 'o'
#             elif (r, c) == t:
#                 char_grid[r][c] = 't'
#             elif (r, c) == T_internal:
#                 char_grid[r][c] = 'X'
#             else:
#                 # 如果在路径中，但不是上述几个特殊点，就画 'x'
#                 if (r, c) not in source_area and (r, c) not in target_area:
#                     char_grid[r][c] = 'x'
#
#         print("\n可视化：")
#         for r in range(R):
#             print(" ".join(char_grid[r]))
