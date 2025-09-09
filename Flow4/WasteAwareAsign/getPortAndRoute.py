import numpy as np
from collections import deque
from typing import List, Set, Tuple, Optional


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def find_boundary_ports(grid: np.ndarray) -> List[Tuple[int, int]]:
    R, C = grid.shape
    ports = []
    for c in range(C):
        if grid[0, c] == 0:
            ports.append((0, c))
        if grid[R - 1, c] == 0:
            ports.append((R - 1, c))
    for r in range(R):
        if grid[r, 0] == 0:
            ports.append((r, 0))
        if grid[r, C - 1] == 0:
            ports.append((r, C - 1))
    return list(set(ports))


def bfs_shortest(
        grid: np.ndarray,
        visited_area: Set[Tuple[int, int]],
        start: Tuple[int, int],
        goal: Tuple[int, int]
) -> List[Tuple[int, int]]:
    R, C = grid.shape
    if start == goal:
        return [start]
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visited = {start} | visited_area
    queue = deque([(start, None)])
    came_from = {start: None}
    while queue:
        cur, _ = queue.popleft()
        if cur == goal:
            path, p = [], cur
            while p is not None:
                path.append(p)
                p = came_from[p]
            return path[::-1]
        for dr, dc in dirs:
            nr, nc = cur[0] + dr, cur[1] + dc
            if not (0 <= nr < R and 0 <= nc < C):
                continue
            if grid[nr, nc] != 0:
                continue
            nxt = (nr, nc)
            if nxt in visited:
                continue
            visited.add(nxt)
            came_from[nxt] = cur
            queue.append((nxt, cur))
    return []


def enumerate_paths(area: Set[Tuple[int, int]], length: int):
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


def find_nearest_boundary_path(
        grid: np.ndarray,
        visited_area: Set[Tuple[int, int]],
        start: Tuple[int, int]
) -> Tuple[Optional[Tuple[int, int]], List[Tuple[int, int]]]:
    R, C = grid.shape
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    blocked = set(visited_area)
    queue = deque([(start, None)])
    came_from = {start: None}
    visited = {start} | blocked
    while queue:
        cell, _ = queue.popleft()
        r, c = cell
        if (r == 0 or r == R - 1 or c == 0 or c == C - 1) and grid[r, c] == 0:
            path, p = [], cell
            while p is not None:
                path.append(p)
                p = came_from[p]
            return cell, list(reversed(path))
        for dr, dc in dirs:
            nr, nc = cell[0] + dr, cell[1] + dc
            if not (0 <= nr < R and 0 <= nc < C):
                continue
            if grid[nr, nc] != 0:
                continue
            nxt = (nr, nc)
            if nxt in visited:
                continue
            visited.add(nxt)
            came_from[nxt] = cell
            queue.append((nxt, cell))
    return None, []


def select_full_route_general(
        grid: np.ndarray,
        areas: List[Set[Tuple[int, int]]],
        required_volumes: Optional[List[int]] = None,
        blocked_cells: Optional[Set[Tuple[int, int]]] = None
) -> Optional[Tuple[
    Tuple[int, int],  # input port I
    List[Tuple[Tuple[int, int], Tuple[int, int]]],  # entry/exit for each area
    Tuple[int, int],  # output port Q
    float,  # total cost
    List[Tuple[int, int]]  # full path
]]:
    R, C = grid.shape
    # 初始化障碍集合：包括 grid != 0 和用户指定的 blocked_cells
    grid_blocked = {(r, c) for r in range(R) for c in range(C) if grid[r, c] != 0}
    blocked = set(blocked_cells) | grid_blocked if blocked_cells else grid_blocked

    ports = find_boundary_ports(grid)
    if required_volumes is None:
        required_volumes = [len(area) for area in areas]

    best = None
    best_cost = float('inf')

    def dfs(idx, prev_exit, visited, path, io_list, cost):
        nonlocal best, best_cost, first_I
        if idx == len(areas):
            Q, end_path = find_nearest_boundary_path(grid, visited, prev_exit)
            if Q and end_path:
                total = cost + len(end_path) - 1
                full = path + end_path[1:]
                if total < best_cost:
                    best_cost = total
                    best = (first_I, io_list.copy(), Q, total, full)
            return

        area = areas[idx]
        L = required_volumes[idx]
        for a_path in enumerate_paths(area, L):
            ent, ext = a_path[0], a_path[-1]
            conn = bfs_shortest(grid, visited, prev_exit, ent)
            if not conn:
                continue
            cost_conn = len(conn) - 1
            new_visited = visited | set(conn) | set(a_path)
            new_path = path + conn[1:] + a_path
            new_cost = cost + cost_conn + (L - 1)
            io_list.append((ent, ext))
            dfs(idx + 1, ext, new_visited, new_path, io_list, new_cost)
            io_list.pop()

    for _first in ports:
        first_I = _first
        dfs(0, first_I, set(blocked), [first_I], [], 0)

    return best


# === 测试 & 格式化输出 ===
if __name__ == "__main__":
    grid = np.zeros((10, 10), dtype=int)
    # 设置一些障碍
    # grid[3, 2] = 1
    # grid[4, 3] = 1
    blocked = set()
    blocked.add((3, 2))
    blocked.add((4, 3))
    blocked.add((3, 3))
    areas = [
        {(1, 1), (1, 2), (2, 2)},
        {(4, 4), (4, 5), (5, 5)},
        {(8, 8)},
        {(2, 8)}
    ]

    res = select_full_route_general(grid, areas, blocked_cells=blocked)
    if res is None:
        print("未找到可行路径")
    else:
        I, io_list, Q, total_cost, full_path = res
        # 这里只演示前两个区域的标记
        (i, o) = io_list[0]
        (t, T_internal) = io_list[1]

        print("选到的入口/出口：")
        print(f"  I = {I},  i(src入口) = {i},  o(src出口) = {o}")
        print(f"  t = {t},  T_internal(目标内部) = {T_internal},  Q = {Q}")
        print("总代价：", total_cost)
        print("完整坐标序列 full_path (共 %d 步)：" % len(full_path))
        print(full_path)

        # 可视化
        R, C = grid.shape
        char_grid = np.full((R, C), '·', dtype=str)
        for r in range(R):
            for c in range(C):
                if grid[r, c] != 0:
                    char_grid[r, c] = '#'  # 障碍
        for (r, c) in full_path:
            if (r, c) == I:
                char_grid[r, c] = 'I'
            elif (r, c) == Q:
                char_grid[r, c] = 'Q'
            elif (r, c) == i:
                char_grid[r, c] = 'i'
            elif (r, c) == o:
                char_grid[r, c] = 'o'
            elif (r, c) == t:
                char_grid[r, c] = 't'
            elif (r, c) == T_internal:
                char_grid[r, c] = 'X'
            else:
                char_grid[r, c] = 'x'
        print("\n可视化：")
        for row in char_grid:
            print(" ".join(row))
