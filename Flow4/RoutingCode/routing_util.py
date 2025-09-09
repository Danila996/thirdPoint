import numpy as np


def get_centroid(area):
    """计算区域（集合或列表）中所有单元的质心，返回 (row, col)"""
    arr = np.array(list(area))
    return np.mean(arr, axis=0)


def select_external_port(candidate_type, grid, task):
    """
    根据动作方向和候选类型，从grid边界上选择一个空闲单元作为外部端口候选
    candidate_type: "in" 表示外部输入端候选，"out" 表示外部输出端候选
    选择时倾向于离相应区域（源区域或目标区域）质心较近
    如果找不到合适单元，返回 None
    """
    rows, cols = grid.shape
    port_area = task["source_area"] if candidate_type == "in" and task["task_type"] == "double" \
        else task["target_area"]
    centroid = get_centroid(task["source_area"]) if candidate_type == "in" and task["task_type"] == "double" \
        else get_centroid(task["target_area"])
    candidates = []
    for direction in range(4):
        if direction == 0:  # 上边界
            r = 0
            for c in range(cols):
                if grid[r, c] == 0.01:
                    continue
                if grid[r, c] == 0 or ((r, c) in port_area and _can_cover_all((r, c), port_area)):
                    candidates.append((r, c))
        elif direction == 1:  # 下边界
            r = rows - 1
            for c in range(cols):
                if grid[r, c] == 0.01:
                    continue
                if grid[r, c] == 0 or ((r, c) in port_area and _can_cover_all((r, c), port_area)):
                    candidates.append((r, c))
        elif direction == 2:  # 左边界
            c = 0
            for r in range(rows):
                if grid[r, c] == 0.01:
                    continue
                if grid[r, c] == 0 or ((r, c) in port_area and _can_cover_all((r, c), port_area)):
                    candidates.append((r, c))
        elif direction == 3:  # 右边界
            c = cols - 1
            for r in range(rows):
                if grid[r, c] == 0.01:
                    continue
                if grid[r, c] == 0 or ((r, c) in port_area and _can_cover_all((r, c), port_area)):
                    candidates.append((r, c))
        else:
            return None

    if not candidates:
        return None

    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates - centroid, axis=1)
    idx = np.argmin(dists)
    # print(centroid, tuple(candidates[idx]), candidates)
    return tuple(candidates[idx])


def _can_cover_all(pos, target_area):
    """
        判断给定坐标集合在去掉某坐标后是否在网格中连续
        :param pos, target_area: 需要去除的坐标点，存储坐标元组的任务，如 {(2, 4), (3, 3), (3, 4)}
        :return: True 表示连续，否则 False
        """
    area_set = set(target_area)
    # 转换为 set 以便快速查找
    area_set.remove(pos)
    # 随便选取一个坐标作为起始点
    start = next(iter(area_set))
    visited = set()
    stack = [start]

    while stack:
        x, y = stack.pop()
        if (x, y) in visited:
            continue
        visited.add((x, y))
        # 检查四个方向：上下左右
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor = (x + dx, y + dy)
            if neighbor in area_set and neighbor not in visited:
                stack.append(neighbor)
    # print("_can_cover_all:", pos, task, len(visited) == len(area_set))
    return len(visited) == len(area_set)
