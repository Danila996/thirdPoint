import numpy as np


def generate_random_grid(grid_shape):
    """
    随机生成基础网格。障碍比例 obstacle_ratio 表示障碍占整个网格的比例。
    障碍用 1 表示，其余单元为 0。
    """
    rows, cols = grid_shape
    grid = np.zeros((rows, cols), dtype=float)
    return grid


def get_centroid(area):
    """计算区域中所有单元的质心，返回 (row, col)"""
    arr = np.array(list(area))
    return np.mean(arr, axis=0)


def generate_random_task(task_id, grid_shape, occupied,
                         min_source_size=2, max_source_size=2,
                         min_target_size=3, max_target_size=3,
                         delta=3):
    """
    随机生成一个任务，同时确保生成的源区域和目标区域与 occupied 中的单元不冲突。
    改进：
      - 源区域为随机矩形区域，但确保所有单元都不在网格最外一圈；
      - 目标区域使用区域生长算法生成，起始点从靠近源区域质心的内部区域采样，
        同时也保证目标区域不占用网格边界。
    """
    rows, cols = grid_shape
    # 生成源区域时，起始位置在 [1, rows-2] 和 [1, cols-2] 内
    while True:
        src_r = np.random.randint(1, rows - 1)
        src_c = np.random.randint(1, cols - 1)
        src_h = np.random.randint(min_source_size, max_source_size + 1)
        src_w = np.random.randint(min_source_size, max_source_size + 1)
        # 确保整个矩形区域不超过网格边界（即最大行索引为 rows-2，最大列索引为 cols-2）
        if src_r + src_h > rows - 1 or src_c + src_w > cols - 1:
            continue
        source_area = {(r, c) for r in range(src_r, src_r + src_h)
                       for c in range(src_c, src_c + src_w)}
        if source_area.isdisjoint(occupied):
            break

    # 计算源区域质心
    centroid = get_centroid(source_area)

    # 目标区域大小随机
    target_size = np.random.randint(min_target_size, max_target_size + 1)
    # 在以质心为中心，限制在内部区域内采样起始点
    valid_start = False
    attempts = 0
    while not valid_start and attempts < 100:
        start_r = int(np.clip(np.random.randint(int(centroid[0] - delta), int(centroid[0] + delta + 1)), 1, rows - 2))
        start_c = int(np.clip(np.random.randint(int(centroid[1] - delta), int(centroid[1] + delta + 1)), 1, cols - 2))
        if (start_r, start_c) in occupied or (start_r, start_c) in source_area:
            attempts += 1
            continue
        target_area = {(start_r, start_c)}
        valid_start = True
    if not valid_start:
        # 退回全局随机选择，但也限制在内部区域
        while True:
            start_r = np.random.randint(1, rows - 1)
            start_c = np.random.randint(1, cols - 1)
            if (start_r, start_c) in occupied or (start_r, start_c) in source_area:
                continue
            target_area = {(start_r, start_c)}
            break

    # 使用区域生长算法生成目标区域，同时保证候选单元不在边界上
    while len(target_area) < target_size:
        candidates = []
        for (r, c) in list(target_area):
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                # 限制在内部区域
                if nr <= 0 or nr >= rows - 1 or nc <= 0 or nc >= cols - 1:
                    continue
                if (nr, nc) not in target_area and (nr, nc) not in occupied and (nr, nc) not in source_area:
                    candidates.append((nr, nc))
        candidates = list(set(candidates))
        if not candidates:
            break
        new_cell = candidates[np.random.randint(0, len(candidates))]
        target_area.add(new_cell)
    if len(target_area) < min_target_size or not target_area.isdisjoint(occupied):
        return generate_random_task(task_id, grid_shape, occupied,
                                    min_source_size, max_source_size,
                                    min_target_size, max_target_size,
                                    delta)
    # 更新 occupied
    occupied.update(source_area)
    occupied.update(target_area)
    spacing = 1
    buffer_zone = set()
    for (r, c) in source_area | target_area:
        for dr in range(-spacing, spacing + 1):
            for dc in range(-spacing, spacing + 1):
                if abs(dr) + abs(dc) <= spacing:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        buffer_zone.add((rr, cc))
    occupied.update(buffer_zone)

    required_volume = len(target_area)
    return {
        "id": task_id,
        "source_area": source_area,
        "target_area": target_area,
        "required_volume": required_volume
    }


if __name__ == "__main__":
    grid_shape = (10, 10)
    grid = np.zeros(grid_shape, dtype=float)
    occupied = set()
    max_tasks = 2
    tasks = [generate_random_task(i + 1, grid_shape, occupied) for i in range(max_tasks)]
    print(tasks)
    for task in tasks:
        for pos in task["source_area"]:
            grid[pos[0], pos[1]] = task["id"] * 10 + 1  # 标记源区域
        for pos in task["target_area"]:
            grid[pos[0], pos[1]] = task["id"] * 10 + 2  # 标记目标区域
    print(grid)
