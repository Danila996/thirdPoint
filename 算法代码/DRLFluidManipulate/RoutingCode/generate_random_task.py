import numpy as np


def generate_random_grid(grid_shape):
    """
    随机生成基础网格。障碍比例 obstacle_ratio 表示障碍占整个网格的比例。
    障碍用 1 表示，其余单元为 0。
    """
    rows, cols = grid_shape
    grid = np.zeros((rows, cols), dtype=float)
    return grid


def generate_random_task(task_id, grid_shape, occupied,
                         min_source_size=2, max_source_size=2,
                         min_target_size=3, max_target_size=3):
    """
    随机生成一个任务，同时确保生成的源区域和目标区域与 occupied 中的单元不冲突。
    - occupied: 一个集合，包含已经被其他任务占用的单元（包括源区域和目标区域）
    - 源区域为随机矩形区域，目标区域使用区域生长算法生成（形状不固定）。
    - 生成后更新 occupied 集合。
    """
    rows, cols = grid_shape
    # 随机生成源区域（矩形）
    while True:
        src_r = np.random.randint(0, rows)
        src_c = np.random.randint(0, cols)
        src_h = np.random.randint(min_source_size, max_source_size + 1)
        src_w = np.random.randint(min_source_size, max_source_size + 1)
        if src_r + src_h > rows or src_c + src_w > cols:
            continue
        source_area = {(r, c) for r in range(src_r, min(rows, src_r + src_h))
                       for c in range(src_c, min(cols, src_c + src_w))}
        # 检查是否与已占用区域有重叠
        if source_area.isdisjoint(occupied):
            break

    # 随机生成目标区域（使用区域生长算法）
    target_size = np.random.randint(min_target_size, max_target_size + 1)
    while True:
        start_r = np.random.randint(0, rows)
        start_c = np.random.randint(0, cols)
        # 确保起始点不在已占用区域和源区域中
        if (start_r, start_c) in occupied or (start_r, start_c) in source_area:
            continue
        target_area = {(start_r, start_c)}
        while len(target_area) < target_size:
            candidates = []
            for (r, c) in list(target_area):
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if (nr, nc) not in target_area and (nr, nc) not in occupied and (nr, nc) not in source_area:
                            candidates.append((nr, nc))
            candidates = list(set(candidates))
            if not candidates:
                break  # 无法扩展时跳出循环
            new_cell = candidates[np.random.randint(0, len(candidates))]
            target_area.add(new_cell)
        # 如果目标区域大小满足要求且不冲突则退出循环
        if len(target_area) >= min_target_size and target_area.isdisjoint(occupied):
            break
    # 更新 occupied：将生成的源区域和目标区域都标记为占用
    occupied.update(source_area)
    occupied.update(target_area)

    required_volume = len(target_area)
    return {
        "id": task_id,
        "source_area": source_area,
        "target_area": target_area,
        "required_volume": required_volume
    }


if __name__ == "__main__":
    # 重新生成基础网格（也可以保持固定，仅随机障碍）
    grid_shape = (10, 10)
    grid = generate_random_grid(grid_shape)
    occupied = set()
    tasks = []
    max_tasks = 2  # 或其他你想要生成的任务数量
    tasks = [generate_random_task(i + 1, grid_shape, occupied) for i in range(max_tasks)]
    print(tasks)
    for task in tasks:
        for pos in task["source_area"]:
            grid[pos[0], pos[1]] = task["id"] * 10 + 1  # 标记任务的源区域
        for pos in task["target_area"]:
            grid[pos[0], pos[1]] = task["id"] * 10 + 2  # 标记任务的目标区域
    print(grid)