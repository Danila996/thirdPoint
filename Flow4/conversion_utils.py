"""
utilities to convert layout modules into routing tasks
"""
from typing import List, Dict, Tuple, Set
from itertools import chain


def adjust_modules(modules: List[Dict], grid_size: Tuple[int, int]) -> None:
    """
    模块位置的原位移动，使任何模块都不会接触网格边界。
    每个模块的左上角（r，c）被夹紧到[1，rows-h-1]和[1，cols-w-1]。
    """
    rows, cols = grid_size
    for m in modules.values():
        r0, c0, h, w = m['position']
        new_r = min(max(r0, 1), rows - h - 1)
        new_c = min(max(c0, 1), cols - w - 1)
        m['position'] = (new_r, new_c, h, w)


def modules_to_tasks(
        modules: List[Dict],
        reagent_specs: Dict[str, Dict],
        start_point: Dict[str, Tuple[int, int]]
) -> List[Dict]:
    """
    将布局输出的 modules 转换为 routing 环境的 tasks 列表和 port_locs 映射。

    - modules: list of dicts, 每项包含：
        'id', 'position' (r,c,h,w), 'reagent_positions', 'generate', 'start_time' 等
    - reagent_specs: 原始试剂规格，mapping module_id -> reagent -> {cells, from}
    - start_point: 外部试剂初始位置 mapping reagent -> (r,c)

    返回:
    - tasks: list of tasks, 每项包含 'id','source_area','target_area',
      'required_volume','task_type','current_segment'
    - port_locs: mapping reagent -> (r,c) 端口坐标
    """
    # 因为外面多了一圈，所以模块要向右下平移
    for m in modules.values():
        r0, c0, h, w = m['position']
        nr, nc = r0 + 1, c0 + 1
        m['position'] = (nr, nc, h, w)
        # 同步更新 module_cells
        m['module_cells'] = {(r, c)
                             for r in range(nr, nr + h)
                             for c in range(nc, nc + w)}

        # 试剂坐标也平移
        new_reagent_positions = {}
        for r, cells in m['reagent_positions'].items():
            new_cells = [(x + 1, y + 1) for (x, y) in cells]
            new_reagent_positions[r] = new_cells
        m['reagent_positions'] = new_reagent_positions

    # 分层生成任务
    tasks_by_level: Dict[int, List[Dict]] = {}
    task_id = 0
    for mod_id, v in modules.items():
        lvl = v['level']
        tasks_by_level.setdefault(lvl, [])
        # module_cells 已在上面计算
        module_cells = v['module_cells']
        for reagent, cells in v['reagent_positions'].items():
            spec = reagent_specs[mod_id][reagent]
            required_vol = spec['cells']
            origin = spec['from']
            if origin == 'external':
                task_type = 'single'
                source_area = set()
                parent_cells = set()
            else:
                task_type = 'double'
                # 母模块 cells
                parent_cells = modules[origin]['module_cells']
                # 源区为母模块所有格子
                source_area = set(parent_cells)
            # 目标区
            target_area = set(cells)
            # 去除重叠
            overlap = source_area & target_area
            if overlap:
                source_area -= overlap
                target_area -= overlap
            obstacles = overlap
            # 重新计算 required_volume
            required_vol = len(target_area)

            task = {
                'id': task_id,
                'module_id': mod_id,
                'module_cells': set(module_cells),
                'parent_cells': set(parent_cells),
                'source_area': set(source_area),
                'target_area': set(target_area),
                'required_volume': required_vol,
                'task_type': task_type,
                'obstacles': set(obstacles)
            }
            tasks_by_level[lvl].append(task)
            task_id += 1

    # 同 level 内重新编号并输出
    final = {}
    for lvl, tlist in tasks_by_level.items():
        for idx, task in enumerate(tlist):
            task['id'] = idx
        final[lvl] = tlist
    return final

# def modules_to_tasks(
#         modules: List[Dict],
#         reagent_specs: Dict[str, Dict],
#         start_point: Dict[str, Tuple[int, int]]
# ) -> List[Dict]:
#     """
#     将布局输出的 modules 转换为 routing 环境的 tasks 列表和 port_locs 映射。
#
#     - modules: list of dicts, 每项包含：
#         'id', 'position' (r,c,h,w), 'reagent_positions', 'generate', 'start_time' 等
#     - reagent_specs: 原始试剂规格，mapping module_id -> reagent -> {cells, from}
#     - start_point: 外部试剂初始位置 mapping reagent -> (r,c)
#
#     返回:
#     - tasks: list of tasks, 每项包含 'id','source_area','target_area',
#       'required_volume','task_type','current_segment'
#     - port_locs: mapping reagent -> (r,c) 端口坐标
#     """
#     # 因为外面多了一圈，所以模块要向右下平移
#     for m in modules.values():
#         r0, c0, h, w = m['position']
#         nr, nc = r0 + 1, c0 + 1
#         m['position'] = (nr, nc, h, w)
#
#         new_reagent_positions = {}
#         for r, cells in m['reagent_positions'].items():
#             new_cells = [(x + 1, y + 1) for (x, y) in cells]
#             new_reagent_positions[r] = new_cells
#         m['reagent_positions'] = new_reagent_positions
#
#     # adjust_modules(modules, (10, 10))
#     # 分层生成任务
#     tasks_by_level: Dict[int, List[Dict]] = {}
#     task_id = 0
#     for v in modules.values():
#         lvl = v['level']
#         tasks_by_level.setdefault(lvl, [])
#         mod_id = v['id']
#         for reagent, cells in v['reagent_positions'].items():
#             # 对当前模块内部的试剂分配
#             spec = reagent_specs[mod_id][reagent]
#             required_vol = spec['cells']
#             origin = spec['from']
#             if origin == 'external':
#                 task_type = 'single'
#                 source_area: Set[Tuple[int, int]] = set()
#             else:
#                 task_type = 'double'
#                 src = modules[origin]['position']
#                 r0, c0, h, w = src
#                 print(f"origin:{origin} is r0:{r0}, c0:{c0}, h:{h}, w:{w}")
#                 source_area = {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}
#             target_area = set(cells)
#             overlap = source_area & target_area
#             if overlap:
#                 source_area -= overlap
#                 target_area -= overlap
#             obstacles = overlap
#             required_vol = len(target_area)
#             task = {'id': task_id, 'source_area': source_area, 'target_area': target_area,
#                     'required_volume': required_vol, 'task_type': task_type, 'obstacles': obstacles}
#             tasks_by_level[lvl].append(task)
#             task_id += 1
#
#     # 3) 同 level 内重新从 0 开始编号
#     tasks: List[Dict] = []
#     for lvl, tlist in sorted(tasks_by_level.items()):
#         for idx, task in enumerate(tlist):
#             task['id'] = idx
#             tasks.append(task)
#     return tasks_by_level
