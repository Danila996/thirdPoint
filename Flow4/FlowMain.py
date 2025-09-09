# src/pipeline.py
import json
import os
from typing import Dict, List, Tuple
import numpy as np

# 布局与布线模型接口
from PlacementCode.LayoutModel import LayoutModel
from RoutingCode.RoutingModel import RoutingModel
from conversion_utils import modules_to_tasks  # 转换模块到布线环境 tasks 格式
from getPairPort2 import select_full_route
from routing_util import select_external_port
from WasteAwareAsign.wastAware import WasteGrid, getCurWasteAttribute

total_time = 0.0


def load_test_cases(path: str = "tests/test_cases3.json") -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_routing_result(routing_result):
    """
    以可读的列表形式打印 routing_result：
      - 显示总线长
      - 显示每条路径的编号、段数，以及路径点的坐标序列
    """
    paths = routing_result.get('paths', {})
    total = routing_result.get('total_wire_length', None)

    # 总线长
    if total is not None:
        print(f"总线长: {total}\n")

    # 遍历每条路径
    for pid, coords in paths.items():
        # coords 预计是列表 of (x,y) 元组
        n = len(coords)
        print(f"Path {pid} ({n} 段):")
        # 坐标拼接
        line = " → ".join(f"({x}, {y})" for x, y in coords)
        print(f"  {line}\n")
    return total


def print_tasks_by_stage(tasks_list):
    print("\n📌 转换为布线任务:")
    for stage, tasks in tasks_list.items():
        print(f"\n--- Stage {stage} ---")
        for task in tasks:
            print(f"  🧪 Task ID: {task['id']}")
            print(f"     Type      : {task['task_type']}")
            print(f"     Volume    : {task['required_volume']}")
            print(f"     Source    : {sorted(task['source_area']) if task['source_area'] else '∅'}")
            print(f"     Target    : {sorted(task['target_area']) if task['target_area'] else '∅'}")
            print(f"     Obstacles : {sorted(task['obstacles']) if task['obstacles'] else '∅'}")


def pipeline_run(name, test_case: Dict, R=10, C=10, time_per_unit: float = 1.0, stage_fixed_time: float = 30.0) -> Dict:
    module_specs = test_case['module_specs']
    reagent_specs = test_case['reagent_specs']
    start_point = test_case['start_point']
    level = test_case['level']

    # 布局预测
    layout_model = LayoutModel.load(os.getenv('LAYOUT_MODEL_PATH', 'ppo_grid_placement_final_' + name))
    raw_positions = layout_model.predict(
        module_specs=module_specs,
        reagent_specs=reagent_specs,
        start_point=start_point,
        level=level
    )

    # 组装 modules
    modules = raw_positions
    # print(f"布局结果:{raw_positions}")
    # 转换为布线任务
    tasks_list = modules_to_tasks(modules, reagent_specs, start_point)
    # print_tasks_by_stage(tasks_list)
    # print(tasks_list)

    # 布线预测
    routing_model = RoutingModel.load(os.getenv('ROUTING_MODEL_PATH', 'parallel_grid_routing_v0'))
    tasks_num = 0
    global_grid = np.zeros((R, C), dtype=int)
    waste_grid = WasteGrid(global_grid)
    total_time = 0.0
    total_path = 0.0
    total_conflict_points = 0
    for tasks in tasks_list.values():
        phase_waste = []
        if tasks_num >= 1:
            pre_tasks = tasks_list[tasks_num - 1]
            # 获取上阶段产生的废液分布（当前只考虑在source和target区域）
            phase_waste = getCurWasteAttribute(pre_tasks)
        print(f"-------------开始第{tasks_num + 1}阶段布线---------------")
        print(f"上一阶段废液分布：{phase_waste}")
        waste_grid.addWasteTasks(phase_waste, tasks)
        # 对记录排除任务/微调布局的任务进行布线
        routing_result = routing_model.predict(tasks=tasks)
        tasks_init = routing_result.get('tasks', {})
        # 处理废液
        tasks_result = waste_grid.addressWasteForTasks(tasks_init)
        phases = waste_grid.build_phases(tasks_result)
        print(f"--- 本阶段并行分组（{len(phases)} 组）---")
        phase_times = []
        for idx, ph in enumerate(phases, 1):
            seg_counts = []
            ids = []
            for it in ph:
                rt = it.get('route') or []
                seg = max(0, len(rt) - 1)
                seg_counts.append(seg)
                ids.append(it.get('id'))
            phase_time = (max(seg_counts) if seg_counts else 0) * time_per_unit
            phase_times.append(phase_time)
            print(f"  Phase {idx}: ids={ids}, 段数(max)={max(seg_counts) if seg_counts else 0}, 用时={phase_time}")
        # 统计线长、时间、冲突点
        stage_move_time = sum(phase_times)
        stage_time = stage_fixed_time + stage_move_time
        total_time += stage_time
        stage_conflict_points = waste_grid.count_fluid_conflicts(tasks_result, only_reagents=True)
        total_conflict_points += stage_conflict_points
        print(f"-------------第{tasks_num + 1}阶段布线结果：")
        total_path += print_routing_result(routing_result)
        print()
        tasks_num += 1
    print(f"==== 用例: {name}总时间: {total_time} ====")
    print(f"==== 用例: {name}总线长: {total_path} ====")
    print(f"==== 用例: {name}总交点: {stage_conflict_points} ====")
    return total_time, total_path, stage_conflict_points


def main():
    cases = load_test_cases()
    R, C = 10, 10
    result = {}
    for name, tc in cases.items():
        print(f"==== 用例: {name} ====")
        total_time, total_path, total_conflict_points = pipeline_run(name, tc, R, C)
        result[name] = {"total_time": total_time, "total_path":total_path, "total_conflict": total_conflict_points}
    print(f"{'Case':<20} {'Total Time':>12} {'Total Path':>12} {'Conflicts':>10}")
    print("-" * 58)
    for name, s in result.items():
        print(f"{name:<20} {s['total_time']:>12.2f} {s['total_path']:>12.2f} {s['total_conflict']:>10}")


if __name__ == '__main__':
    main()
