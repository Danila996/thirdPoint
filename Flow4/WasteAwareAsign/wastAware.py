from typing import List
from itertools import combinations

import numpy as np
import heapq
from WasteAwareAsign.getPortAndRoute import select_full_route_general


def getWasteAttribute(area, route):
    waste = []
    min_index = len(route)
    max_index = -1

    for pos in area:
        if pos in route:
            idx = route.index(pos)
            if idx < min_index:
                min_index = idx
            if idx > max_index:
                max_index = idx
        else:
            # area 中有不在 route 里的，直接算作 waste
            waste.append(pos)

    # 如果找到了至少一个 pos 在 route 里，插入前一个或自身
    if 0 <= min_index < len(route):
        prev = min_index - 1 if min_index > 0 else min_index
        waste.append(route[prev])

    # 如果 max_index 不是最后一个，插入下一个
    if 0 <= max_index < len(route) - 1:
        waste.append(route[max_index + 1])

    return waste


def manhattan_region(area_a, area_b, return_cells=False):
    """计算 area_a 与 area_b 的曼哈顿包围（最小矩形），并可返回所有格子列表"""
    pts = list(area_a) + list(area_b)
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cells = []
    if return_cells:
        for i in range(x_min, x_max + 1):
            for j in range(y_min, y_max + 1):
                cells.append((i, j))
    return (x_min, y_min), (x_max, y_max), cells


def clusterWaste(waste_set):
    """把一堆 (r,c) 点按 4 邻域做连通分量，返回 List[Set[(r,c)]]。"""
    clusters = []
    unvis = set(waste_set)
    while unvis:
        start = unvis.pop()
        comp = {start}
        queue = [start]
        for pos in queue:
            r, c = pos
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nei = (r + dr, c + dc)
                if nei in unvis:
                    unvis.remove(nei)
                    comp.add(nei)
                    queue.append(nei)
        clusters.append(comp)
    return clusters


def canParallelFlush(flush_route, tasks):
    """检测 flush_route 是否和已有 tasks 冲突。"""
    if flush_route is None:
        return False
    for t in tasks:
        if 'route' in t:
            if set(flush_route) & set(t['route']):  # 路径冲突
                return False
            if set(flush_route) & t['obstacles']:  # 占用冲突
                return False
        else:
            return False
    return True


def astar(start, goal, forbidden, R, C):
    """
    A* 寻路：从 start 到 goal，避开 forbidden 中的格子。
    start, goal: (r,c) 二元组
    forbidden: set of (r,c)
    R, C: 网格行数和列数
    返回：List[(r,c)] 从 start 到 goal 的最短路径
    抛出 ValueError 如果找不到可行路径
    """

    def heuristic(a, b):
        # 曼哈顿启发
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = [(heuristic(start, goal), start)]
    came_from = {}  # 记录走过的前驱
    g_score = {start: 0}
    visited = set()

    while open_set:
        f, current = heapq.heappop(open_set)
        if current == goal:
            # 重建路径
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        if current in visited:
            continue
        visited.add(current)

        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nbr = (current[0] + dr, current[1] + dc)
            # 边界检查
            if not (0 <= nbr[0] < R and 0 <= nbr[1] < C):
                continue
            if nbr in forbidden:
                continue

            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(nbr, float('inf')):
                came_from[nbr] = current
                g_score[nbr] = tentative_g
                heapq.heappush(open_set, (tentative_g + heuristic(nbr, goal), nbr))
    return None


def re_planRoute(old_route, forbidden, R, C):
    """
    在 old_route 的冲突区间做局部切段 + 重连：
      1. 找出所有冲突点在 old_route 中的索引区间 [i0, i1]
      2. 入口 entry = old_route[i0-1]（或起点），出口 exit = old_route[i1+1]（或终点）
      3. 调用 A* 在 entry->exit 之间重连，得到 segment
      4. 拼接：old_route[:i0] + segment[1:-1] + old_route[i1+1:]
    old_route: List[(r,c)]
    forbidden: set of (r,c)
    R, C: 网格尺寸
    返回：新的完整路径 List[(r,c)]
    """
    # 1) 找冲突索引
    conflict_idxs = [i for i, p in enumerate(old_route) if p in forbidden]
    if not conflict_idxs:
        return old_route  # 无冲突，直接返回

    i0, i1 = min(conflict_idxs), max(conflict_idxs)
    entry = old_route[i0 - 1] if i0 > 0 else old_route[0]
    exit = old_route[i1 + 1] if i1 < len(old_route) - 1 else old_route[-1]
    prefix = set(old_route[:i0]) - {entry}
    suffix = set(old_route[i1 + 1:]) - {exit}
    extended_forbidden = set(forbidden) | prefix | suffix
    # 2) 用 A* 找局部无冲突路径
    segment = astar(entry, exit, extended_forbidden, R, C)
    if segment is None:
        return segment
    # 3) 拼接前缀 + 新段 + 后缀
    new_route = []
    # 前缀：0 .. i0-1
    new_route.extend(old_route[:i0])
    # 中段：segment 去首尾
    if len(segment) >= 2:
        new_route.extend(segment[1:-1])

    # 后缀：i1+1 .. end
    new_route.extend(old_route[i1 + 1:])
    return new_route


def getCurWasteAttribute(tasks):
    #  获取布线任务产生的废液列表
    waste = []
    for task in tasks:
        source_area = task["source_area"]
        target_area = task["target_area"]
        route = task["route"]
        waste.extend(getWasteAttribute(source_area, route))
        waste.extend(getWasteAttribute(target_area, route))
    return waste


class WasteGrid:
    def __init__(self, global_grid):
        self.R, self.C = global_grid.shape
        self.global_grid = np.zeros((self.R, self.C), dtype=int)
        self.global_waste = set()
        self.flush_num = 0
        self.flushTasks = []
        self.severe_waste = set()
        self.dangerous_waste = set()

    def addWasteTasks(self, phase_waste, tasks):
        # 将当前获取的废液分布加入到全局废液分布中
        self.addWasteToGlobal(phase_waste)
        # 添加排除任务
        severe_all = []
        dangerous_all = set()
        for task in tasks:
            area_s = task['source_area']
            area_t = task['target_area']
            #  获取任务形成的区域是否会有废液
            total_conflict = set(self.getCurWasteConflict(area_s, area_t))
            if total_conflict:
                # 1、在源、目标区域内部的废液区域：严重废液区域
                severe_s = total_conflict & area_s
                severe_t = total_conflict & area_t
                # 2、微调布局：看是否可以在不影响重叠覆盖的情况下调整模块位置，消除严重废液
                severe = severe_s | severe_t
                ok = None
                if len(severe) != 0:
                    ok, new_bbox = self.microAdjustLayout(task, self.global_waste, tasks)
                if ok:
                    new_area_s = task['source_area']
                    new_area_t = task['target_area']
                    total_conflict = set(self.getCurWasteConflict(new_area_s, new_area_t))
                    dangerous = sorted(total_conflict)
                else:
                    # 3、将严重废液则记录下来
                    severe_all.append(severe_s)
                    severe_all.append(severe_t)
                    # 记录在源、目标区域围成曼哈顿区域的废液区域：危险废液区域
                    dangerous = sorted(total_conflict - severe_s - severe_t)
                dangerous_all.update(dangerous)
        # 全局严重废液集合
        severe_waste_set = set().union(*severe_all)
        self.severe_waste.update(severe_waste_set)
        self.dangerous_waste.update(dangerous_all)
        # 生成对应的排除任务
        flush_tasks = self.planFlushTasks(severe_waste_set, tasks, 0)
        self.flushTasks = flush_tasks

    def addressWasteForTasks(self, tasks):
        # waste_fluid = set(self.getCurFluidConflictWithWaste(tasks))
        # print(f"waste_fluid:{waste_fluid}")
        obstacle_all = {}
        for task in tasks:
            obstacle = []
            for pos in self.dangerous_waste:
                if pos in task["route"]:
                    obstacle.append(pos)
            obstacle_all[task["id"]] = obstacle
        # 这个obstacle_all每个任务的阻碍废液可能会重叠，这样后面要记得处理

        # for ft in self.flushTasks:
        #     # 这边的任务实际上是对严重废液的排除进行处理
        #     parallel_flush = ft['parallel']
        #     if parallel_flush:
        #         tasks.append(ft)
        #     else:
        #         # 串行：立即清洗(这边还是要修改的）
        #         print(f"execute waste:{ft['waste_points']}")
        #         self.flushWaste(ft['waste_points'])
        #         if len(ft['waste_points']) > self.flush_time:
        #             self.flush_time = len(ft['waste_points'])
        has_address = []
        for t in tasks:
            tid = t["id"]
            obs = obstacle_all[tid]
            for pos in obs:
                if pos in has_address:
                    obs.remove(pos)
            if not obs:
                continue
            new_route, delta_reward, flush_task = self.handleObstacleWaste(
                t, obs, tasks
            )
            # 更新 task
            t['route'] = new_route
            if flush_task:
                dep_ids = [ft['id'] for ft in flush_task]
                t.setdefault('_deps', set()).update(dep_ids)
                self.flushTasks.extend(flush_task)
                has_address.extend(obs)
                # self.flushWaste(obs)
                # self.flushTasks.extend(flush_task)
        return tasks

    def handleObstacleWaste(self, task, obstacle_cells, tasks,
                            reroute_threshold=5):
        """
        对单个 task 的障碍废液做处理：
          1) 尝试绕路
          2) 若 Δ 太大或失败，微调布局
          3) 再失败，生成并行 flush_task
        Returns: (new_route, reward_delta, flush_task_or_None)
        """
        orig_route = task['route']
        orig_len = len(orig_route)

        # 新的 forbidden 集合
        forbidden = set(self.global_waste) | set(obstacle_cells)
        # 1) 尝试局部“切段+重连”
        new_route = re_planRoute(orig_route, forbidden, self.R, self.C)
        if new_route:
            delta = len(new_route) - orig_len
            # 如果增量在阈值内，就接受绕行
            if delta <= reroute_threshold:
                reward = -0.2 * delta
                return new_route, reward, None

        # # 2) 微调布局
        # ok, new_bbox = self.microAdjustLayout(task, forbidden, tasks)
        # print(f"new_bbox:{new_bbox}")
        #
        # if ok:
        #     # 重新生成完整 route（避开所有 global_waste）
        #     try:
        #         rerouted = re_planRoute(orig_route, self.global_waste, self.R, self.C)
        #     except ValueError:
        #         rerouted = None
        #     if rerouted:
        #         return rerouted, -1.0, None
        # 3) Fallback：并行 flush
        flush_task = self.planFlushTasks(obstacle_cells, tasks, 1)
        # flush_task = {
        #     'id': f'flush_obs_{task["id"]}_{self.step}',
        #     'task_type': 'flush',
        #     'waste_points': set(obstacle_cells),
        #     'route': self.planCoverageRoute(set(obstacle_cells), tasks),
        #     'obstacles': set().union(*(t.get('obstacles', set()) for t in tasks)),
        #     'parallel': True
        # }
        return orig_route, -0.5 * len(obstacle_cells), flush_task

    def microAdjustLayout(self, task, avoid_cells, tasks):
        """
        基于微小平移避开 avoid_cells：
        - task: 包含 'source_area'、'target_area'（Set[(r,c)])
        - avoid_cells: 障碍废液格子 Set[(r,c)]
        - tasks: 本阶段所有任务列表（用于计算模块间重叠）
        返回 (success, (new_source_area, new_target_area)) 或 (False, None)
        """
        # 原区域集合
        src, tgt = task['source_area'], task['target_area']
        forbidden = set()
        forbidden.update(avoid_cells)

        # 计算原 overlap 与原 conflict
        def count_overlap(area):
            forbid = set()
            cnt = 0
            for other in tasks:
                if other is task:
                    continue
                other_area = other['source_area'] | other['target_area']
                forbid = forbid | other['obstacles']
                cnt += len(area & other_area)
            return cnt, forbid

        best = (None, 0.0)  # ( (new_src, new_tgt), cost )
        # 四个微移方向
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            # 平移后的区域
            new_src = {(r + dr, c + dc) for r, c in src}
            new_tgt = {(r + dr, c + dc) for r, c in tgt}
            new_area = new_src | new_tgt

            # 越界检测
            if any(r < 0 or r >= self.R or c < 0 or c >= self.C for r, c in new_area):
                continue

            # 计算新 overlap 和新 conflict
            new_overlap, forbid = count_overlap(new_area)
            new_set = set(new_area)
            avoid_set = set(forbidden | forbid)
            new_conflict = len(new_set & avoid_set)
            if new_overlap > 0 or new_conflict > 0:
                continue
            best = (new_src, new_tgt)
            break

        # 决策：只有最优 cost < 0 时才接受
        areas = best
        if areas[0]:
            task['source_area'], task['target_area'] = areas
            return True, areas
        else:
            return False, None

    def addWasteToGlobal(self, cur_waste):
        #  添加当前废液分布到全局分布
        for pos in cur_waste:
            self.global_grid[pos[0], pos[1]] = 1
        if hasattr(cur_waste, "ndim"):  # numpy
            self.global_waste.update(map(tuple, cur_waste))
        else:
            self.global_waste.update(tuple(p) for p in cur_waste)
        # self.global_waste.update(cur_waste)

    def getCurWasteConflict(self, area_a, area_b):
        #  在给定区域下求曼哈顿区域是否有废液
        if len(area_b) == 0:
            return []
        bottom_left, top_right, cells = manhattan_region(area_a, area_b, True)
        conflict = []
        for pos in cells:
            if self.global_grid[pos[0], pos[1]] == 1:
                conflict.append(pos)
        return conflict

    def mergeClusters(self, clusters, tasks):
        """
        动态地把初始簇列表合并成最终簇：
        只要两个簇的合并簇能规划出一条无冲突的 flush 路径，就把它们合并。
        """
        merged = [set(c) for c in clusters]
        changed = True

        # 已经「确定不再与其他簇合并」的簇，它的 flush_route 存在于 final_routes
        final_routes = [None] * len(merged)
        while changed:
            changed = False
            # 逐对尝试合并
            for i in range(len(merged)):
                for j in range(i + 1, len(merged)):
                    a = merged[i] if isinstance(merged[i], list) else [merged[i]]
                    b = merged[j] if isinstance(merged[j], list) else [merged[j]]
                    union_ij = a + b
                    # 规划合并后的覆盖路径
                    res = self.planCoverageRoute(union_ij, tasks + [
                        {'route': r, 'obstacles': set()}
                        for r in final_routes if r is not None
                    ])
                    if res is None:
                        continue
                    I, io_list, Q, total_cost, full_path = res
                    # 如果无冲突，则真正合并
                    if canParallelFlush(full_path, tasks + [
                        {'route': r, 'obstacles': set()}
                        for r in final_routes if r is not None
                    ]):
                        # 更新第 i 个簇
                        merged[i] = union_ij
                        final_routes[i] = full_path
                        # 删除第 j 个簇及其路线
                        del merged[j]
                        del final_routes[j]
                        changed = True
                        break
                if changed:
                    break

        # 对剩余每个簇，都要给它生成一次路径
        for idx, clust in enumerate(merged):
            if final_routes[idx] is None:
                clust_list = [clust]
                res = self.planCoverageRoute(clust_list, tasks)
                if res is not None:
                    I, io_list, Q, total_cost, final_routes[idx] = res

        return merged, final_routes

    def planCoverageRoute(self, cluster, tasks):
        """
        给一个簇的点集 cluster，生成一条“覆盖路径”：
        """
        # 障碍合并：所有模块占用 + 历史 global_waste
        forbidden = set().union(*(t['obstacles'] for t in tasks)) | set(self.global_waste)
        for sub_cluster in cluster:
            forbidden = forbidden - set(sub_cluster)
        grid = np.zeros((self.R, self.C), dtype=int)
        res = select_full_route_general(grid, cluster, blocked_cells=forbidden)
        return res

    def planFlushTasks(self, severe_waste_set, tasks, typ):
        """
        1) 连通分量聚类
        2) 为每簇生成覆盖路径 & 判断并行属性
        """
        # 1. 连通分量
        clusters = clusterWaste(severe_waste_set)
        # 2. 按距离合并簇

        clusters, final_routes = self.mergeClusters(clusters, tasks)
        flush_tasks = []
        for idx, clust in enumerate(clusters):
            route = final_routes[idx]
            parallel = canParallelFlush(route, tasks)
            flush_tasks.append({
                'id': f'flush_{self.flush_num}',
                'task_type': 'flush',
                'type': typ,  # typ(0):严重废液处理；typ(1):阻塞废液处理；
                'waste_points': clust,
                'route': route,
                'obstacles': set().union(*(t['obstacles'] for t in tasks)),
                'parallel': parallel
            })
            self.flush_num += 1
        return flush_tasks

    def flushWaste(self, cells):
        # 模拟清洗：将 self.global_waste 中这些 cells 标记为已清洗
        if hasattr(cells, "ndim"):  # numpy
            self.global_waste.difference_update(map(tuple, cells))
        else:
            self.global_waste.difference_update(tuple(p) for p in cells)
        # self.global_waste = [p for p in self.global_waste if p not in cells]

    def getCurFluidConflictWithWaste(self, tasks):
        waste_conflict = []
        for task in tasks:
            for pos in task['route']:
                if pos in self.global_waste:
                    waste_conflict.append(pos)
        return waste_conflict

    def getCurFluidConflict(self, tasks):
        grid = np.zeros((self.R, self.C), dtype=int)

        # 将所有 route 统一转成集合，便于交集
        line_sets = []
        for t in tasks:
            r = t.get("route")
            line_sets.append(set(r) if r else set())

        pairwise = {}
        for i, j in combinations(range(len(line_sets)), 2):
            inter = line_sets[i] & line_sets[j]
            pairwise[(i, j)] = inter
            for (r, c) in inter:
                grid[r, c] = 1

        common = set.intersection(*line_sets) if line_sets else set()
        return pairwise, common, grid

    # 新增：统计冲突点（默认只统计试剂任务）
    def count_fluid_conflicts(self, tasks, only_reagents: bool = True) -> int:
        # 只保留试剂任务（排除 flush 任务）
        if only_reagents:
            tasks = [t for t in tasks if t.get('task_type', 'route') != 'flush']
        # 只保留有路径的任务
        tasks = [t for t in tasks if t.get('route')]

        _, _, grid = self.getCurFluidConflict(tasks)
        return int(np.sum(grid))  # grid 中置 1 的格子数 = 冲突点唯一去重后的个数

    def _route_set(self, t):
        r = t.get('route', [])
        return set(r) if r else set()

    def _obstacles_set(self, t):
        o = t.get('obstacles', set())
        return set(o) if o else set()

    def _conflict(self, a, b):
        """
        两个任务是否冲突：
          1) 路径与路径相交
          2) a 路径与 b 的 obstacles 相交
          3) b 路径与 a 的 obstacles 相交
        """
        ra = self._route_set(a)
        rb = self._route_set(b)
        if not ra or not rb:
            # 没有路径的任务（异常/未规划好），保守视为冲突，避免并行
            return True
        if ra & rb:
            return True
        if ra & self._obstacles_set(b):
            return True
        if rb & self._obstacles_set(a):
            return True
        return False

    def _pack_one_phase(self, ready_items):
        """
        从“已满足依赖”的任务集合里，贪心选出一批两两不冲突的任务，组成一个并行 phase。
        """
        phase = []
        for item in ready_items:
            if all(not self._conflict(item, other) for other in phase):
                phase.append(item)
        return phase

    def build_phases(self, tasks):
        """
        按依赖 + 冲突分组，生成并行 phases。
        输入：
          tasks: 经过 addressWasteForTasks 处理后的本阶段布线任务（已可能带有 _deps）
          self.flushTasks: 本阶段需要执行的 flush 任务（来自 addWasteTasks 和 handleObstacleWaste）
        输出：
          phases: List[List[task_or_flush]]
        规则：
          - 只有 item 的依赖 _deps 全部完成，才允许被调度。
          - 同一 phase 内两两不冲突。
        """
        # 组装调度对象池：布线任务 + flush 任务（有路径者）
        pool = []
        # 布线任务可能有 _deps
        pool.extend(tasks)
        # flush 任务有 parallel/route 等属性；无路径的先跳过
        for ft in self.flushTasks:
            if ft.get('route'):
                pool.append(ft)

        # 去掉完全无路可走的（route为空），避免死循环
        pool = [it for it in pool if self._route_set(it)]

        phases = []
        done_ids = set()
        remaining = list(pool)

        while remaining:
            # 先挑出已满足依赖的 ready 集合
            ready = [it for it in remaining
                     if set(it.get('_deps', set())) <= done_ids]

            if not ready:
                # 存在未满足依赖但也无法推进的情形（可能有坏数据或循环依赖）
                # 为保守起见，强制取一个出来单独执行（或者 raise 更合适）
                solo = remaining[0]
                phases.append([solo])
                done_ids.add(solo['id'])
                remaining = [x for x in remaining if x is not solo]
                continue

            # 在 ready 中打一个最大不冲突并行集
            phase = self._pack_one_phase(ready)
            phases.append(phase)
            done_ids |= {it['id'] for it in phase}
            remaining = [x for x in remaining if x not in phase]

        return phases

    def fluidLoading(self, tasks):
        """
            根据 getCurFluidConflict 生成的 pairwise 冲突，
            把所有 tasks 划分到若干个 phase，使得同一 phase 内互不冲突。
            返回:
              phases: List[List[int]]
                每个子列表是一组可以并行路由的 task 索引
              pairwise, common, grid  供调用者调试或可视化使用
        """
        pairwise, common, grid = self.getCurFluidConflict(tasks)
        n = len(tasks)

        # 构建冲突图：conflict_graph[i] = {j1, j2, …}
        conflict_graph = {i: set() for i in range(n)}
        for (i, j), inter in pairwise.items():
            if inter:  # 只有真正有交集才算冲突
                conflict_graph[i].add(j)
                conflict_graph[j].add(i)

        # 贪心分阶段调度：每次从剩余任务里摘取一个不互冲的最大集合
        unscheduled = set(range(n))
        phases = []
        while unscheduled:
            phase = set()
            # 尝试把尽量多的 unscheduled 任务放到同一 phase
            for i in sorted(unscheduled):
                # i 与当前 phase 中所有任务都不冲突，才加入
                if all((j not in conflict_graph[i]) for j in phase):
                    phase.add(i)
            phases.append(sorted(phase))
            unscheduled -= phase

        return phases, pairwise, common, grid


# ---------- Test Harness ----------
if __name__ == '__main__':
    # Dummy routing model that returns tasks unchanged
    class DummyRoutingModel:
        def predict(self, tasks):
            return {'tasks': tasks}


    R, C = 10, 10
    global_grid = np.zeros((R, C), dtype=int)

    # Initialize WasteGrid and dummy routing model
    waste_grid = WasteGrid(global_grid)
    routing_model = DummyRoutingModel()

    # Pre-existing global waste
    existing_global = {(1, 1), (1, 2), (2, 1)}
    waste_grid.global_waste = set(existing_global)
    for item in existing_global:
        waste_grid.global_grid[item[0], item[1]] = 1

    # Phase waste from previous stage
    phase_waste = {(0, 1), (2, 2), (2, 3), (0, 5), (5, 4)}

    # Example tasks
    tasks = [
        {
            'id': 0,
            'source_area': {(0, 0), (0, 1)},
            'target_area': {(3, 3)},
            'route': [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3)],
            'obstacles': set()
        },
        {
            'id': 1,
            'source_area': {(4, 4)},
            'target_area': {(5, 5)},
            'route': [(4, 4), (5, 4), (5, 5)],
            'obstacles': set()
        }
    ]

    print("Before addWasteTasks:")
    print(" Global waste:", waste_grid.global_waste)
    print(" Tasks:", tasks)

    # 1. addWasteTasks
    waste_grid.addWasteTasks(phase_waste, tasks)
    print("\nAfter addWasteTasks (flushTasks generated):")
    print(" flushTasks:", waste_grid.flushTasks)

    # 2. Initial routing (dummy)
    routing_result = routing_model.predict(tasks=tasks)
    tasks_init = routing_result['tasks']

    # 3. addressWasteForTasks
    tasks_result = waste_grid.addressWasteForTasks(tasks_init)
    print("\nAfter addressWasteForTasks (final tasks):")
    for t in tasks_result:
        print(t)
    for ft in waste_grid.flushTasks:
        print(f"ft:{ft}")
