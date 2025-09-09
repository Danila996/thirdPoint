import shutil

import pulp
from typing import Dict, Set, Tuple, List
import matplotlib.pyplot as plt
# from pulp import LpStatus, PULP_CBC_CMD, GUROBI_CMD
import itertools, collections
from pulp import (
    LpProblem, LpMinimize, LpVariable, lpSum,
    LpStatus, value, GUROBI_CMD
)
from gurobipy import Model, GRB, quicksum


def assign_reagents_fast_ilp(
        module_cells: Set[Tuple[int, int]],
        reserved_overlap: Dict[str, Set[Tuple[int, int]]],
        reagent_specs: Dict[str, int],
        start_pos: Dict[str, Tuple[int, int]],
        num_variants: int = 3,
        lambda_box: float = 1.0,
        K: int = 8
) -> List[Dict[str, List[Tuple[int, int]]]]:
    reagents = sorted(reagent_specs, key=reagent_specs.get)
    # 1) 全部 overlap
    global_reserved = set().union(*reserved_overlap.values())
    # 2) 预筛 avail：对每个 r 取 K 最近点的并集
    pre_avails = set()
    for r in reagents:
        r0, c0 = start_pos[r]
        dists = [((abs(r0 - ri) + abs(c0 - ci)), (ri, ci))
                 for (ri, ci) in module_cells - global_reserved]
        pre_avails |= {pt for _, pt in sorted(dists)[:K]}
    avail = sorted(pre_avails)
    N = len(avail)
    total_new = sum(reagent_specs[r] - len(reserved_overlap.get(r, ()))
                    for r in reagents)

    # 3) 预compute cost 和行列列表
    cost = {(r, i): abs(avail[i][0] - start_pos[r][0]) + abs(avail[i][1] - start_pos[r][1])
            for r in reagents for i in range(N)}
    rows = [avail[i][0] for i in range(N)]
    cols = [avail[i][1] for i in range(N)]
    M_row = max(rows) - min(rows) + 1
    M_col = max(cols) - min(cols) + 1

    solutions = []
    banned = []

    for v in range(num_variants):
        prob = pulp.LpProblem(f"fast_var_{v}", pulp.LpMinimize)
        # x 变量
        x = {r: pulp.LpVariable.dicts(f"x_{r}", range(N), cat="Binary")
             for r in reagents}
        # 包围盒变量
        min_r = {r: pulp.LpVariable(f"min_r_{r}", lowBound=0, upBound=max(rows), cat="Integer")
                 for r in reagents}
        max_r = {r: pulp.LpVariable(f"max_r_{r}", lowBound=0, upBound=max(rows), cat="Integer")
                 for r in reagents}
        min_c = {r: pulp.LpVariable(f"min_c_{r}", lowBound=0, upBound=max(cols), cat="Integer")
                 for r in reagents}
        max_c = {r: pulp.LpVariable(f"max_c_{r}", lowBound=0, upBound=max(cols), cat="Integer")
                 for r in reagents}

        # 目标：起点距离 + 包围盒周长
        obj = pulp.lpSum(cost[(r, i)] * x[r][i] for r in reagents for i in range(N))
        obj += lambda_box * pulp.lpSum((max_r[r] - min_r[r]) + (max_c[r] - min_c[r])
                                       for r in reagents)
        prob += obj

        # ========== 约束 ==========
        # a) 约束1：每个 reagent 恰好分配 k_new 个新单元
        for r in reagents:
            k_new = reagent_specs[r] - len(reserved_overlap.get(r, ()))
            # prob += pulp.lpSum(x[r][i] for i in range(N)) == k_new
            # 2：slack
            prob += pulp.lpSum(x[r][i] for i in range(N)) >= k_new
            prob += pulp.lpSum(x[r][i] for i in range(N)) <= k_new

        # b) 约束2：同一格点最多给一个 reagent
        for i in range(N):
            prob += pulp.lpSum(x[r][i] for r in reagents) <= 1

        # c) 约束3：至少一个新单元与 overlap 相邻（保证能“挂上” overlap 区）
        neigh4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for r in reagents:
            k_new = reagent_specs[r] - len(reserved_overlap.get(r, ()))
            if k_new > 0 and reserved_overlap.get(r):
                adj = [i for i, (ri, ci) in enumerate(avail)
                       if any((ri + dr, ci + dc) in reserved_overlap[r] for dr, dc in neigh4)]
                prob += pulp.lpSum(x[r][i] for i in adj) >= 1

        # d) 包围盒线性化：对每个选中点 i，min≤coord≤max
        for r in reagents:
            for i, (ri, ci) in enumerate(avail):
                prob += min_r[r] <= ri + M_row * (1 - x[r][i])
                prob += max_r[r] >= ri - M_row * (1 - x[r][i])
                prob += min_c[r] <= ci + M_col * (1 - x[r][i])
                prob += max_c[r] >= ci - M_col * (1 - x[r][i])

        # e) Ban‐the‐best（全局+ per‐reagent）
        for sel in banned:
            prob += pulp.lpSum(x[r][i] for (r, i) in sel) <= total_new - 1
            for r in reagents:
                idxs = [i for (rr, i) in sel if rr == r]
                if idxs:
                    prob += pulp.lpSum(x[r][i] for i in idxs) <= len(idxs) - 1
        # print(f"reagents:{reagents}")
        for a in range(len(reagents) - 1):
            r_a = reagents[a]
            r_b = reagents[a + 1]
            # ∑ i * x[r_a][i] ≤ ∑ i * x[r_b][i]
            prob += (
                            pulp.lpSum(i * x[r_a][i] for i in range(N))
                            <=
                            pulp.lpSum(i * x[r_b][i] for i in range(N))
                    ), f"sym_break_{r_a}_le_{r_b}"
        # 求解（限时+多线程）
        # print("before into prob.solve")
        # print("→ cbc 可执行在：", shutil.which("cbc"))
        solver = GUROBI_CMD(msg=False,
                            timeLimit=1,  # 限时10秒
                            threads=8,  # 多线程
                            options=[('MIPGap', 0.1)])  # 允许1% gap
        # solver = PULP_CBC_CMD(
        #     msg=True,
        #     timeLimit=2,
        #     threads=8,
        #     presolve=True,
        #     options=["maxSolutions 1"]
        # )
        status = prob.solve(solver)
        if LpStatus[status] == "Infeasible":
            raise RuntimeError("真无解")
        # print("after into prob.solve")
        # 提取解
        sel_vars = []
        mapping = {}
        for r in reagents:
            pts = list(reserved_overlap.get(r, []))
            for i in range(N):
                val = pulp.value(x[r][i])
                if (val or 0.0) > 0.5:
                    pts.append(avail[i])
                    sel_vars.append((r, i))
            mapping[r] = pts
        solutions.append(mapping)
        banned.append(sel_vars)

    return solutions


# —— 使用示例 —— #
if __name__ == "__main__":
    # 1) 定义模块格点
    module_cells = {(r, c) for r in range(10) for c in range(10)}
    print(len(module_cells))
    # 2) 假设两个 reagent 的重叠单元
    reserved_overlap = {
        # "A": {(1, 1), (1, 2)},
        # "B": {(3, 3)},
    }
    # 3) 总需求（含重叠）
    reagent_specs = {"A": 10, "B": 10}
    # 4) 起始位置（用于计算 cost）
    start_pos = {"A": (0, 0), "B": (5, 5)}

    sols = assign_reagents_fast_ilp(
        module_cells, reserved_overlap,
        reagent_specs, start_pos,
        num_variants=3
    )
    schemes = sols
    for idx, sol in enumerate(sols, 1):
        print(f"\n--- 方案 {idx} ---")
        for r, cells in sol.items():
            print(f"{r}: {cells}")

    markers = {'A': 'o', 'B': 's'}
    colors = {'A': 'tab:blue', 'B': 'tab:orange'}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, scheme, idx in zip(axes, schemes, range(1, 4)):
        for reagent, points in scheme.items():
            xs = [c for r, c in points]
            ys = [r for r, c in points]
            ax.scatter(xs, ys,
                       marker=markers[reagent],
                       color=colors[reagent],
                       label=reagent,
                       alpha=0.7)
        ax.set_title(f"方案 {idx}")
        ax.set_xlabel("列 (Column)")
        ax.set_ylabel("行 (Row)")
        ax.set_aspect('equal', 'box')
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()


def brute_assign_fill_single_solution(
        module_cells: Set[Tuple[int, int]],
        reserved_overlap: Dict[str, Set[Tuple[int, int]]],
        reagent_specs: Dict[str, int],
        start_pos: Dict[str, Tuple[float, float]],
        lambda_box: float = 1.0,
) -> List[Dict[str, List[Tuple[int, int]]]]:
    reagents = sorted(reagent_specs, key=reagent_specs.get)
    global_reserved = set().union(*reserved_overlap.values())
    avail = sorted(module_cells - global_reserved)
    N = len(avail)
    k_new = {r: reagent_specs[r] - len(reserved_overlap.get(r, ())) for r in reagents}
    assert sum(k_new.values()) == N

    # 四邻
    neigh4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # 哈密顿路径检测
    def has_hamiltonian(cells: Set[Tuple[int, int]]) -> bool:
        """在 cells 上看是否有一条遍历所有点的简单路径（哈密顿路径）"""
        if not cells:
            return True
        # 构建邻接列表
        adj = {p: [] for p in cells}
        for (x, y) in cells:
            for dx, dy in neigh4:
                q = (x + dx, y + dy)
                if q in cells:
                    adj[(x, y)].append(q)

        L = len(cells)
        seen = set()

        def dfs(u, visited_count):
            if visited_count == L:
                return True
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    if dfs(v, visited_count + 1):
                        return True
                    seen.remove(v)
            return False

        # 尝试每个点作为起点
        for start in cells:
            seen.clear()
            seen.add(start)
            if dfs(start, 1):
                return True
        return False

    # 一旦找到解就直接返回
    def rec(idx: int, remaining: Set[int],
            current: Dict[str, List[int]], score_acc: float):
        if idx == len(reagents):
            # 构造完整解
            sol = {}
            for r in reagents:
                pts = set(reserved_overlap.get(r, ()))
                pts |= {avail[i] for i in current[r]}
                sol[r] = list(pts)
            return sol

        r = reagents[idx]
        k = k_new[r]
        ov = reserved_overlap.get(r, set())

        for comb in itertools.combinations(remaining, k):
            pts_new = {avail[i] for i in comb}
            # ① 挂靠 overlap
            if k > 0 and ov:
                if not any(
                        (x + dx, y + dy) in ov
                        for (x, y) in pts_new
                        for dx, dy in neigh4
                ):
                    continue
            # ② 哈密顿连通：必须 overlap∪新点 上存在哈密顿路径
            full_pts = pts_new | ov
            if not has_hamiltonian(full_pts):
                continue

            # ③ 得分：距离+包围盒
            dist = sum(
                abs(x - start_pos[r][0]) + abs(y - start_pos[r][1])
                for (x, y) in pts_new
            )
            xs, ys = zip(*pts_new) if pts_new else ([0], [0])
            bbox = (max(xs) - min(xs)) + (max(ys) - min(ys))
            sc = dist + lambda_box * bbox

            current[r] = list(comb)
            result = rec(idx + 1, remaining - set(comb), current, score_acc + sc)
            if result:
                return result
            del current[r]

        return None

    # 启动递归
    solution = rec(0, set(range(N)), {}, 0.0)

    # 如果没有找到解，返回空
    return [solution] if solution else [{}]


def brute_assign_fill_single_solution2(
        module_cells: Set[Tuple[int, int]],
        reserved_overlap: Dict[str, Set[Tuple[int, int]]],
        reagent_specs: Dict[str, int],
        start_pos: Dict[str, Tuple[float, float]],
        lambda_box: float = 1.0,
) -> List[Dict[str, List[Tuple[int, int]]]]:
    reagents = sorted(reagent_specs, key=reagent_specs.get)
    global_reserved = set().union(*reserved_overlap.values())
    avail = sorted(module_cells - global_reserved, key=lambda p: (p[0], p[1]))
    N = len(avail)
    k_new = {r: reagent_specs[r] - len(reserved_overlap.get(r, ())) for r in reagents}
    assert sum(k_new.values()) == N

    neigh4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def has_hamiltonian(cells: Set[Tuple[int, int]]) -> bool:
        """在 cells 上看是否有一条遍历所有点的简单路径（哈密顿路径）"""
        if not cells:
            return True
        # 构建邻接列表
        adj = {p: [] for p in cells}
        for (x, y) in cells:
            for dx, dy in neigh4:
                q = (x + dx, y + dy)
                if q in cells:
                    adj[(x, y)].append(q)

        L = len(cells)
        seen = set()

        def dfs(u, visited_count):
            if visited_count == L:
                return True
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    if dfs(v, visited_count + 1):
                        return True
                    seen.remove(v)
            return False

        # 尝试每个点作为起点
        for start in cells:
            seen.clear()
            seen.add(start)
            if dfs(start, 1):
                return True
        return False

    def rec(idx: int, remaining: Set[int]) -> Dict[str, List[Tuple[int, int]]] or None:
        if idx == len(reagents):
            # 构造解
            sol = {}
            for r in reagents:
                pts = set(reserved_overlap.get(r, ()))
                pts |= {avail[i] for i in current[r]}
                sol[r] = list(pts)
            return sol

        r = reagents[idx]
        k = k_new[r]
        ov = reserved_overlap.get(r, set())

        # 1) 先把所有 comb 排序
        scored_combs = []
        for comb in itertools.combinations(remaining, k):
            pts_new = {avail[i] for i in comb}
            # ① overlap 挂靠
            if k > 0 and ov:
                if not any((x + dx, y + dy) in ov for (x, y) in pts_new for dx, dy in neigh4):
                    continue
            # 计算局部得分（距离+bbox），但**暂不**检验连通
            dist = sum(abs(x - start_pos[r][0]) + abs(y - start_pos[r][1]) for (x, y) in pts_new)
            if pts_new:
                xs, ys = zip(*pts_new)
                bbox = (max(xs) - min(xs)) + (max(ys) - min(ys))
            else:
                bbox = 0
            sc = dist + lambda_box * bbox
            scored_combs.append((sc, comb))

        # 2) 按得分从小到大遍历
        for _, comb in sorted(scored_combs, key=lambda x: x[0]):
            pts_new = {avail[i] for i in comb}
            # 再检验连通
            full = pts_new | ov
            if not has_hamiltonian(full):
                continue

            current[r] = list(comb)
            sol = rec(idx + 1, remaining - set(comb))
            if sol:
                return sol
            del current[r]

        return None

    current: Dict[str, List[int]] = {}
    solution = rec(0, set(range(N)))
    return [solution] if solution else [{}]


def assign_reagents_fast_ilp_with_warmstart(
        module_cells: Set[Tuple[int, int]],
        reserved_overlap: Dict[str, Set[Tuple[int, int]]],
        reagent_specs: Dict[str, int],
        start_pos: Dict[str, Tuple[int, int]],
        brute_solution: Dict[str, List[Tuple[int, int]]],
        num_variants: int = 3,
        lambda_box: float = 1.0,
        K: int = 8
) -> List[Dict[str, List[Tuple[int, int]]]]:
    # 1) reagents 顺序
    reagents = sorted(reagent_specs, key=lambda r: reagent_specs[r])
    # 2) 所有 overlap 单元
    global_reserved = set().union(*reserved_overlap.values())

    # 3) 预筛 avail：先把 brute_solution 里的“新分配”点加进
    pre_avails: Set[Tuple[int, int]] = set()
    for r, pts in brute_solution.items():
        ov = reserved_overlap.get(r, set())
        pre_avails |= {p for p in pts if p not in ov}

    # 4) 再加每个 r 最近的 K 个点
    for r in reagents:
        r0, c0 = start_pos[r]
        dists = [
            (abs(r0 - ri) + abs(c0 - ci), (ri, ci))
            for ri, ci in (module_cells - global_reserved)
        ]
        dists.sort(key=lambda x: x[0])
        pre_avails |= {pt for _, pt in dists[:K]}

    # 5) 最终可分配新单元
    avail = sorted(pre_avails)
    N = len(avail)
    coord2idx = {avail[i]: i for i in range(N)}

    # 6) 建模型
    m = Model("assign_reagents")
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.TimeLimit = 2  # 5 秒钟找到可行解
    m.Params.MIPGap = 0.1  # 1% gap

    # 7) 决策变量 x[r][i]
    x = {
        r: m.addVars(N, vtype=GRB.BINARY, name=f"x_{r}")
        for r in reagents
    }

    # 8) 包围盒变量
    max_row = max((cell[0] for cell in avail), default=0)
    max_col = max((cell[1] for cell in avail), default=0)
    min_r = {r: m.addVar(lb=0, ub=max_row, vtype=GRB.INTEGER, name=f"min_r_{r}") for r in reagents}
    max_r = {r: m.addVar(lb=0, ub=max_row, vtype=GRB.INTEGER, name=f"max_r_{r}") for r in reagents}
    min_c = {r: m.addVar(lb=0, ub=max_col, vtype=GRB.INTEGER, name=f"min_c_{r}") for r in reagents}
    max_c = {r: m.addVar(lb=0, ub=max_col, vtype=GRB.INTEGER, name=f"max_c_{r}") for r in reagents}

    m.update()

    # 9) 目标：距离 + 包围盒周长
    cost = {}
    for r in reagents:
        for i, (ri, ci) in enumerate(avail):
            cost[(r, i)] = abs(ri - start_pos[r][0]) + abs(ci - start_pos[r][1])
    obj = quicksum(cost[(r, i)] * x[r][i] for r in reagents for i in range(N))
    obj += lambda_box * quicksum((max_r[r] - min_r[r]) + (max_c[r] - min_c[r])
                                 for r in reagents)
    m.setObjective(obj, GRB.MINIMIZE)

    # 10) 约束
    # 10.1) 每个 reagent 精确分配 k_new 个新单元
    for r in reagents:
        k_new = reagent_specs[r] - len(reserved_overlap.get(r, ()))
        m.addConstr(quicksum(x[r][i] for i in range(N)) == k_new, name=f"count_{r}")

    # 10.2) 同一格点至多 1 个 reagent
    for i in range(N):
        m.addConstr(quicksum(x[r][i] for r in reagents) <= 1, name=f"mutex_{i}")

    # 10.3) overlap 邻接（至少一块挂上去）
    neigh4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for r in reagents:
        ov = reserved_overlap.get(r, set())
        if ov:
            adj_idxs = []
            for i, (ri, ci) in enumerate(avail):
                if any((ri + dr, ci + dc) in ov for dr, dc in neigh4):
                    adj_idxs.append(i)
            if adj_idxs:
                m.addConstr(quicksum(x[r][i] for i in adj_idxs) >= 1, name=f"adj_{r}")

    # 10.4) 准备连通约束用的邻居列表 & 与 overlap 邻接表
    neighbor_idxs = {i: [] for i in range(N)}
    for i, (ri, ci) in enumerate(avail):
        for j, (rj, cj) in enumerate(avail):
            if abs(ri - rj) + abs(ci - cj) == 1:
                neighbor_idxs[i].append(j)
    adj_to_overlap: Dict[str, Set[int]] = {}
    for r in reagents:
        s = set()
        ov = reserved_overlap.get(r, set())
        for i, (ri, ci) in enumerate(avail):
            if any((ri + dr, ci + dc) in ov for dr, dc in neigh4):
                s.add(i)
        adj_to_overlap[r] = s

    # 10.5) 局部连通：每个点要么连到 overlap，要么连到另一个选点
    for r in reagents:
        for i in range(N):
            coeff = 1 if i in adj_to_overlap[r] else 0
            m.addConstr(
                x[r][i] <= quicksum(x[r][j] for j in neighbor_idxs[i]) + coeff,
                name=f"conn_{r}_{i}"
            )

    # 10.6) 包围盒线性化
    M_row = max_row - (min(cell[0] for cell in avail) if avail else 0) + 1
    M_col = max_col - (min(cell[1] for cell in avail) if avail else 0) + 1
    for r in reagents:
        for i, (ri, ci) in enumerate(avail):
            # 当 x[r,i]=1 时，4 条约束锁死 min_r[r]=ri, max_r[r]=ri
            # m.addConstr(min_r[r] <= ri + M_row * (1 - x[r][i]))
            # m.addConstr(min_r[r] >= ri - M_row * (1 - x[r][i]))
            # m.addConstr(max_r[r] >= ri - M_row * (1 - x[r][i]))
            # m.addConstr(max_r[r] <= ri + M_row * (1 - x[r][i]))
            # # 同理对列
            # m.addConstr(min_c[r] <= ci + M_col * (1 - x[r][i]))
            # m.addConstr(min_c[r] >= ci - M_col * (1 - x[r][i]))
            # m.addConstr(max_c[r] >= ci - M_col * (1 - x[r][i]))
            # m.addConstr(max_c[r] <= ci + M_col * (1 - x[r][i]))
            m.addConstr(min_r[r] <= ri + M_row * (1 - x[r][i]), name=f"minr_{r}_{i}")
            m.addConstr(max_r[r] >= ri - M_row * (1 - x[r][i]), name=f"maxr_{r}_{i}")
            m.addConstr(min_c[r] <= ci + M_col * (1 - x[r][i]), name=f"minc_{r}_{i}")
            m.addConstr(max_c[r] >= ci - M_col * (1 - x[r][i]), name=f"maxc_{r}_{i}")

    # 10.7) 对称性破除（可选）
    for a in range(len(reagents) - 1):
        ra, rb = reagents[a], reagents[a + 1]
        lhs = quicksum(i * x[ra][i] for i in range(N))
        rhs = quicksum(i * x[rb][i] for i in range(N))
        m.addConstr(lhs <= rhs, name=f"sym_{ra}_le_{rb}")

    # 11) Warm‐start 注入
    for r, pts in brute_solution.items():
        ov = reserved_overlap.get(r, set())
        for p in pts:
            if p not in ov and p in coord2idx:
                i = coord2idx[p]
                m.getVarByName(f"x_{r}[{i}]").start = 1

    # 12) 求解
    m.optimize()

    # 13) 提取解
    solutions: List[Dict[str, List[Tuple[int, int]]]] = []
    if m.Status == GRB.INFEASIBLE:
        return solutions
    # if m.SolCount == 0:
    #     raise RuntimeError("ILP 未找到可行解")
    seen = set()
    for _ in range(num_variants):
        sol: Dict[str, List[Tuple[int, int]]] = {}
        for r in reagents:
            pts = list(reserved_overlap.get(r, []))
            for i in range(N):
                if x[r][i].X > 0.5:
                    pts.append(avail[i])
            sol[r] = pts
        # 构造一个 hashable 的 key，用于去重
        key = tuple((r, tuple(sorted(sol[r]))) for r in reagents)
        if key in seen:
            continue
        seen.add(key)
        solutions.append(sol)

        if len(solutions) >= num_variants:
            break
        # break

    return solutions


def brute_assign_fill(
        module_cells: Set[Tuple[int, int]],
        reserved_overlap: Dict[str, Set[Tuple[int, int]]],
        reagent_specs: Dict[str, int],
        start_pos: Dict[str, Tuple[float, float]],
        lambda_box: float = 1.0,
        num_variants: int = 3
) -> List[Dict[str, List[Tuple[int, int]]]]:
    reagents = sorted(reagent_specs)
    global_reserved = set().union(*reserved_overlap.values())
    avail = sorted(module_cells - global_reserved)
    N = len(avail)
    k_new = {r: reagent_specs[r] - len(reserved_overlap.get(r, ())) for r in reagents}
    assert sum(k_new.values()) == N

    # 四邻
    neigh4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # 哈密顿路径检测
    def has_hamiltonian(cells: Set[Tuple[int, int]]) -> bool:
        """在 cells 上看是否有一条遍历所有点的简单路径（哈密顿路径）"""
        if not cells:
            return True
        # 构建邻接列表
        adj = {p: [] for p in cells}
        for (x, y) in cells:
            for dx, dy in neigh4:
                q = (x + dx, y + dy)
                if q in cells:
                    adj[(x, y)].append(q)

        L = len(cells)
        seen = set()

        def dfs(u, visited_count):
            if visited_count == L:
                return True
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    if dfs(v, visited_count + 1):
                        return True
                    seen.remove(v)
            return False

        # 尝试每个点作为起点
        for start in cells:
            seen.clear()
            seen.add(start)
            if dfs(start, 1):
                return True
        return False

    scored: List[Tuple[float, Dict[str, List[Tuple[int, int]]]]] = []

    def rec(idx: int, remaining: Set[int],
            current: Dict[str, List[int]],
            score_acc: float):
        if idx == len(reagents):
            # 构造完整解
            sol = {}
            for r in reagents:
                pts = set(reserved_overlap.get(r, ()))
                pts |= {avail[i] for i in current[r]}
                sol[r] = list(pts)
            scored.append((score_acc, sol))
            return

        r = reagents[idx]
        k = k_new[r]
        ov = reserved_overlap.get(r, set())

        for comb in itertools.combinations(remaining, k):
            pts_new = {avail[i] for i in comb}
            # ① 挂靠 overlap
            if k > 0 and ov:
                if not any(
                        (x + dx, y + dy) in ov
                        for (x, y) in pts_new
                        for dx, dy in neigh4
                ):
                    continue
            # ② 哈密顿连通：必须 overlap∪新点 上存在哈密顿路径
            full_pts = pts_new | ov
            if not has_hamiltonian(full_pts):
                continue

            # ③ 得分：距离+包围盒
            dist = sum(
                abs(x - start_pos[r][0]) + abs(y - start_pos[r][1])
                for (x, y) in pts_new
            )
            xs, ys = zip(*pts_new) if pts_new else ([0], [0])
            bbox = (max(xs) - min(xs)) + (max(ys) - min(ys))
            sc = dist + lambda_box * bbox

            current[r] = list(comb)
            rec(idx + 1, remaining - set(comb), current, score_acc + sc)
            del current[r]

    # 启动递归
    rec(0, set(range(N)), {}, 0.0)
    # 选最优
    scored.sort(key=lambda x: x[0])
    return [sol for _, sol in scored[:num_variants]]


def brute_assign_fill2(
        module_cells: Set[Tuple[int, int]],
        reserved_overlap: Dict[str, Set[Tuple[int, int]]],
        reagent_specs: Dict[str, int],
        start_pos: Dict[str, Tuple[float, float]],
        lambda_box: float = 1.0,
        num_variants: int = 3,
        topM: int = 50
) -> List[Dict[str, List[Tuple[int, int]]]]:
    # 四邻方向
    neigh4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # 哈密顿路径检测
    def has_hamiltonian(cells: Set[Tuple[int, int]]) -> bool:
        if not cells:
            return True
        # 构建邻接表
        adj = {p: [] for p in cells}
        for x, y in cells:
            for dx, dy in neigh4:
                q = (x + dx, y + dy)
                if q in cells:
                    adj[(x, y)].append(q)
        L = len(cells)
        seen = set()

        def dfs(u, cnt):
            if cnt == L:
                return True
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    if dfs(v, cnt + 1):
                        return True
                    seen.remove(v)
            return False

        for start in cells:
            seen.clear()
            seen.add(start)
            if dfs(start, 1):
                return True
        return False

    # 1) 准备 avail 和 k_new
    reagents = sorted(reagent_specs)
    global_overlap = set().union(*reserved_overlap.values())
    avail = sorted(module_cells - global_overlap)
    N = len(avail)
    k_new = {
        r: reagent_specs[r] - len(reserved_overlap.get(r, set()))
        for r in reagents
    }
    assert sum(k_new.values()) == N, "pure-fill 要求 ∑k_new == |avail|"

    # 2) 为每个试剂预筛候选块
    blocks_by_r: Dict[str, List[Tuple[int, float, List[int]]]] = {}
    for r in sorted(reagents, key=lambda r: -k_new[r]):
        ov = reserved_overlap.get(r, set())
        k = k_new[r]
        blks = []
        for comb in itertools.combinations(range(N), k):
            pts = [avail[i] for i in comb]
            # 挂靠过滤（仅当 k>0 且有 overlap）
            if k > 0 and ov:
                if not any(
                        (x + dx, y + dy) in ov
                        for (x, y) in pts
                        for dx, dy in neigh4
                ):
                    continue
            # 哈密顿连通过滤
            full = set(pts) | ov
            if not has_hamiltonian(full):
                continue
            # 计算 score
            dist = sum(abs(x - start_pos[r][0]) + abs(y - start_pos[r][1]) for x, y in pts)
            xs, ys = zip(*pts) if pts else ([0], [0])
            bbox = (max(xs) - min(xs)) + (max(ys) - min(ys))
            score = dist + lambda_box * bbox
            # mask + 记录
            mask = 0
            for i in comb:
                mask |= 1 << i
            blks.append((mask, score, list(comb)))
        # 按 score 排序并保留 topM
        blks.sort(key=lambda x: x[1])
        blocks_by_r[r] = blks[:topM]

    # 3) 跨试剂递归组合
    solutions: List[Tuple[float, Dict[str, List[Tuple[int, int]]]]] = []

    def dfs(idx: int, used_mask: int, score_acc: float, assign: Dict[str, int]):
        if idx == len(reagents):
            # 构造解
            sol: Dict[str, List[Tuple[int, int]]] = {}
            for r, bi in assign.items():
                mask, s, inds = blocks_by_r[r][bi]
                pts = [avail[i] for i in inds]
                sol[r] = list(reserved_overlap.get(r, set())) + pts
            solutions.append((score_acc, sol))
            return
        r = reagents[idx]
        for bi, (mask, s, inds) in enumerate(blocks_by_r[r]):
            if mask & used_mask:
                continue
            dfs(idx + 1, used_mask | mask, score_acc + s, {**assign, r: bi})

    dfs(0, 0, 0.0, {})

    # 4) 取最优 num_variants
    solutions.sort(key=lambda x: x[0])
    return [sol for _, sol in solutions[:num_variants]]
