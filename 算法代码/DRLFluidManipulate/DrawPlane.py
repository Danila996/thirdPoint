import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import networkx as nx

# 随机生成500个引脚点
np.random.seed(42)  # 固定随机种子以保持结果一致
num_points = 2000
points = np.random.randint(0, 10000, size=(num_points, 2))

# 使用 Delaunay 三角剖分构建图的边
tri = Delaunay(points)
edges = set()
for simplex in tri.simplices:
    for i in range(3):
        for j in range(i + 1, 3):
            edges.add(tuple(sorted((simplex[i], simplex[j]))))

# 创建图
G = nx.Graph()
for edge in edges:
    p1, p2 = points[edge[0]], points[edge[1]]
    dist = np.linalg.norm(p1 - p2)
    G.add_edge(edge[0], edge[1], weight=dist)

# 计算最小生成树（Minimum Spanning Tree, MST）
mst = nx.minimum_spanning_tree(G)

# 绘制图像
plt.figure(figsize=(10, 6))
# 绘制 MST 边
for edge in mst.edges:
    p1, p2 = points[edge[0]], points[edge[1]]
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=0.5)

# 绘制引脚点
plt.scatter(points[:, 0], points[:, 1], c='#1F77B4', s=10)

# 设置图像格式
plt.xlim(-500, 10500)
plt.ylim(-500, 10500)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()
