import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from stable_baselines3 import PPO
from MultiAgentFlatWrapper import MultiAgentFlatWrapper
from MultiAgentGridEnv import MultiAgentGridEnv
import os

# 创建测试环境
module_specs = {
    "op1": {"size": (2, 3), "duration": 2, "dependencies": ["r1", "r2"], "generate": "r9"},
    "op2": {"size": (2, 3), "duration": 3, "dependencies": ["r3", "r4"], "generate": "r10"},
    "op3": {"size": (2, 3), "duration": 4, "dependencies": ["r5", "r6"], "generate": "r11"},
    "op4": {"size": (2, 3), "duration": 2, "dependencies": ["r7", "r8"], "generate": "r12"},
    "op5": {"size": (2, 3), "duration": 2, "dependencies": ["op1", "op2"], "generate": "r13"},
    "op6": {"size": (3, 4), "duration": 2, "dependencies": ["op3", "op4"], "generate": "r14"},
    "op7": {"size": (4, 5), "duration": 1, "dependencies": ["op5", "op6"], "generate": "r15"}
}
reagent_specs = {
    "op1": {"r1": {"cells": 3, "from": "external"}, "r2": {"cells": 3, "from": "external"}},
    "op2": {"r3": {"cells": 2, "from": "external"}, "r4": {"cells": 4, "from": "external"}},
    "op3": {"r5": {"cells": 4, "from": "external"}, "r6": {"cells": 2, "from": "external"}},
    "op4": {"r7": {"cells": 3, "from": "external"}, "r8": {"cells": 3, "from": "external"}},
    "op5": {"r9": {"cells": 2, "from": "op1"}, "r10": {"cells": 4, "from": "op2"}},
    "op6": {"r11": {"cells": 6, "from": "op3"}, "r12": {"cells": 6, "from": "op4"}},
    "op7": {"r13": {"cells": 12, "from": "op5"}, "r14": {"cells": 8, "from": "op6"}}
}
start_point = {
    "r1": (0, 0), "r2": (2, 0), "r3": (0, 4), "r4": (0, 6),
    "r5": (5, 0), "r6": (7, 0), "r7": (9, 0), "r8": (8, 0)
}

# 创建环境实例
raw_env = MultiAgentGridEnv(
    grid_size=(10, 10),
    module_specs=module_specs,
    reagent_specs=reagent_specs,
    start_point=start_point
)

# 使用包装器将多智能体环境转化为平铺形式
env = MultiAgentFlatWrapper(raw_env)

# 加载训练好的 PPO 模型
model = PPO.load("ppo_grid_placement_final")

# 创建输出文件夹
output_folder = "./layout_steps"
os.makedirs(output_folder, exist_ok=True)


# 可视化布局的函数
def plot_step(grid, active_modules, step):
    """
    绘制当前步骤的布局图，显示单元中心的坐标，并带有网格。
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # 设置背景为白色
    ax.set_facecolor('white')

    # 绘制网格背景
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            color = 'white' if grid[r, c] == 0 else 'lightgrey'
            ax.add_patch(Rectangle((c, r), 1, 1, color=color,
                                   edgecolor='black', linewidth=1.5, linestyle='--'))
            ax.text(c + 0.5, r + 0.5, f"({c},{r})", color='gray',
                    ha='center', va='center', fontsize=7, alpha=0.7)

    # 定义颜色映射
    reagent_colors = {
        "r1": "lightblue", "r2": "lightgreen", "r3": "lightcoral", "r4": "gold",
        "r5": "cyan", "r6": "purple", "r7": "pink", "r8": "lime",
        "r9": "orange", "r10": "brown", "r11": "darkgreen", "r12": "navy",
        "r13": "violet", "r14": "olive"
    }

    # 绘制模块和试剂
    for module_id, module in active_modules.items():
        row, col, height, width = module["position"]

        # 模块边界
        rect = Rectangle((col, row), width, height, linewidth=2, edgecolor='none', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        ax.text(col + width / 2, row + height / 2, module_id, color='black', ha='center', va='center', fontsize=12,
                fontweight='bold')

        reagent_distribution = module["reagent_positions"]
        # 显示试剂单元
        print(f"模块 {module_id} 分配情况：")
        for reagent, cells in reagent_distribution.items():
            color = reagent_colors.get(reagent, "lightgrey")
            print(f"  - {reagent}: {cells}，数量: {len(cells)}")
            for (cell_r, cell_c) in cells:
                ax.add_patch(Rectangle((cell_c, cell_r), 1, 1, edgecolor='black',
                                       facecolor=color, linewidth=0.5, linestyle='--'))
                ax.text(cell_c + 0.1, cell_r + 0.1, reagent, color='black',
                        ha='left', va='top', fontsize=9, fontweight='bold')

    # 设置坐标轴与网格
    ax.set_xlim(0, grid.shape[1])
    ax.set_ylim(0, grid.shape[0])
    ax.invert_yaxis()  # y轴反转以匹配数组索引
    ax.set_xticks(np.arange(0.5, grid.shape[1], 1))
    ax.set_yticks(np.arange(0.5, grid.shape[0], 1))
    ax.set_xticklabels(range(grid.shape[1]))
    ax.set_yticklabels(range(grid.shape[0]))
    ax.grid(True, color='gray', linestyle='-', linewidth=0.5)

    # 设置标题
    ax.set_title(f"Placement step: {step}", fontsize=16)

    # 保存图片
    file_path = f"./layout_steps/step_{step:02d}.png"
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close(fig)
    print(f"✅ 步骤 {step} 可视化已保存至 {file_path}")



def reshape_grid(grid):
    """将一维 grid 转换为二维网格"""
    return np.reshape(grid, grid_size)


# 测试过程
obs, _ = env.reset()
done = False
total_reward = 0
step_count = 0
grid_size = (10, 10)
# 进行推理并绘制每一步的布局过程
while not done:
    actions = {}
    obs_length = grid_size[0] * grid_size[1] + 1  # 每个 agent 的观测长度
    actions, _ = model.predict(obs)  # 为每个 agent 选择动作
    obs, reward, done, truncated, info = env.step(actions)
    total_reward += reward
    step_count += 1
    agent_idx = len(env.agent_ids)
    start_idx = (agent_idx - 1) * obs_length
    end_idx = agent_idx * obs_length - 1
    agent_obs = obs[start_idx:end_idx]  # 提取每个 agent 的观测
    grid_2d = reshape_grid(agent_obs)

    # 绘制当前布局步骤
    plot_step(grid_2d, raw_env.active_modules, step_count)

print(f"🎯 测试完成：总奖励 = {total_reward:.2f}")
print(f"🎯 模型推断完成，共 {step_count} 步。")
