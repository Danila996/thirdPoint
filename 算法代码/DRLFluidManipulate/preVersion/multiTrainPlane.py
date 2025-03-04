# -------------------------------
# 训练代码：使用 Stable Baselines3 PPO
# -------------------------------
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

# 定义训练数据（与前面相同）
from MultiAgentFlatWrapper import MultiAgentFlatWrapper
from MultiAgentGridEnv import MultiAgentGridEnv

module_specs = {
    "op1": {"size": (2, 3), "duration": 4, "dependencies": ["r1", "r2"], "generate": "r9"},
    "op2": {"size": (2, 3), "duration": 4, "dependencies": ["r3", "r4"], "generate": "r10"},
    "op3": {"size": (2, 3), "duration": 4, "dependencies": ["r5", "r6"], "generate": "r11"},
    "op4": {"size": (2, 3), "duration": 4, "dependencies": ["r7", "r8"], "generate": "r12"},
    "op5": {"size": (2, 3), "duration": 2, "dependencies": ["op1", "op2"], "generate": "r13"},
    "op6": {"size": (3, 4), "duration": 2, "dependencies": ["op3", "op4"], "generate": "r14"},
    "op7": {"size": (4, 5), "duration": 1, "dependencies": ["op5", "op6"], "generate": "r15"}
}
reagent_specs = {
    "op1": {
        "r1": {"cells": 3, "from": "external"},
        "r2": {"cells": 3, "from": "external"}
    },
    "op2": {
        "r3": {"cells": 2, "from": "external"},
        "r4": {"cells": 4, "from": "external"}
    },
    "op3": {
        "r5": {"cells": 4, "from": "external"},
        "r6": {"cells": 2, "from": "external"}
    },
    "op4": {
        "r7": {"cells": 3, "from": "external"},
        "r8": {"cells": 3, "from": "external"}
    },
    "op5": {
        "r9": {"cells": 2, "from": "op1"},
        "r10": {"cells": 4, "from": "op2"}
    },
    "op6": {
        "r11": {"cells": 6, "from": "op3"},
        "r12": {"cells": 6, "from": "op4"}
    },
    "op7": {
        "r13": {"cells": 12, "from": "op5"},
        "r14": {"cells": 8, "from": "op6"}
    }
}
start_point = {
    "r1": (0, 0),
    "r2": (2, 0),
    "r3": (0, 4),
    "r4": (0, 6),
    "r5": (5, 0),
    "r6": (7, 0),
    "r7": (9, 0),
    "r8": (8, 0)
}

# 创建原始环境实例
raw_env = MultiAgentGridEnv(
    grid_size=(10, 10),
    module_specs=module_specs,
    reagent_specs=reagent_specs,
    start_point=start_point
)
# 用包装器转换为平铺版环境
env = MultiAgentFlatWrapper(raw_env)
env = Monitor(env)

# 检查环境是否符合要求
check_env(env)

# 配置检查点回调，每 1000 步保存一次模型
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./models/', name_prefix='ppo_grid_placement')

tensorboard_log_dir = "./ppo_tensorboard_multi/"

# 指定使用 GPU 1
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# 使用 PPO 模型，因平铺后的观测为单一 numpy 数组，故使用 MlpPolicy
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log_dir, device=device)

# 开始训练，例如 150,000 个时间步
model.learn(total_timesteps=200000, callback=checkpoint_callback)

# 保存最终模型
model.save("ppo_grid_placement_final")
