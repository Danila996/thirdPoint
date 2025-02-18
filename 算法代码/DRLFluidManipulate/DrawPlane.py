from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

# 检查自定义环境
from GridPlacementEnv import GridPlacementEnv

# 定义一个测试用例
# module_specs = {
#     "op1": {"size": (2, 3), "duration": 4, "dependencies": ["r1", "r2"]},
#     "op2": {"size": (2, 3), "duration": 4, "dependencies": ["r3", "r4"]},
#     "op3": {"size": (2, 3), "duration": 4, "dependencies": ["r5", "r6"]},
#     "op4": {"size": (2, 3), "duration": 4, "dependencies": ["r7", "r8"]},
#     "op5": {"size": (2, 3), "duration": 6, "dependencies": ["op1", "op2"]},  # 容量6
#     "op6": {"size": (3, 4), "duration": 6, "dependencies": ["op3", "op4"]},
#     "op7": {"size": (4, 5), "duration": 7, "dependencies": ["op5", "op6"]},
# }
# reagent_specs = {
#     "op1": {
#         "r1": {"cells": 6, "from": "external"},
#         "r2": {"cells": 6, "from": "external"}
#     },
#     "op2": {
#         "r3": {"cells": 6, "from": "external"},
#         "r4": {"cells": 6, "from": "external"}
#     },
#     "op3": {
#         "r5": {"cells": 6, "from": "external"},
#         "r6": {"cells": 6, "from": "external"}
#     },
#     "op4": {
#         "r7": {"cells": 6, "from": "external"},
#         "r8": {"cells": 6, "from": "external"}
#     },
#     "op5": {
#         "r9": {"cells": 6, "from": "op1"},   # op1 输出6，op5容量6，只接收2
#         "r10": {"cells": 6, "from": "op2"}  # op2 输出6，op5容量6，只接收4
#     },
#     "op6": {
#         "r11": {"cells": 6, "from": "op3"},
#         "r12": {"cells": 6, "from": "op4"}
#     },
#     "op7": {
#         "r13": {"cells": 6, "from": "op5"},
#         "r14": {"cells": 6, "from": "op6"}
#     }
# }
# start_point = {
#     "r1": (0, 0),
#     "r2": (2, 0),
#     "r3": (0, 2),
#     "r4": (2, 2),
#     "r5": (4, 0),
#     "r6": (6, 0),
#     "r7": (4, 2),
#     "r8": (6, 2)
# }
module_specs = {
    "op1": {"size": (2, 3), "duration": 4, "dependencies": ["r1", "r2"]},
    "op2": {"size": (2, 3), "duration": 4, "dependencies": ["r3", "r4"]},
    "op3": {"size": (2, 3), "duration": 4, "dependencies": ["r5", "r6"]},
    "op4": {"size": (2, 3), "duration": 4, "dependencies": ["r7", "r8"]},
    "op5": {"size": (3, 4), "duration": 6, "dependencies": ["op1", "op2"]},  # 容量 12
    "op6": {"size": (3, 4), "duration": 6, "dependencies": ["op3", "op4"]},
    "op7": {"size": (4, 5), "duration": 7, "dependencies": ["op5", "op6"]},
}
reagent_specs = {
    # M1 - M4：外部输入，占满各自设备容量 (2x3=6)
    "op1": {
        "r1": {"cells": 3, "from": "external"},
        "r2": {"cells": 3, "from": "external"}
    },
    "op2": {
        "r3": {"cells": 3, "from": "external"},
        "r4": {"cells": 3, "from": "external"}
    },
    "op3": {
        "r5": {"cells": 3, "from": "external"},
        "r6": {"cells": 3, "from": "external"}
    },
    "op4": {
        "r7": {"cells": 3, "from": "external"},
        "r8": {"cells": 3, "from": "external"}
    },

    # M5 和 M6：容量 12，试剂占满
    "op5": {
        "r9": {"cells": 6, "from": "op1"},
        "r10": {"cells": 6, "from": "op2"}
    },
    "op6": {
        "r11": {"cells": 6, "from": "op3"},
        "r12": {"cells": 6, "from": "op4"}
    },

    # M7：容量 20（4x5），试剂占满
    "op7": {
        "r13": {"cells": 10, "from": "op5"},
        "r14": {"cells": 10, "from": "op6"}
    }
}
start_point = {
    "r1": (0, 0),
    "r2": (2, 0),
    "r3": (0, 2),
    "r4": (2, 2),
    "r5": (4, 0),
    "r6": (6, 0),
    "r7": (4, 2),
    "r8": (6, 2)
}

env = GridPlacementEnv(
    grid_size=(10, 10),
    module_specs=module_specs,
    reagent_specs=reagent_specs,
    start_point=start_point
)

# 检查环境有效性
check_env(env)

# 包装环境，监控训练过程
env = Monitor(env)

# 配置保存检查点
checkpoint_callback = CheckpointCallback(
    save_freq=1000, save_path="./models/", name_prefix="ppo_grid"
)

# 设置 TensorBoard 日志路径
tensorboard_log_dir = "./ppo_tensorboard3/"
# **修改此处：使用 MultiInputPolicy**
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log=tensorboard_log_dir
)

# 训练模型
model.learn(total_timesteps=150000, callback=checkpoint_callback)

# 保存模型
model.save("ppo_grid_placement")
