from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

# 检查自定义环境
env = GridPlacementEnv(
    grid_size=(10, 10),
    module_specs={
        "op1": {"size": (2, 4), "duration": 5, "dependencies": ["r1", "r2", "r3"]},
        "op2": {"size": (2, 2), "duration": 5, "dependencies": ["r1", "r3"]},
        "op3": {"size": (3, 3), "duration": 5, "dependencies": ["op1", "op2"]},
        "op4": {"size": (0, 0), "duration": 5, "dependencies": ["op3"]}
    },
    reagent_specs={
        "op1": {
            "r1": {"cells": 2, "from": "external"},
            "r2": {"cells": 2, "from": "external"},
            "r3": {"cells": 2, "from": "external"}
        },
        "op2": {
            "r1": {"cells": 2, "from": "external"},
            "r3": {"cells": 2, "from": "external"}
        },
        "op3": {
            "r4": {"cells": 3, "from": "op1"},
            "r5": {"cells": 3, "from": "op2"},
            "r6": {"cells": 4, "from": "op3"}
        }
    },
    start_point={
        "r1": (0, 0),
        "r2": (5, 0),
        "r3": (0, 5)
    }
)

# 检查环境有效性
check_env(env)

# 包装环境，监控训练过程
env = Monitor(env)

# 配置保存检查点
checkpoint_callback = CheckpointCallback(
    save_freq=1000, save_path="./models/", name_prefix="ppo_grid"
)

# 创建模型
model = PPO("MlpPolicy", env, verbose=1)

# 训练模型
model.learn(total_timesteps=100000, callback=checkpoint_callback)

# 保存模型
model.save("ppo_grid_placement")
