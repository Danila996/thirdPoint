import os
from typing import Dict, List, Tuple
from stable_baselines3 import PPO
from .MultiAgentGridEnv import MultiAgentGridEnv
from .MultiAgentFlatWrapper import MultiAgentFlatWrapper


class LayoutModel:
    def __init__(self, model_path: str):
        # 加载 PPO 训练好的布局模型
        self.model = PPO.load(model_path)

    @staticmethod
    def load(path: str) -> 'LayoutModel':
        return LayoutModel(path)

    def predict(
            self,
            module_specs: Dict[str, Dict],
            reagent_specs: Dict[str, Dict],
            start_point: Dict[str, Tuple[int, int]],
            level: Dict[str, int]
    ) -> Dict[str, Tuple[int, int, int, int]]:
        # 构建多智能体布局环境
        grid_size = (8, 8)  # 可根据需求传入
        raw_env = MultiAgentGridEnv(
            grid_size=grid_size,
            module_specs=module_specs,
            reagent_specs=reagent_specs,
            start_point=start_point,
            level=level
        )
        env = MultiAgentFlatWrapper(raw_env)

        obs, _ = env.reset()
        done = False
        while not done:
            actions, _ = self.model.predict(obs)
            obs, _, done, _, _ = env.step(actions)

        # 收集最终模块位置
        positions = raw_env.active_modules
        return positions
