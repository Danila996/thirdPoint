import glob
import os
from typing import Dict, List, Tuple
from sb3_contrib import MaskablePPO
from RoutingCode.ParallelGridRoutingEnv import env as routing_env_fn
from RoutingCode.SB3ActionMaskWrapper import SB3ActionMaskWrapper
# from RoutingCode.CustomCombinedExtractor import CustomExtractor_routing
import sys
import RoutingCode.CustomCombinedExtractor as _cce_mod
from RoutingCode.astar_expert import expert_action

sys.modules['CustomCombinedExtractor'] = _cce_mod
sys.modules['__main__.CustomCombinedExtractor'] = _cce_mod


class RoutingModel:
    def __init__(self, model_path: str):
        # 加载 MaskablePPO 训练好的布线模型
        self.model = MaskablePPO.load(model_path)

    @staticmethod
    def load(path: str) -> 'RoutingModel':
        # 自动寻找最新的策略文件
        if os.path.isdir(path):
            files = glob.glob(os.path.join(path, '*.zip'))
            model_file = max(files, key=os.path.getctime)
        else:
            model_file = path
        return RoutingModel(model_file)

    def predict(
            self,
            tasks: List[Dict],
    ) -> Dict:
        # 初始化布线环境，注入 tasks 与端口信息
        env = routing_env_fn(render_mode="human", case=tasks)
        env = SB3ActionMaskWrapper(env)
        print("各任务信息及端口位置：")
        env.reset()
        step_count = 0
        print("开始执行各布线任务：")
        for agent in env.agent_iter():
            if agent is None:
                break
            observation, reward, termination, truncation, info = env.last()
            mask = env.get_action_masks(env.tasks[env.agent_name_mapping[agent]])
            # action = self.model.predict(observation, action_masks=mask)[0].item()
            actions = expert_action(env, agent)
            if actions is None:
                action = 0
            else:
                action = actions[0][0]
            env.step(action)
            if action is not None:
                print_info = ""
                direction = action
                if direction == 0:
                    print_info = "上"
                elif direction == 1:
                    print_info = "下"
                elif direction == 2:
                    print_info = "左"
                elif direction == 3:
                    print_info = "右"
                print(f"掩码信息(上，下，左，右): {mask}")
                task = env.tasks[int(agent[6:])]
                cur_task = task["cur_task"]
                seg = task['current_segment']
                print(
                    f"{chr(ord('A') + cur_task - 1) + str(cur_task)} 选择动作:{direction}({print_info}), 获得奖励: {reward}, 是否结束: {termination}"
                    f", 是否截断: {truncation}, 当前阶段: {seg}"
                    f", 已覆盖单元:{env.tasks[int(agent[6:])][f'seg{seg}']['covered']}")
            step_count += 1
            if len(env.agents) == 0:
                break
            isLast = True if env.agent_selection == env.agents[-1] else False
            if isLast:
                env.render()
            # 记录路径
            # pos = env.agent_pos[agent]
            # wiring_paths.setdefault(agent, []).append(pos)

        # 组织输出
        total_length = 0
        wiring_paths = {}
        for task in env.tasks:
            path = task["route"]
            wiring_paths[task["id"]] = path
            total_length += len(path)
        return {'paths': wiring_paths, 'total_wire_length': total_length, 'tasks': tasks}
