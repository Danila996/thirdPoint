from gymnasium import ObservationWrapper, spaces


class ActionMaskObservationWrapper(ObservationWrapper):
    """
    该包装器将合法动作掩码添加到观测中，使得返回的观测是一个字典，
    包含两个字段：
      "observation": 原始观测（例如网格状态）
      "action_mask": 当前环境下各任务合法动作的掩码，形状为 (num_tasks, 4)
    """

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        # 定义新的 observation_space 为字典类型，包含原始观测和合法动作掩码
        self.observation_space = spaces.Dict({
            "observation": env.observation_space,
            "action_mask": spaces.Box(low=0, high=1, shape=(env.num_tasks, 4), dtype=bool)
        })

    def observation(self, observation):
        # 每次返回观测时，同时计算合法动作掩码
        action_mask = self.env.get_action_masks()
        return {"observation": observation, "action_mask": action_mask}
