import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class CustomExtractor_routing(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=3200):
        super(CustomExtractor_routing, self).__init__(observation_space, features_dim)

        # 观测空间现在包含 "obstacles", "goal_positions", "goals_positions", "cur_agent_positions", "agent_positions"
        self.final_layers = nn.Sequential(
            # Block 1: 输入通道数由 5 而非原来的 4
            nn.Conv2d(in_channels=5, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            # Block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            # Block 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            nn.Flatten()
        )

        # 通过一次前向传播自动计算展平后的特征维度，并更新 _features_dim
        with th.no_grad():
            sample = observation_space.sample()
            # 获取示例观测：所有的网格 (H, W)
            H, W = sample["obstacles"].shape
            # 将各个观测转换为 (1, 1, H, W) 的张量
            grid_sample   = th.as_tensor(sample["obstacles"]).unsqueeze(0).unsqueeze(0).float()
            goal_sample   = th.as_tensor(sample["goal_positions"]).unsqueeze(0).unsqueeze(0).float()
            goals_sample  = th.as_tensor(sample["goals_positions"]).unsqueeze(0).unsqueeze(0).float()
            cur_agent_sample = th.as_tensor(sample["cur_agent_positions"]).unsqueeze(0).unsqueeze(0).float()
            agent_sample  = th.as_tensor(sample["agent_positions"]).unsqueeze(0).unsqueeze(0).float()

            # 拼接 5 个通道，得到 (1, 5, H, W)
            sample_tensor = th.cat([grid_sample, goal_sample, goals_sample, cur_agent_sample, agent_sample], dim=1)
            n_flatten = self.final_layers(sample_tensor).shape[1]
            self._features_dim = n_flatten

    def forward(self, observations: dict) -> th.Tensor:
        # 从输入字典中提取各个观测，并转换为 (batch, 1, H, W)
        obstacles = observations["obstacles"].float().unsqueeze(1)
        goal_positions = observations["goal_positions"].float().unsqueeze(1)
        goals_positions = observations["goals_positions"].float().unsqueeze(1)
        cur_agent_positions = observations["cur_agent_positions"].float().unsqueeze(1)
        agent_positions = observations["agent_positions"].float().unsqueeze(1)

        # 拼接成 (batch, 5, H, W)
        x = th.cat([obstacles, goal_positions, goals_positions, cur_agent_positions, agent_positions], dim=1)

        # 通过卷积层提取特征
        features = self.final_layers(x)
        return features
