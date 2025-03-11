import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomExtractor_routing(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=3200):
        super(CustomExtractor_routing, self).__init__(observation_space, features_dim)

        # 假设我们将 "grid" 和 "agent_positions" 拼接成2个通道
        self.final_layers = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 通过一个前向传播来计算展平后的特征数量，并更新 features_dim
        with th.no_grad():
            sample = observation_space.sample()
            # 提取观测数据，并转换为 tensor，注意这里 unsqueeze 添加 batch 和 channel 维度
            grid_sample = th.as_tensor(sample["grid"]).unsqueeze(0).unsqueeze(0).float()           # 形状 (1,1,H,W)
            agent_sample = th.as_tensor(sample["agent_positions"]).unsqueeze(0).unsqueeze(0).float()  # 形状 (1,1,H,W)
            sample_tensor = th.cat([grid_sample, agent_sample], dim=1)  # 拼接后形状 (1,2,H,W)
            n_flatten = self.final_layers(sample_tensor).shape[1]
            self._features_dim = n_flatten

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # 从字典中提取 "grid" 和 "agent_positions" 数据
        grid = observations["grid"]             # 形状: (batch, H, W)
        agent_positions = observations["agent_positions"]  # 形状: (batch, H, W)
        # 给每个增加一个 channel 维度，变成 (batch, 1, H, W)
        grid = grid.unsqueeze(1).float()
        agent_positions = agent_positions.unsqueeze(1).float()
        # 在 channel 维度拼接，得到 (batch, 2, H, W)
        x = th.cat([grid, agent_positions], dim=1)
        return self.final_layers(x)
