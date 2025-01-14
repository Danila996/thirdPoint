

"""
    这个版本的网络模型是串行的，同个网络卷三个矩阵
"""
import torch
from torch.nn import Linear, ReLU, Dropout, Flatten,Transformer
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomExtractor_placement(BaseFeaturesExtractor):

    def __init__(self, observation_space,features_dim=12544):
        super(CustomExtractor_placement, self).__init__(observation_space, features_dim)
        self.Din = observation_space.shape[0]
        self.Hin = observation_space.shape[1]
        self.Win = observation_space.shape[2]
        #kenel固定3,stride固定1
        Dout = (self.Din - 3) / 1 + 1
        Hout = (self.Hin - 3) / 1 + 1
        Wout = (self.Win - 3) / 1 + 1
        pad_d = ((Dout - 1) * 1 + 3 - self.Din) / 2
        pad_h = ((Hout - 1) * 1 + 3 - self.Hin) / 2
        pad_w = ((Wout - 1) * 1 + 3 - self.Win) / 2
        self.final_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            # Transformer(d_model=32,nhead=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
            # nn.Linear(in_features=4096, out_features=256),
            # nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # observations = torch.unsqueeze(observations)
        # tgt = observations.clone()
        # for layer in self.final_layers:
        #     if isinstance(layer, Transformer):
        #         observations = layer(src=observations, tgt=tgt)
        #     else:
        #         observations = layer(observations)
        return self.final_layers(observations)




