"""Uses Stable-Baselines3 to train agents in the Connect Four environment using invalid action masking.

For information about invalid action masking in PettingZoo, see https://pettingzoo.farama.org/api/aec/#action-masking
For more information about invalid action masking in SB3, see https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html

Author: Elliot (https://github.com/elliottower)
"""
import gymnasium
from gymnasium import spaces
import pettingzoo.utils


class SB3ActionMaskWrapper(pettingzoo.utils.BaseWrapper, gymnasium.Env):
    """Wrapper to allow PettingZoo environments to be used with SB3 illegal action masking."""

    def reset(self, seed=None, options=None):
        """Gymnasium-like reset function which assigns obs/action spaces to be the same for each agent.

        This is required as SB3 is designed for single-agent RL and doesn't expect obs/action spaces to be functions
        """
        super().reset(seed, options)
        # Strip the action mask out from the observation space

        obs = super().observation_space(self.possible_agents[0])
        obs_without_mask = spaces.Dict({k: v for k, v in obs.spaces.items() if k != "action_mask"})
        self.observation_space = obs_without_mask
        self.action_space = super().action_space(self.possible_agents[0])
        # Return initial observation, info (PettingZoo AEC envs do not by default)
        return self.observe(self.agent_selection), {}

    def step(self, action):
        """Gymnasium-like step function, returning observation, reward, termination, truncation, info."""
        super().step(action)
        # 如果没有下一个 agent，则直接返回
        # print(f"wraaper:{self.agent_selection}")
        if self.agent_selection is None:
            return
        return super().last()

    def observe(self, agent):
        """"返回不包含action_mask的原始观察值。"""
        obs = super().observe(agent)
        # 返回一个新的字典，剥离掉"action_mask"
        return {"obstacles": obs["obstacles"], "goal_positions": obs["goal_positions"],
                "goals_positions": obs["goals_positions"], "cur_agent_positions": obs["cur_agent_positions"],
                "agent_positions": obs["agent_positions"]}

    def action_mask(self):
        """Separate function used in order to access the action mask."""
        obs = super().observe(self.agent_selection)
        # print(obs["action_mask"], self.agent_selection)
        # self.render()
        return obs["action_mask"]


