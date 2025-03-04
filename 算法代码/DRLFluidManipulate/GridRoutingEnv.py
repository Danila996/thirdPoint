import numpy as np
import gymnasium as gym
from gymnasium import spaces
from abc import ABC


class WiringEnv(gym.Env):
    """
    自定义布线环境
    状态：当前在二维网格上的位置 (row, col)
    动作：0: 上，1: 下，2: 左，3: 右
    奖励：
      - 每走一步扣1分（鼓励短路径）
      - 越界扣 -5 分，且结束回合
      - 撞上障碍扣 -10 分，且结束回合
      - 到达目标奖励 100 分，结束回合
    """

    def __init__(self, rows=10, cols=10, obstacles=None, start=(0, 0), target=(9, 9)):
        super(WiringEnv, self).__init__()
        self.rows = rows
        self.cols = cols
        self.start = start
        self.target = target
        # 障碍列表，每个障碍用 (row, col) 表示
        self.obstacles = obstacles if obstacles is not None else []

        # 定义动作空间：0:上, 1:下, 2:左, 3:右
        self.action_space = spaces.Discrete(4)
        # 定义观察空间：当前位置在网格中的坐标 (row, col)
        self.observation_space = spaces.Box(low=np.array([0, 0]),
                                            high=np.array([rows - 1, cols - 1]),
                                            dtype=np.int32)
        self.reset()

    def reset(self):
        self.state = self.start
        return np.array(self.state, dtype=np.int32)

    def step(self, action):
        row, col = self.state
        if action == 0:  # 向上移动
            next_state = (row - 1, col)
        elif action == 1:  # 向下移动
            next_state = (row + 1, col)
        elif action == 2:  # 向左移动
            next_state = (row, col - 1)
        elif action == 3:  # 向右移动
            next_state = (row, col + 1)
        else:
            next_state = (row, col)

        # 检查是否越界
        if not (0 <= next_state[0] < self.rows and 0 <= next_state[1] < self.cols):
            reward = -5  # 越界惩罚
            done = True
            return np.array(self.state, dtype=np.int32), reward, done, {}

        # 检查是否撞上障碍
        if next_state in self.obstacles:
            reward = -10  # 障碍惩罚
            done = True
            return np.array(self.state, dtype=np.int32), reward, done, {}

        self.state = next_state

        # 每走一步扣1分
        reward = -1
        done = False
        # 如果到达目标，给予高奖励，并结束回合
        if self.state == self.target:
            reward = 100
            done = True

        return np.array(self.state, dtype=np.int32), reward, done, {}

    def render(self, mode='human'):
        grid = [['.' for _ in range(self.cols)] for _ in range(self.rows)]
        # 标记障碍（用 '#' 表示）
        for (r, c) in self.obstacles:
            grid[r][c] = '#'
        # 标记起点和目标
        sr, sc = self.start
        tr, tc = self.target
        grid[sr][sc] = 'S'
        grid[tr][tc] = 'T'
        # 标记当前智能体位置
        ar, ac = self.state
        grid[ar][ac] = 'A'
        for row in grid:
            print(' '.join(row))
        print()


# DRL训练示例（基于Stable-Baselines3中的DQN）
if __name__ == "__main__":
    # 如果还没有安装 stable-baselines3，请使用：pip install stable-baselines3[extra]
    from stable_baselines3 import DQN
    from stable_baselines3.common.env_checker import check_env

    # 定义障碍（可以看作布局中已经放置好的组件或障碍）
    obstacles = [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (4, 3)]
    # 创建环境，示例中起点为 (0,0)，目标为 (7,7)
    env = WiringEnv(rows=10, cols=10, obstacles=obstacles, start=(0, 0), target=(7, 7))

    # 检查环境是否符合Gym标准
    check_env(env, warn=True)

    # 创建DQN模型，使用多层感知机策略
    model = DQN("MlpPolicy", env, verbose=1,
                learning_rate=0.001,
                buffer_size=50000,
                learning_starts=1000,
                batch_size=32,
                gamma=0.99,
                target_update_interval=1000)

    # 开始训练，总步数根据实际情况调整
    model.learn(total_timesteps=10000)

    # 测试训练好的模型
    obs = env.reset()
    env.render()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print("动作：", action, "新状态：", obs, "奖励：", reward)
        env.render()
