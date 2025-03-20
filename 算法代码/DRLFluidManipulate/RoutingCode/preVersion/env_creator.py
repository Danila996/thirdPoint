from pettingzoo.utils import wrappers, parallel_to_aec
from ParallelGridRoutingEnv import ParallelGridRoutingEnv  # 这是你贴出来的环境类所在的文件
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv


def env_creator(config=None):
    # 如果需要传入自定义参数，可以在 config 里传
    base_grid = None  # 在这里也可以指定你的base_grid
    env = ParallelGridRoutingEnv(base_grid=base_grid)
    # 对 ParallelEnv 进行 PettingZoo 推荐的包装器
    # 如 OrderEnforcingWrapper, etc.
    # aec_env = parallel_to_aec(env)
    return ParallelPettingZooEnv(env)
