import os
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from env_creator import env_creator
from ray.tune.registry import register_env
from ParallelGridRoutingEnv import ParallelGridRoutingEnv


def main():
    # 初始化 Ray
    ray.init()
    env_name = "my_parallel_env"
    register_env(env_name, lambda config: env_creator(config))
    # 先创建一个环境实例，用来获取 observation_space 和 action_space，
    # 以便配置多智能体的 policies
    # test_env = env_creator()
    # test_env.reset()
    #
    # # 我们假设所有智能体共享同一个 policy（如果你需要每个智能体单独策略，也可以配置多策略）
    # agent_ids = test_env.agents
    # sample_agent_id = agent_ids[0]
    # sample_obs_space = test_env.observation_spaces[sample_agent_id]
    # sample_act_space = test_env.action_spaces[sample_agent_id]

    # 配置多智能体
    # 这里我们只定义了一个 "shared_policy" 给所有 agent 使用
    # policies = {
    #     "shared_policy": (
    #         None,  # 让RLlib自动选择默认的模型
    #         sample_obs_space,
    #         sample_act_space,
    #         {}
    #     )
    # }

    # def policy_mapping_fn(agent_id, *args, **kwargs):
    #     # 如果你希望所有agent都用同一个策略，可以直接返回 "shared_policy"
    #     # 如果你想区分不同agent，可在此根据 agent_id 返回不同策略名
    #     return "shared_policy"

    # 构建 PPO 算法配置
    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        # .env_runners(num_env_runners=2)   # 可以根据硬件设置 worker 数量
        .environment(env=env_name, clip_actions=True)
        .env_runners(num_env_runners=2)
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.99,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            num_epochs=10,
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 5000000 if not os.environ.get("CI") else 50000},
        checkpoint_freq=10,
        storage_path="~/ray_results/" + env_name,
        config=config.to_dict(),
    )


if __name__ == "__main__":
    main()
