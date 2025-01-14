from PlacementEnv import PlacementEnv
from stable_baselines3 import PPO
from stable_baselines3 import SAC
import os
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import argparse
from CustomCombinedExtractor_placement import CustomExtractor_placement
from stable_baselines3.common.vec_env import SubprocVecEnv,DummyVecEnv
import pickle
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from utlis import mask_fn
from datetime import date
# 创建解析器
parser = argparse.ArgumentParser(description='Process some integers.')

# 添加期望解析的命令行参数
parser.add_argument('--reward_overbound', type=float, help='Reward for overbound')
parser.add_argument('--reward_fail', type=float, help='Reward for fail')
parser.add_argument('--reward_finish', type=float, help='Reward for finish')
parser.add_argument('--weight_ASPR',type=float,help='weight of ASPR')
parser.add_argument('--weight_surdist',type=float,help='weight of placing in center area')
#reinforcement params
parser.add_argument('--learning_rate', type=float, help='Learning rate')
parser.add_argument('--n_epochs', type=int, help='Number of epochs')
parser.add_argument('--n_steps', type=int, help='Number of steps')
parser.add_argument('--batch_size', type=int, help='Batch size')
parser.add_argument('--total_timesteps', type=int, help='Total timesteps')
parser.add_argument('--model_name', type=str, help='Model name')

# 解析命令行参数
args = parser.parse_args()

model_dir = f"models_{date.today()}"
log_dir = f"placement_ppo_log_{date.today()}"
param_dir = f"params_{date.today()}"
if not os.path.exists(os.path.join(os.getcwd(), model_dir)):
    os.mkdir(os.path.join(os.getcwd(), model_dir))
if not os.path.exists(os.path.join(os.getcwd(),log_dir)):
    os.mkdir(os.path.join(os.getcwd(), log_dir))
if not os.path.exists(os.path.join(os.getcwd(), param_dir)):
    os.mkdir(os.path.join(os.getcwd(), param_dir))



#-------------对超参数进行赋值----------------------------------
learning_rate = args.learning_rate
n_epochs = args.n_epochs
stats_window_size = 1
total_timesteps = args.total_timesteps
model_name = args.model_name
#---------------------------------------------------------------

# ----------------------------------------------------上面初始化文件------------------------------------------------
policy_kwargs = dict(
    features_extractor_class=CustomExtractor_placement,
)
new_log = configure(log_dir, ['stdout', 'csv', 'log'])
# 定义参数组合
# Bioassay_list = [
#                  'placement_PS-2.pkl',
#                  'placement_syn30.pkl','placement_syn50.pkl']
Bioassay_list = ['placement_IVD-1.pkl', 'placement_IVD-2.pkl', 'placement_PS-1.pkl',
                 'placement_PS-2.pkl','placement_syn10.pkl','placement_syn20.pkl','placement_PCR.pkl',
                 'placement_syn30.pkl','placement_syn40.pkl','placement_syn50.pkl']
vecenv_params =[]
reward_param = {'reward_overbound': args.reward_overbound,
                'reward_finish': args.reward_finish, 'reward_fail': args.reward_fail, 'weight_ASPR': args.weight_ASPR,
                'weight_surdist': args.weight_surdist}
for bioassay in Bioassay_list:
    with open(bioassay,"rb") as file:
        bioassay_env_param = pickle.load(file)
        bioassay_env_param['env_params'] = reward_param
        tmp_ModuleAttribute = bioassay_env_param['ModuleAttribute']
        tmp_ModuleAttribute.pop(0)
        tmp_ModuleAttribute.pop(-1)
        bioassay_env_param['ModuleAttribute'] = tmp_ModuleAttribute
        vecenv_params.append(bioassay_env_param)

# def mask_fn(env: PlacementEnv) -> np.ndarray:
#     return env.get_mask()
# 创建多环境实例的函数
def make_env(param):
    def _init():
        env = PlacementEnv(**param)  # 使用参数创建环境实例
        env = ActionMasker(env, mask_fn)
        env = Monitor(env,log_dir)
        return env
    return _init

if __name__ == "__main__":
    store_param_path = os.path.join(os.getcwd(), param_dir)
    if not os.path.exists(store_param_path):
        os.mkdir(store_param_path)

    with open(os.path.join(store_param_path,f'param_{model_name}.pkl'),"wb") as file:
        pickle.dump(args,file)

    # 创建多个环境实例
    num_envs = 10
    envs = [make_env(vecenv_params[0]) for i in range(num_envs)]
    env = SubprocVecEnv(envs)
    timesteps_schedule = {
        'placement_PCR.pkl':args.total_timesteps,
        'placement_syn10.pkl': args.total_timesteps,
        'placement_syn20.pkl': args.total_timesteps,
        'placement_syn30.pkl': 2*args.total_timesteps,
        'placement_syn40.pkl': 2*args.total_timesteps,
        'placement_syn50.pkl':2*args.total_timesteps,
        'placement_IVD-1.pkl': args.total_timesteps,
        'placement_IVD-2.pkl': args.total_timesteps,
        'placement_PS-1.pkl': args.total_timesteps,
        'placement_PS-2.pkl': 2*args.total_timesteps,
    }
    #pattern4------------------------------------------------------------
    # # 创建多环境实例
    # # vec_env = DummyVecEnv(envs)
    # model = MaskablePPO(MaskableActorCriticPolicy, env, learning_rate=learning_rate, n_epochs=n_epochs,device='cuda:0',
    #                     n_steps=args.n_steps,batch_size=args.batch_size,
    #             stats_window_size=stats_window_size, tensorboard_log=log_dir, policy_kwargs=policy_kwargs, verbose=1)
    # #=======================================================
    #
    # remain_timesteps = total_timesteps
    # ptr_env = 0
    # while remain_timesteps > 0:
    #     ptr_env = ptr_env% len(Bioassay_list)
    #     env_new = [make_env(vecenv_params[ptr_env]) for i in range(num_envs)]
    #     env_wrap = SubprocVecEnv(env_new)
    #     model.set_env(env_wrap)
    #     model.learn(total_timesteps=timesteps_schedule[Bioassay_list[ptr_env]], log_interval=1, tb_log_name=model_name,reset_num_timesteps=False)
    #     remain_timesteps -= timesteps_schedule[Bioassay_list[ptr_env]]
    #     ptr_env += 1
    #
    # model.save(f"{model_dir}/{model_name}")
    #----------------------------------------------------------------------
    #training for every benchmark
    for benchamrk in Bioassay_list:
        model = MaskablePPO(MaskableActorCriticPolicy, env, learning_rate=learning_rate, n_steps=args.n_steps,
                            batch_size=args.batch_size,
                            n_epochs=n_epochs, stats_window_size=stats_window_size, tensorboard_log=log_dir,
                            device='cuda:0',
                            policy_kwargs=policy_kwargs, verbose=1)
        env_new = [make_env(vecenv_params[Bioassay_list.index(benchamrk)]) for i in range(num_envs)]
        env_wrap = SubprocVecEnv(env_new)
        model.set_env(env_wrap)
        model.learn(total_timesteps=timesteps_schedule[benchamrk], log_interval=1,
                    tb_log_name=f"{model_name}_{benchamrk.replace('.pkl','')}",reset_num_timesteps=False)
        model.save(f"models_{date.today()}/{model_name}_{benchamrk.replace('.pkl','')}")







