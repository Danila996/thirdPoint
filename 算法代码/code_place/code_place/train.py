
import subprocess


#base
param_base =     {'reward_overbound': -0.1, 'reward_fail': -1,'reward_finish': 0, 'weight_ASPR': 0.4,'weight_surdist':0,
    #reinforcement learning param
     'learning_rate': 0.0008, 'n_epochs': 3, 'total_timesteps': 400000,'n_steps':512,'batch_size':128,
    #model param
     'model_name': 'base_model'}
     #the range of the target hyperparameters
params_dict = {
    "reward_overbound": [-0.1],
    "reward_fail": [-1],
    "reward_finish": [1],
    "weight_ASPR": [0.2,0.3],
    "weight_surdist": [0.2,0.4],
    #reinforcement learning param
    "learning_rate": [0.0001,0.0005],
    "n_epochs": [2,3],
    "n_steps": [1024],
    "batch_size": [64,256],
    "total_timesteps": [600000],
}

target_params =['batch_size','total_timesteps','weight_ASPR']
generated_params = [param_base.copy()]

for param in target_params:
    for value in params_dict[param]:
        new_param = param_base.copy()
        new_param[param] = value
        if type(value) == int:
            model_name = f'placement_{param}_{value}'
        else:
            model_name = f'placement_{param}_{str(value).replace(".","_")}'
        new_param['model_name'] = model_name
        generated_params.append(new_param.copy())



for params in generated_params:
    # 构建命令行参数
    cmd = [
        'python', 'main.py',  # 假设主文件名为main.py
        f'--weight_ASPR={params["weight_ASPR"]}',
        f'--weight_surdist={params["weight_surdist"]}',
        f'--reward_overbound={params["reward_overbound"]}',
        f'--reward_fail={params["reward_fail"]}',
        f'--reward_finish={params["reward_finish"]}',
        # reinforcement learning param
        f'--learning_rate={params["learning_rate"]}',
        f'--n_epochs={params["n_epochs"]}',
        f'--total_timesteps={params["total_timesteps"]}',
        f'--model_name={params["model_name"]}',
        f'--batch_size={params["batch_size"]}',
        f'--n_steps={params["n_steps"]}'
        # model param
    ]

    # 执行命令行
    subprocess.run(cmd)




# params_list = [
#     {'weightDist': 0.8, 'weightWP': 0.2,'scale':2, 'reward_overbound': -0.1, 'reward_finsubmission': 0.1,
#      'reward_fail': -1, 'reward_finish': 1, 'reward_goback': -0.05,
#      #reinforcement learning param
#      'learning_rate': 0.0003, 'n_epochs': 10,'n_steps':4096,'batch_size':128,'total_timesteps': 1200000,
#      #model param
#      'model_name': 'routing_lr_a'},
#     {'weightDist': 0.8, 'weightWP': 0.2, 'scale': 2, 'reward_overbound': -0.1, 'reward_finsubmission': 0.1,
#      'reward_fail': -1, 'reward_finish': 1, 'reward_goback': -0.05,
#      # reinforcement learning param
#      'learning_rate': 0.0005, 'n_epochs': 10, 'n_steps': 4096, 'batch_size': 128, 'total_timesteps': 1200000,
#      # model param
#      'model_name': 'routing_lr_b'},
#     {'weightDist': 0.8, 'weightWP': 0.2, 'scale': 2, 'reward_overbound': -0.1, 'reward_finsubmission': 0.1,
#      'reward_fail': -1, 'reward_finish': 1, 'reward_goback': -0.05,
#      # reinforcement learning param
#      'learning_rate': 0.0007, 'n_epochs': 10, 'n_steps': 4096, 'batch_size': 128, 'total_timesteps': 1200000,
#      # model param
#      'model_name': 'routing_lr_c'}]
# # 循环遍历参数组合
