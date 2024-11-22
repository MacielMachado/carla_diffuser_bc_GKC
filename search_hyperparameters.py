'''

Seed fixo!
------------------------------------------------------------------------
- N: 20, 50
- Learning Rate: 1e-5,
- Embedding: 64, 128, 256
- Batch Size: 32, 256, 512
- Net Type: Transformer, FC
- Extra Diffusion Steps: 0, 8, 16
- Beta range: (1e-4, 0.01), (1e-4, 0.02), (1e-4, 0.05)
- Beta scheduler: Linear, Cosine, Quadrativ
- Alpha: (Fixed: 0.8, 0.5, 0.3), Exponential, Cosine
- Com modelo pré-treinado e sem modelo pré-treinado
------------------------------------------------------------------------
- Avaliar a distância média percorrida antes de uma infração em 5 rodadas
partindo de um mesmo ponto

'''

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from data_collect import reward_configs, terminal_configs, obs_configs
from learn_bc import learn_bc, set_seed, reset_weights
from rl_birdview_wrapper import RlBirdviewWrapper
from carla_gym.envs import EndlessFixedSpawnEnv
from expert_dataset import ExpertDataset
from carla_gym.envs import EndlessEnv
from agent_policy import AgentPolicy
import numpy as np
import itertools
import json
import gym
import git
import torch as th
import os


# model = MyModel()
# reset_weights(model)  # Reinicializa os pesos para os mesmos valores


spawn_point = {
    'pitch':360.0,
    'roll':0.0,
    'x':338.6842956542969,
    'y':176.00216674804688,
    'yaw':269.9790954589844,
    'z':0.0
}

def env_maker():
    cfg = json.load(open("config.json", "r"))
    env = EndlessFixedSpawnEnv(obs_configs=obs_configs, reward_configs=reward_configs,
                    terminal_configs=terminal_configs, host='localhost', port=2020,
                    seed=2021, no_rendering=True, **env_configs, spawn_point=spawn_point)
    env = RlBirdviewWrapper(env)
    return env


def get_git_commit_hash():
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha


env_configs = {
    'carla_map': 'Town01',
    'weather_group': 'dynamic_1.0',
    'routes_group': ''
}

if __name__ == '__main__':

    # Iteração de Parâmetros
    pretrained_bc_list = [False]
    alpha_list = ['fixed_0-9', 'fixed_1-0', 'cosine']
    batch_size_list = [512, 32]
    embedding_dim_list = [64, 128]
    lrate_list = [1e-5, 1e-4, 1e-3]
    lrate_type = ['fixed']
    beta_scheduler_list = ["linear"]
    n_T_list = [20, 50]
    net_type_list = ["transformer"]
    betas_list = [(1e-4, 0.01), (1e-4, 0.02)]

    # lrate_type = ['cosine', 'fixed']
    # beta_scheduler_list = ["quadratic", "linear"]
    # pretrained_bc_list = [False, True]

    device = 'cuda'

    # Gerando as combinações
    params_product = itertools.product(
        betas_list,
        batch_size_list,
        net_type_list,
        lrate_list,
        embedding_dim_list,
        n_T_list,
        beta_scheduler_list,
        lrate_type,
        alpha_list,
        pretrained_bc_list,
    )
    params_list = list(params_product)

    # params_list = list(np.load("eval_distribution_diffusion/configs.npy", allow_pickle=True))
    set_seed(42)

    for index, item in enumerate(params_list):
        # if index < 4:
        #     continue
        betas = item[0]
        batch_size = item[1]
        net_type = item[2]
        lrate = item[3]
        embedding_dim = item[4]
        n_T = item[5]
        beta_scheduler = item[6]
        lrate_type = item[7]
        alpha = item[8]
        pretrained_bc = item[9]
        n_epoch=500
        device=device
        pretrained_model = 'ckpt_mse_sem_trajetoria_update/bc_ckpt_4_min_eval.pth'

        env = DummyVecEnv([env_maker])
        resume_last_train = False
        observation_space = {}
        observation_space['birdview'] = gym.spaces.Box(low=0, high=255, shape=(3, 192, 192), dtype=np.uint8)
        observation_space['state'] = gym.spaces.Box(low=-10.0, high=30.0, shape=(6,), dtype=np.float32)
        observation_space = gym.spaces.Dict(**observation_space)

        action_space = gym.spaces.Box(low=np.array([0, -1]), high=np.array([1, 1]), dtype=np.float32)

        policy_kwargs = {
            'observation_space': observation_space,
            'action_space': action_space,
            'policy_head_arch': [256, 256],
            'features_extractor_entry_point': 'torch_layers:XtMaCNN',
            'features_extractor_kwargs': {'states_neurons': [256,256]},
            'distribution_entry_point': 'distributions:DiagGaussianDistribution',
            'architecture': 'mse_diffusion',
            'beta_scheduler': beta_scheduler,
            'net_type': net_type,
            'embedding_dim': embedding_dim,
            'betas': betas,
            'n_T': n_T,
        }

        policy = AgentPolicy(**policy_kwargs)
        reset_weights(policy)


        if pretrained_model:
            trained_state_dict = th.load(pretrained_model)

            diff_policy_dict = policy.state_dict()
            chaves_a_reutilizar = [
                'features_extractor', 
                'features_extractor.linear', 
                'features_extractor.state_linear', 
                'policy_head'
            ]

            for key in trained_state_dict['policy_state_dict']:
                if any([key.startswith(chave) for chave in chaves_a_reutilizar]):
                    diff_policy_dict[key] = trained_state_dict['policy_state_dict'][key]

            policy.load_state_dict(diff_policy_dict)

            # for name, param in policy.named_parameters():
            #     if any([name.startswith(chave) for chave in chaves_a_reutilizar]):
            #         param.requires_grad = False

        policy.to(device)
        gail_train_loader = th.utils.data.DataLoader(
            ExpertDataset(
                'gail_experts',
                n_routes=8,
                n_eps=1,
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        
        gail_val_loader = th.utils.data.DataLoader(
            ExpertDataset(
                'gail_experts',
                n_routes=2,
                n_eps=1,
                route_start=8
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        
        path = f'eval_distribution_diffusion/version_{index}/'
        os.makedirs(path, exist_ok=True)

        data = policy_kwargs.copy()
        data['lrate'] = lrate
        data['lrate_type'] = lrate_type
        data['alpha_schedule'] = alpha
        data['pre_trained'] = pretrained_bc
        data.pop('observation_space')
        data.pop('action_space')

        with open(f'{path}data.json', 'w') as f:
            json.dump(data, f, separators=(',', ':'), indent=4)


        config={
            'policy_head_arch': [256, 256],
            'features_extractor_entry_point': 'torch_layers:XtMaCNN',
            'features_extractor_kwargs': {'states_neurons': [256,256]},
            'distribution_entry_point': 'distributions:BetaDistribution',
            'architecture': 'distribution_diffusion',
            'beta_scheduler': beta_scheduler,
            'net_type': net_type,
            'embedding_dim': embedding_dim,
            'betas': betas,
            'n_T': n_T,
            'lrate': lrate,
            'lrate_type': lrate_type,
            'alpha': alpha,
            'batch_size': batch_size,
            'commit_hash': get_git_commit_hash()
            }

        try:
            learn_bc(policy, device, gail_train_loader, gail_val_loader,
                    env, resume_last_train=False, lrate=lrate,
                    lrate_type=lrate_type, alpha_schedule=alpha, path=path,
                    config=config, name=f'version_{index}')
            env.close()
            del env
        except Exception as exception:
            print("---------------------------------------------------")
            print(f"The process couldn't be trained due to ")
            print(f'{exception}')
            print("---------------------------------------------------")
            continue