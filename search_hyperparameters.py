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


env_configs = {
    'carla_map': 'Town01',
    'weather_group': 'dynamic_1.0',
    'routes_group': ''
}

if __name__ == '__main__':

    # Iteração de Parâmetros
    pretrained_bc_list = [True, False]
    lrate_type = ['cosine', 'fixed']
    alpha_list = ['cosine', 'fixed_08', 'fixed_03', 'exponential']
    beta_scheduler_list = ["quadratic", "linear"]
    betas_list = [(1e-4, 0.01), (1e-4, 0.02), (1e-4, 0.05)]
    n_T_list = [20, 50]
    lrate_list = [1e-5, 5e-4, 1e-3]
    net_type_list = ["transformer", "fc"]
    batch_size_list = [1024, 512, 32]
    embedding_dim_list = [64, 128]

    device = 'cuda'

    # Gerando as combinações
    params_product = itertools.product(
        pretrained_bc_list,
        lrate_type,
        alpha_list,
        beta_scheduler_list,
        betas_list,
        n_T_list,
        lrate_list,
        net_type_list,
        batch_size_list,
        embedding_dim_list,
    )
    params_list = list(params_product)
    set_seed(42)

    for index, item in enumerate(params_list):
        pretrained_bc = item[0]
        lrate_type = item[1]
        alpha = item[2]
        beta_scheduler = item[3]
        betas = item[4]
        n_T = item[5]
        lrate = item[6]
        net_type = item[7]
        batch_size = item[8]
        embedding_dim = item[9]
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
        
        path = f'eval/version_{index}/'
        os.makedirs(path, exist_ok=True)

        data = policy_kwargs.copy()
        data['lrate'] = lrate
        data['lrate_type'] = lrate_type
        data['alpha_schedule'] = alpha
        data.pop('observation_space')
        data.pop('action_space')

        with open(f'{path}data.json', 'w') as f:
            json.dump(data, f, separators=(',', ':'), indent=4)

        try:
            learn_bc(policy, device, gail_train_loader, gail_val_loader,
                    env, resume_last_train=False, lrate=lrate,
                    lrate_type=lrate_type, alpha_schedule=alpha, path=path)
            
        except Exception as exception:
            print("---------------------------------------------------")
            print(f"The process couldn't be trained due to ")
            print(f'{exception}')
            print("---------------------------------------------------")
            continue