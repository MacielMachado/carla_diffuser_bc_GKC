import numpy as np
import time
import gym
from gym.wrappers.monitoring.video_recorder import ImageEncoder
from agent_policy import AgentPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
import json
from carla_gym.envs import EndlessEnv
from data_collect import reward_configs, terminal_configs, obs_configs
from rl_birdview_wrapper import RlBirdviewWrapper
import torch as th
from pathlib import Path
import os
import math
from stable_baselines3.common.vec_env import DummyVecEnv
from carla_gym.envs import EndlessFixedSpawnEnv


env_configs = {
    'carla_map': 'Town01',
    'num_zombie_vehicles': [0, 150],
    'num_zombie_walkers': [0, 300],
    'weather_group': 'dynamic_1.0'
}

env_configs = {
    'carla_map': 'Town01',
    'weather_group': 'dynamic_1.0',
    'routes_group': ''
}

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


def evaluate_policy(env, policy, video_path, min_eval_steps=1000):
    # env = DummyVecEnv([env_maker]) 
    policy = policy.eval()
    t0 = time.time()
    for i in range(env.num_envs):
        env.set_attr('eval_mode', True, indices=i)
    obs = env.reset()
    previous_position = obs['gnss'][0]

    list_render = []
    ep_stat_buffer = []
    route_completion_buffer = []
    ep_events = {}
    for i in range(env.num_envs):
        ep_events[f'venv_{i}'] = []
    n_step = 0
    n_timeout = 0
    env_done = np.array([False]*env.num_envs)
    distance_traveled = 0
    distance_traveled_list = []
    while n_step < min_eval_steps or not np.all(env_done):
        if policy.architecture == 'mse':
            actions = policy.forward_mse(obs, deterministic=True, clip_action=True)
            log_probs = np.array([0.0])
            mu = np.array([[0.0, 0.0]])
            sigma = np.array([0.0, 0.0])

        elif policy.architecture == 'diffusion' or policy.architecture == 'mse_diffusion':
            actions = policy.forward_diffusion(obs, deterministic=True, clip_action=True)
            log_probs = np.array([0.0])
            mu = np.array([[0.0, 0.0]])
            sigma = np.array([0.0, 0.0])

        elif policy.architecture == 'distribution':
            actions, log_probs, mu, sigma, _ = policy.forward(obs, deterministic=True, clip_action=True)

        obs, reward, done, info = env.step(actions)

        distance_traveled += calculate_distance_traveled(obs['gnss'][0], previous_position)
        print(distance_traveled)

        for i in range(env.num_envs):
            env.set_attr('action_log_probs', log_probs[i], indices=i)
            env.set_attr('action_mu', mu[i], indices=i)
            env.set_attr('action_sigma', sigma[i], indices=i)
        # print(f"name: {video_path}, n_step: {n_step}, np.all(env_done): {np.all(env_done)}")
        list_render.append(env.render(mode='rgb_array'))

        n_step += 1
        env_done |= done

        for i in np.where(done)[0]:
            if not info[i]['timeout']:
                ep_stat_buffer.append(info[i]['episode_stat'])
            if n_step < min_eval_steps or not np.all(env_done):
                route_completion_buffer.append(info[i]['route_completion'])
            distance_traveled_list.append(distance_traveled)
            distance_traveled = 0
            ep_events[f'venv_{i}'].append(info[i]['episode_event'])
            n_timeout += int(info[i]['timeout'])
            obs = env.reset()
            previous_position = obs['gnss'][0]

    for ep_info in info:
        route_completion_buffer.append(ep_info['route_completion'])

    # conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge
    encoder = ImageEncoder(video_path, list_render[0].shape, 30, 30)
    for im in list_render:
        encoder.capture_frame(im)
    encoder.close()

    avg_ep_stat = get_avg_ep_stat(ep_stat_buffer, prefix='eval/')
    avg_route_completion = get_avg_route_completion(route_completion_buffer, prefix='eval/')
    avg_ep_stat['eval/eval_timeout'] = n_timeout

    duration = time.time() - t0
    avg_ep_stat['time/t_eval'] = duration
    avg_ep_stat['time/fps_eval'] = n_step * env.num_envs / duration

    for i in range(env.num_envs):
        env.set_attr('eval_mode', False, indices=i)
    obs = env.reset()
    # del env
    return avg_ep_stat, avg_route_completion, ep_events, np.array(distance_traveled_list).mean()


def get_avg_ep_stat(ep_stat_buffer, prefix=''):
    avg_ep_stat = {}
    n_episodes = float(len(ep_stat_buffer))
    if n_episodes > 0:
        for ep_info in ep_stat_buffer:
            for k, v in ep_info.items():
                k_avg = f'{prefix}{k}'
                if k_avg in avg_ep_stat:
                    avg_ep_stat[k_avg] += v
                else:
                    avg_ep_stat[k_avg] = v

        for k in avg_ep_stat.keys():
            avg_ep_stat[k] /= n_episodes
    avg_ep_stat[f'{prefix}completed_n_episodes'] = n_episodes

    return avg_ep_stat


def get_avg_route_completion(ep_route_completion, prefix=''):
    avg_ep_stat = {}
    n_episodes = float(len(ep_route_completion))
    if n_episodes > 0:
        for ep_info in ep_route_completion:
            for k, v in ep_info.items():
                k_avg = f'{prefix}{k}'
                if k_avg in avg_ep_stat:
                    avg_ep_stat[k_avg] += v
                else:
                    avg_ep_stat[k_avg] = v

        for k in avg_ep_stat.keys():
            avg_ep_stat[k] /= n_episodes
    avg_ep_stat[f'{prefix}avg_n_episodes'] = n_episodes

    return avg_ep_stat


def calculate_distance_traveled(current_position, previous_position):
    """
    Calcula a distância percorrida por um veículo desde a última posição registrada.
    
    Args:
        vehicle: Objeto do veículo no CARLA.
        previous_position: Última posição registrada do veículo (carla.Location).
    
    Returns:
        distance: Distância percorrida desde a última posição (float).
        current_position: Posição atual do veículo (carla.Location).
    """
    
    if previous_position is None:
        # Primeira chamada: sem deslocamento
        return 0.0, current_position

    # Calcular o deslocamento 3D
    dx = current_position[0] - previous_position[0]
    dy = current_position[1] - previous_position[1]
    dz = current_position[2] - previous_position[2]
    distance = math.sqrt(dx**2 + dy**2)
    
    return distance

def env_maker():
    cfg = json.load(open("config.json", "r"))
    env = EndlessFixedSpawnEnv(obs_configs=obs_configs, reward_configs=reward_configs,
                    terminal_configs=terminal_configs, host='localhost', port=2020,
                    seed=2021, no_rendering=True, **env_configs, spawn_point=spawn_point)
    env = RlBirdviewWrapper(env)
    return env


if __name__ == '__main__':
    # env = SubprocVecEnv([env_maker])
    env = DummyVecEnv([env_maker]) 
    # env = env_maker

    resume_last_train = False

    observation_space = {}
    observation_space['birdview'] = gym.spaces.Box(low=0, high=255, shape=(3, 192, 192), dtype=np.uint8)
    observation_space['state'] = gym.spaces.Box(low=-10.0, high=30.0, shape=(6,), dtype=np.float32)
    observation_space = gym.spaces.Dict(**observation_space)

    action_space = gym.spaces.Box(low=np.array([0, -1]), high=np.array([1, 1]), dtype=np.float32)

    # network
    policy_kwargs = {
        'observation_space': observation_space,
        'action_space': action_space,
        'policy_head_arch': [256, 256],
        'features_extractor_entry_point': 'torch_layers:XtMaCNN',
        'features_extractor_kwargs': {'states_neurons': [256,256]},
        'distribution_entry_point': 'distributions:DiagGaussianDistribution',
        'architecture': 'mse',
        'betas': (1e-4, 0.02),
        'n_T': 20,
}
    avg_lenght_in_m = 0
    avg_lenght_in_m_max = 0
    available_models = sorted(os.listdir('ckpt_mse_sem_trajetoria_update'), key=lambda x: os.path.getctime(os.path.join('ckpt_mse_sem_trajetoria_update', x)))
    for model in available_models:
        for i in range(5):
            # model_path = 'ckpt_mse_sem_trajetoria_update/bc_ckpt_9_min_eval.pth'
            model_path = 'ckpt_mse_sem_trajetoria_update/' + model 
            video_path = 'evaluate_2/video_mse_sem_trajetoria_update'
            os.makedirs(video_path, exist_ok=True)
            video_path = video_path + f'/{model.split(".")[0]}_{i}.mp4'
            policy = AgentPolicy(**policy_kwargs)
            policy.load_state_dict(th.load(model_path)['policy_state_dict'])
            policy.to('cuda')
            avg_ep_stat, avg_route_completion, ep_events, distance_traveled = evaluate_policy(env=env, policy=policy, video_path=video_path)
            avg_lenght_in_m += avg_route_completion['eval/route_length_in_m']
        
            print(f'model: {model} - index: {i} - {avg_lenght_in_m/(i+1)}')

        avg_lenght_in_m = avg_lenght_in_m/(i+1)
        if avg_lenght_in_m > avg_lenght_in_m_max:
            avg_lenght_in_m_max = avg_lenght_in_m
            best_model = model
    print("-----------------------------------------------------------")
    print(f'best_model: {best_model}')
    print(f'avg_lenght_in_m_max: {avg_lenght_in_m_max}')
