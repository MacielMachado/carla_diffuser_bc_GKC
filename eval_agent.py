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


env_configs = {
    'carla_map': 'Town01',
    'num_zombie_vehicles': [0, 150],
    'num_zombie_walkers': [0, 300],
    'weather_group': 'dynamic_1.0'
}


def evaluate_policy(env, policy, video_path, min_eval_steps=3000):
    policy = policy.eval()
    t0 = time.time()
    for i in range(env.num_envs):
        env.set_attr('eval_mode', True, indices=i)
    obs = env.reset()

    list_render = []
    ep_stat_buffer = []
    route_completion_buffer = []
    ep_events = {}
    for i in range(env.num_envs):
        ep_events[f'venv_{i}'] = []
    n_step = 0
    n_timeout = 0
    env_done = np.array([False]*env.num_envs)
    # while n_step < min_eval_steps:
    while n_step < min_eval_steps or not np.all(env_done):
        if policy.architecture == 'mse':
            actions = policy.forward_mse(obs, deterministic=True, clip_action=True)
            log_probs = np.array([0.0])
            mu = np.array([[0.0, 0.0]])
            sigma = np.array([0.0, 0.0])

        elif policy.architecture == 'diffusion':
            actions = policy.forward_diffusion(obs, deterministic=True, clip_action=True)
            log_probs = np.array([0.0])
            mu = np.array([[0.0, 0.0]])
            sigma = np.array([0.0, 0.0])

        elif policy.architecture == 'distribution':
            actions, log_probs, mu, sigma, _ = policy.forward(obs, deterministic=True, clip_action=True)

        obs, reward, done, info = env.step(actions)

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
            ep_events[f'venv_{i}'].append(info[i]['episode_event'])
            n_timeout += int(info[i]['timeout'])

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
    return avg_ep_stat, avg_route_completion, ep_events

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


def env_maker():
    cfg = json.load(open("config.json", "r"))
    env = EndlessEnv(obs_configs=obs_configs, reward_configs=reward_configs,
                    terminal_configs=terminal_configs, host='localhost', port=2020,
                    seed=2021, no_rendering=True, **env_configs)
    env = RlBirdviewWrapper(env)
    return env


if __name__ == '__main__':
    env = SubprocVecEnv([env_maker])

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
        'architecture': 'diffusion',
        'betas': (1e-4, 0.02),
        'n_T': 20,
}
    
    model_path = 'ckpt_mse_sem_trajetoria/bc_ckpt_8_min_eval.pth'

    policy = AgentPolicy(**policy_kwargs)
    policy.load_state_dict(th.load(model_path))
