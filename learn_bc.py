import torch.optim as optim
import numpy as np
import tqdm
import torch as th
from pathlib import Path
import wandb
import gym
import json

from expert_dataset import ExpertDataset
from agent_policy import AgentPolicy
from carla_gym.envs import EndlessEnv
from rl_birdview_wrapper import RlBirdviewWrapper
from data_collect import reward_configs, terminal_configs, obs_configs
from eval_agent import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv


env_configs = {
    'carla_map': 'Town01',
    'num_zombie_vehicles': [0, 150],
    'num_zombie_walkers': [0, 300],
    'weather_group': 'dynamic_1.0'
}


def learn_bc(policy, device, expert_loader, eval_loader, env, resume_last_train):
    output_dir = Path('outputs_diff_mse_diffusion')
    output_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = output_dir / 'checkpoint.txt'

    ckpt_dir = Path('ckpt_diff_sem_trajetoria_mse_diffusion')
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if resume_last_train:
        with open(last_checkpoint_path, 'r') as f:
            wb_run_path = f.read()
        api = wandb.Api()
        wandb_run = api.run(wb_run_path)
        wandb_run_id = wandb_run.id
        ckpt_path = (ckpt_dir / 'ckpt_latest.pth').as_posix()
        saved_variables = th.load(ckpt_path, map_location='cuda')
        train_kwargs = saved_variables['train_init_kwargs']
        start_ep = train_kwargs['start_ep']
        i_steps = train_kwargs['i_steps']

        policy.load_state_dict(saved_variables['policy_state_dict'])
        wandb.init(project='gail-carla2', id=wandb_run_id, resume='must')
    else:
        run = wandb.init(project='diffusion', reinit=True)
        with open(last_checkpoint_path, 'w') as log_file:
            log_file.write(wandb.run.path)
        start_ep = 0
        i_steps = 0

    video_path = Path('video_diff_sem_trajetoria_mse_diffusion')
    video_path.mkdir(parents=True, exist_ok=True)

    initial_lr = 1e-5
    optimizer = optim.Adam(policy.parameters(), lr=initial_lr)
    episodes = 500
    ent_weight = 0.01
    min_eval_loss = np.inf
    eval_step = int(1e5)
    steps_last_eval = 0

    for i_episode in tqdm.tqdm(range(start_ep, episodes)):
        current_lr = decay_lr(i_episode, initial_lr, episodes)
        alpha = exponential_decay(i_episode, episodes)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        total_loss = 0
        i_batch = 0
        policy = policy.train()
        
        for expert_batch in expert_loader:
            expert_obs_dict, expert_action = expert_batch
            obs_tensor_dict = {
                'state': expert_obs_dict['state'].float().to(device),
                'birdview': expert_obs_dict['birdview'].float().to(device)
            }
            expert_action = expert_action.to(device)

            if policy.architecture == 'distribution':
                alogprobs, entropy_loss = policy.evaluate_actions(obs_tensor_dict, expert_action)
                bcloss = -alogprobs.mean()
                loss = bcloss + ent_weight * entropy_loss
            elif policy.architecture == 'mse':
                loss = policy.evaluate_actions_mse(obs_tensor_dict, expert_action)
            elif policy.architecture == 'diffusion':
                loss = policy.evaluate_actions_diffusion(obs_tensor_dict, expert_action)
            elif policy.architecture == 'mse_diffusion':
                loss_mse, loss_diffusion = policy.evaluate_actions_mse_diffusion(obs_tensor_dict, expert_action)
                loss = alpha * loss_mse + (1-alpha) * loss_diffusion

            total_loss += loss
            i_batch += 1
            i_steps += expert_obs_dict['state'].shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_eval_loss = 0
        i_eval_batch = 0
        for expert_batch in eval_loader:
            expert_obs_dict, expert_action = expert_batch
            obs_tensor_dict = {
                'state': expert_obs_dict['state'].float().to(device),
                'birdview': expert_obs_dict['birdview'].float().to(device)
            }
            expert_action = expert_action.to(device)

            with th.no_grad():
                if policy.architecture == 'distribution':
                    alogprobs, entropy_loss = policy.evaluate_actions(obs_tensor_dict, expert_action)
                    bcloss = -alogprobs.mean()
                    eval_loss = bcloss + ent_weight * entropy_loss
                elif policy.architecture == 'mse':
                    eval_loss = policy.evaluate_actions_mse(obs_tensor_dict, expert_action)
                elif policy.architecture == 'diffusion' or policy.architecture == 'mse_diffusion':
                    eval_loss = policy.evaluate_actions_diffusion(obs_tensor_dict, expert_action)

            total_eval_loss += eval_loss
            i_eval_batch += 1
        
        loss = total_loss / i_batch
        eval_loss = total_eval_loss / i_eval_batch
        wandb.log({'loss': loss, 'eval_loss': eval_loss, 'current_lr': current_lr}, step=i_steps)

        if i_steps - steps_last_eval > eval_step:
            eval_video_path = (video_path / f'bc_eval_{i_steps}.mp4').as_posix()
            avg_ep_stat, avg_route_completion, ep_events = evaluate_policy(env, policy, eval_video_path)
            env.reset()
            wandb.log(avg_ep_stat, step=i_steps)
            wandb.log(avg_route_completion, step=i_steps)
            steps_last_eval = i_steps

        if min_eval_loss > eval_loss:
            ckpt_path = (ckpt_dir / f'bc_ckpt_{i_episode}_min_eval.pth').as_posix()
            th.save({'policy_state_dict': policy.state_dict()}, ckpt_path)
            min_eval_loss = eval_loss

        train_init_kwargs = {'start_ep': i_episode, 'i_steps': i_steps}
        ckpt_path = (ckpt_dir / 'ckpt_latest.pth').as_posix()
        th.save({'policy_state_dict': policy.state_dict(), 'train_init_kwargs': train_init_kwargs}, ckpt_path)
        wandb.save(f'./{ckpt_path}')
    run.finish()

def decay_lr(epoch, lrate, episodes):
    return lrate * ((np.cos((epoch / episodes) * np.pi) + 1) / 2)

def exponential_decay(epoch, total_episodes, initial_value=1.0, final_value=0.01):
    decay_rate = -np.log(final_value / initial_value) / total_episodes
    return initial_value * np.exp(-decay_rate * epoch)

def env_maker():
    cfg = json.load(open("config.json", "r"))
    env = EndlessEnv(obs_configs=obs_configs, reward_configs=reward_configs,
                    terminal_configs=terminal_configs, host='localhost', port=2020,
                    seed=2021, no_rendering=True, **env_configs)
    env = RlBirdviewWrapper(env)
    return env

if __name__ == '__main__':
    env = DummyVecEnv([env_maker])

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
        'architecture': 'mse_diffusion',
        'betas': (1e-4, 0.02),
        'n_T': 50,
}

    device = 'cuda'

    policy = AgentPolicy(**policy_kwargs)
    # ckpt_path = 'ckpt_mse_sem_trajetoria_update/bc_ckpt_4_min_eval.pth'

    # trained_state_dict = th.load(ckpt_path)

    # diff_policy_dict = policy.state_dict()
    # chaves_a_reutilizar = [
    #     'features_extractor', 
    #     'features_extractor.linear', 
    #     'features_extractor.state_linear', 
    #     'policy_head'
    # ]

    # for key in trained_state_dict['policy_state_dict']:
    #     if any([key.startswith(chave) for chave in chaves_a_reutilizar]):
    #         diff_policy_dict[key] = trained_state_dict['policy_state_dict'][key]

    # policy.load_state_dict(diff_policy_dict)

    # for name, param in policy.named_parameters():
    #     if any([name.startswith(chave) for chave in chaves_a_reutilizar]):
    #         param.requires_grad = False

    policy.to(device)

    batch_size = 24

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

    learn_bc(policy, device, gail_train_loader, gail_val_loader, env, resume_last_train)