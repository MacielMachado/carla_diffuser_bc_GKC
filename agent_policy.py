# modified from https://github.com/zhejz/carla-roach/blob/main/agents/rl_birdview/models/ppo_policy.py

from typing import Union, Dict, Tuple, Any
from functools import partial
import gym
import torch as th
import torch.nn as nn
import numpy as np
from PIL import Image
from diffusion import Model_mlp_diff_embed, ddpm_schedules

from carla_gym.utils.config_utils import load_entry_point


class AgentPolicy(nn.Module):

    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 policy_head_arch=[256, 256],
                 features_extractor_entry_point=None,
                 features_extractor_kwargs={},
                 distribution_entry_point=None,
                 distribution_kwargs={},
                 architecture='distribution',
                 betas=(1e-4, 0.02), 
                 n_T=20,
                 drop_prob=0.0):

        super(AgentPolicy, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor_entry_point = features_extractor_entry_point
        self.features_extractor_kwargs = features_extractor_kwargs
        self.distribution_entry_point = distribution_entry_point
        self.distribution_kwargs = distribution_kwargs
        self.architecture = architecture
        self.n_T = n_T
        self.betas = betas
        self.drop_prob = drop_prob
        self.guide_w = 0.0

        self.loss_mse = nn.MSELoss()

        if th.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.optimizer_class = th.optim.Adam
        self.optimizer_kwargs = {'eps': 1e-5}

        features_extractor_class = load_entry_point(features_extractor_entry_point)
        self.features_extractor = features_extractor_class(observation_space, **features_extractor_kwargs)

        distribution_class = load_entry_point(distribution_entry_point)
        self.action_dist = distribution_class(int(np.prod(action_space.shape)), **distribution_kwargs)

        if 'StateDependentNoiseDistribution' in distribution_entry_point:
            self.use_sde = True
            self.sde_sample_freq = 4
        else:
            self.use_sde = False
            self.sde_sample_freq = None

        # best_so_far
        # self.net_arch = [dict(pi=[256, 128, 64], vf=[128, 64])]
        self.policy_head_arch = list(policy_head_arch)
        self.activation_fn = nn.ReLU
        self.ortho_init = False
        self.latent_to_action = nn.Linear(256, 2)

        self.nn_downstream = Model_mlp_diff_embed(
            x_dim=256,
            n_hidden=128,
            y_dim=2,
            embed_dim=128,
            output_dim=2,
            is_dropout=False,
            is_batch=False,
            activation="relu",
            net_type='transformer',
            use_prev=False)

        self._build()
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)
            
    def reset_noise(self, n_envs: int = 1) -> None:
        assert self.use_sde, 'reset_noise() is only available when using gSDE'
        self.action_dist.sample_weights(self.dist_sigma, batch_size=n_envs)

    def _build(self) -> None:
        last_layer_dim_pi = self.features_extractor.features_dim
        policy_net = []
        for layer_size in self.policy_head_arch:
            policy_net.append(nn.Linear(last_layer_dim_pi, layer_size))
            policy_net.append(self.activation_fn())
            last_layer_dim_pi = layer_size

        self.policy_head = nn.Sequential(*policy_net).to(self.device)
        # mu->alpha/mean, sigma->beta/log_std (nn.Module, nn.Parameter)
        self.dist_mu, self.dist_sigma = self.action_dist.proba_distribution_net(last_layer_dim_pi)

        last_layer_dim_vf = self.features_extractor.features_dim

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # feature_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                # self.features_extractor: np.sqrt(2),
                self.policy_head: np.sqrt(2),
                # self.action_net: 0.01,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)

    def _get_features(self, obs_dict) -> th.Tensor:
        """
        :param birdview: th.Tensor (num_envs, frame_stack*channel, height, width)
        :param state: th.Tensor (num_envs, state_dim)
        """
        state = obs_dict['state']

        birdview = obs_dict['birdview'].float() / 255.0
        features = self.features_extractor(birdview, state)

        return features

    def _get_action_dist_from_features(self, features: th.Tensor):
        latent_pi = self.policy_head(features)
        mu = self.dist_mu(latent_pi)
        if isinstance(self.dist_sigma, nn.Parameter):
            sigma = self.dist_sigma
        else:
            sigma = self.dist_sigma(latent_pi)
        return self.action_dist.proba_distribution(mu, sigma), mu.detach().cpu().numpy(), sigma.detach().cpu().numpy()

    def evaluate_actions(self, obs_dict: Dict[str, th.Tensor], actions: th.Tensor):
        features = self._get_features(obs_dict)

        distribution, _, _ = self._get_action_dist_from_features(features)
        actions = self.scale_action(actions)
        log_prob = distribution.log_prob(actions)
        return log_prob, distribution.entropy_loss()

    def forward(self, obs_dict: Dict[str, np.ndarray], deterministic: bool = False, clip_action: bool = False):
        '''
        used in collect_rollouts(), do not clamp actions
        '''
        with th.no_grad():
            obs_tensor_dict = dict([(k, th.as_tensor(v).to(self.device)) for k, v in obs_dict.items()])
            features = self._get_features(obs_tensor_dict)
            distribution, mu, sigma = self._get_action_dist_from_features(features)
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)

        actions = actions.cpu().numpy()
        actions = self.unscale_action(actions)
        if clip_action:
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        log_prob = log_prob.cpu().numpy()
        features = features.cpu().numpy()
        return actions, log_prob, mu, sigma, features
    
    def forward_mse(self, obs_dict: Dict[str, np.ndarray], deterministic: bool = False, clip_action: bool = False):
        with th.no_grad():
            obs_tensor_dict = dict([(k, th.as_tensor(v).to(self.device)) for k, v in obs_dict.items()])
            features = self._get_features(obs_tensor_dict)
            actions = self._get_action(features)

        actions = actions.cpu().numpy()
        if clip_action:
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return actions
            

    def scale_action(self, action: th.Tensor, eps=1e-7) -> th.Tensor:
        # input action \in [a_low, a_high]
        # output action \in [d_low+eps, d_high-eps]
        d_low, d_high = self.action_dist.low, self.action_dist.high  # scalar

        if d_low is not None and d_high is not None:
            a_low = th.as_tensor(self.action_space.low.astype(np.float32)).to(action.device)
            a_high = th.as_tensor(self.action_space.high.astype(np.float32)).to(action.device)
            action = (action-a_low)/(a_high-a_low) * (d_high-d_low) + d_low
            action = th.clamp(action, d_low+eps, d_high-eps)
        return action

    def unscale_action(self, action: np.ndarray, eps=0.0) -> np.ndarray:
        # input action \in [d_low, d_high]
        # output action \in [a_low+eps, a_high-eps]
        d_low, d_high = self.action_dist.low, self.action_dist.high  # scalar

        if d_low is not None and d_high is not None:
            # batch_size = action.shape[0]
            a_low, a_high = self.action_space.low, self.action_space.high
            # same shape as action [batch_size, action_dim]
            # a_high = np.tile(self.action_space.high, [batch_size, 1])
            action = (action-d_low)/(d_high-d_low) * (a_high-a_low) + a_low
            # action = np.clip(action, a_low+eps, a_high-eps)
        return action

    def get_init_kwargs(self) -> Dict[str, Any]:
        init_kwargs = dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            policy_head_arch=self.policy_head_arch,
            features_extractor_entry_point=self.features_extractor_entry_point,
            features_extractor_kwargs=self.features_extractor_kwargs,
            distribution_entry_point=self.distribution_entry_point,
            distribution_kwargs=self.distribution_kwargs,
        )
        return init_kwargs
    
    def _get_action(self, features: th.Tensor):
        return self.latent_to_action(features)

    def evaluate_actions_mse(self, obs_dict: Dict[str, th.Tensor], actions: th.Tensor):
        features = self._get_features(obs_dict)
        actions_pred = self._get_action(features)
        loss = self.compute_mse_loss(actions, actions_pred)
        return loss
    
    def compute_mse_loss(self, actions, actions_pred):
        assert len(actions) == len(actions_pred)
        y_diff_pow_2 = th.pow(actions - actions_pred, 2)
        y_diff_sum = th.sum(y_diff_pow_2)/len(actions)
        mse = th.pow(y_diff_sum, 0.5)
        return mse

    @classmethod
    def load(cls, path):
        if th.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        saved_variables = th.load(path, map_location=device)
        # Create policy object
        model = cls(**saved_variables['policy_init_kwargs'])
        # Load weights
        model.load_state_dict(saved_variables['policy_state_dict'])
        model.to(device)
        return model, saved_variables['train_init_kwargs']

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def evaluate_actions_diffusion(self, obs_dict: Dict[str, th.Tensor], actions: th.Tensor):
        _ts = th.randint(1, self.n_T + 1, (actions.shape[0], 1)).to(self.device)
        
        # dropout context with some probability
        context_mask = th.bernoulli(th.zeros(obs_dict['birdview'].shape[0]) + self.drop_prob).to(self.device)

        # randomly sample some noise, noise ~ N(0, 1)
        noise = th.randn_like(actions).to(self.device)

        # add noise to clean target actions
        y_t = self.sqrtab[_ts] * actions + self.sqrtmab[_ts] * noise

        # use nn model to predict noise
        features = self._get_features(obs_dict)
        latent_pi = self.policy_head(features)
        noise_pred_batch = self.nn_downstream(y_t, latent_pi, _ts / self.n_T, context_mask)

        return self.loss_mse(noise, noise_pred_batch)













    def evaluate_actions_mse_diffusion(self, obs_dict: Dict[str, th.Tensor], actions: th.Tensor):
        features = self._get_features(obs_dict)
        
        # MSE
        actions_pred = self._get_action(features)
        loss_mse = self.compute_mse_loss(actions, actions_pred)

        # Diffusion
        _ts = th.randint(1, self.n_T + 1, (actions.shape[0], 1)).to(self.device)
        context_mask = th.bernoulli(th.zeros(obs_dict['birdview'].shape[0]) + self.drop_prob).to(self.device)
        noise = th.randn_like(actions).to(self.device)
        y_t = self.sqrtab[_ts] * actions + self.sqrtmab[_ts] * noise
        latent_pi = self.policy_head(features)
        noise_pred_batch = self.nn_downstream(y_t, latent_pi, _ts / self.n_T, context_mask)
        loss_diffusion = self.loss_mse(noise, noise_pred_batch)

        return loss_mse, loss_diffusion


    def forward_diffusion(self, obs_dict: Dict[str, np.ndarray], deterministic: bool = False, clip_action: bool = False):
        # also use this as a shortcut to avoid doubling batch when guide_w is zero
        is_zero = False
        if self.guide_w > -1e-3 and self.guide_w < 1e-3:
            is_zero = True

        # how many noisy actions to begin with
        n_sample = len(obs_dict['birdview'])
        y_shape = (n_sample, 2)

        # sample initial noise, y_0 ~ N(0, 1),
        y_i = th.randn(y_shape).to(self.device)
        
        context_mask = th.zeros(len(obs_dict['birdview'])).to(self.device)
        
        # run denoising chain
        y_i_store = []  # if want to trace how y_i evolved
        for i in range(self.n_T, 0, -1):
            t_is = th.tensor([i / self.n_T]).to(self.device)
            t_is = t_is.repeat(n_sample, 1)

            if not is_zero:
                # double batch
                y_i = y_i.repeat(2, 1)
                t_is = t_is.repeat(2, 1)

            z = th.randn(y_shape).to(self.device) if i > 1 else 0
            with th.no_grad():
                obs_tensor_dict = dict([(k, th.as_tensor(v).to(self.device)) for k, v in obs_dict.items()])
                features = self._get_features(obs_tensor_dict)
                latent_pi = self.policy_head(features)
                eps = self.nn_downstream(y_i, latent_pi, t_is / self.n_T, context_mask)

            actions = self.oneover_sqrta[i] * (y_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
        actions = actions.cpu().numpy()
        if clip_action:
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return actions









        with th.no_grad():
            obs_tensor_dict = dict([(k, th.as_tensor(v).to(self.device)) for k, v in obs_dict.items()])
            features = self._get_features(obs_tensor_dict)
            actions = self._get_action(features)

        actions = actions.cpu().numpy()
        if clip_action:
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return actions