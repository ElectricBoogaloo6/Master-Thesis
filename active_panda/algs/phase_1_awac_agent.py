from copy import deepcopy
import itertools
import numpy as np
import torch
print("PyTorch version:", torch.__version__)
print("PyTorch CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")
from torch.optim import Adam
import gym
import panda_gym
import time
import torch.nn.functional as F
import os
import pandas as pd

import sys
import argparse
from datetime import datetime
import random
from torch.utils.tensorboard import SummaryWriter

from phase_1_func import categorize_uncertainties, choose_starting_position, CustomResetWrapper, choose_task, choose_random_position, get_base_env, trajectories_for_position

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
ACTIVE_PANDA_DIR = os.path.dirname(CURRENT_DIR)
ENVS_DIR = os.path.join(ACTIVE_PANDA_DIR, 'envs')
RESULTS_DIR = os.path.join(ACTIVE_PANDA_DIR, "Phase_1_results")
sys.path.append(ENVS_DIR)
sys.path.append(PARENT_DIR + '/utils/')
from utils_awac import ActionNormalizer, ResetWrapper, TimeFeatureWrapper, TimeLimitWrapper, reconstruct_state, save_results
import core as core


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size, device):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

        self.device = device

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32, idxs=None):
        if idxs is None:
            idxs = np.random.randint(0, self.size, size=batch_size)

        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k, v in batch.items()}

        # return {k: torch.FloatTensor(v).to(self.device) for k, v in batch.items()}

    def sample_batch_array(self, batch_size=32, idxs=None):
        if idxs is None:
            idxs = np.random.randint(0, self.size, size=batch_size)

        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])

        return batch
    

class AWACAgent:
    def __init__(self,
                    env,
                    test_env,
                    task_name,
                    writer,
                    actor_critic=core.MLPActorCritic,
                    with_layer_norm=False,
                    steps_per_epoch=1000,
                    epochs=1000,
                    replay_size=int(2000000),
                    gamma=0.99,
                    polyak=0.995,
                    lr=3e-4,
                    p_lr=3e-4,
                    alpha=0.0,
                    batch_size=1024,
                    start_steps=10000,
                    update_after=0,
                    update_every=50,
                    num_test_episodes=10,
                    max_ep_len=1000,
                    logger_kwargs=dict(),
                    save_freq=1,
                    algo='SAC',
                    # current_condition=None,
                    group=None,
                    log_data=None,
                    participant_code=None,
                    testing_flow=False,
                    training_flow=False,
                    roll_demos_in_training=False,
                    load_pretrained=False,
                    pretrained_model_step=None,
                    pretrained_model_path=None,
                    show_uncertainty_categories=None,
                    show_uncertainty_category_colors=None,
                    condition='blind',
                    continue_experiment_training=False,
                    selection_strategy='Expectation-oriented'):
        
        self.env = env
        self.test_env = test_env
        self.obs_dim = env.observation_space['observation'].shape[0] + 3  # add goal position
        self.act_dim = env.action_space.shape[0]
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("DEVICE: ", self.device)

        # Create actor-critic module and target networks
        self.ac = actor_critic(self.obs_dim, self.env.action_space, device=self.device,
                               special_policy='awac', with_layer_norm=with_layer_norm)
        self.ac_targ = actor_critic(self.obs_dim, self.env.action_space, device=self.device,
                                    special_policy='awac', with_layer_norm=with_layer_norm)
        self.ac_targ.load_state_dict(self.ac.state_dict())
        self.gamma = gamma

        # tensorboard logger
        self.writer = writer

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                          size=replay_size, device=self.device)
        self.demo_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                          size=replay_size, device=self.device)
        
        self.algo = algo
        self.p_lr = p_lr
        self.lr = lr
        self.alpha = 0

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.p_lr, weight_decay=1e-4)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.update_after = update_after
        self.update_every = update_every
        self.batch_size = batch_size
        self.save_freq = save_freq
        self.polyak = polyak

        # Set up model saving
        print("Running Offline RL algorithm: {}".format(self.algo))

    def populate_replay_buffer(self, demos_list):
        for demo in demos_list:
            demo_length = demo['state_trajectory'].shape[0]
            for step in range(demo_length):
                state = demo['state_trajectory'][step].copy()
                action = demo['action_trajectory'][step].copy()
                next_state = demo['next_state_trajectory'][step].copy()
                reward = demo['state_trajectory'][step][0]
                done = demo['done_trajectory'][step][0]

                transition = [state, action, reward, next_state, done]
                self.demo_buffer.store(*transition)

        print("[Populate offline base demos]: Finished")

    def sample_mixed_batch(self):
        demo_batch_size = int(self.batch_size / 2)
        rollout_batch_size = int(self.batch_size / 2)

        if self.demo_buffer.size > 0:
            demo_batch = self.demo_buffer.sample_batch_array(demo_batch_size)
        else:
            demo_batch = self.replay_buffer.sample_batch_array(demo_batch_size)

        rollout_batch = self.replay_buffer.sample_batch_array(rollout_batch_size)

        demo_obs = demo_batch['obs']
        demo_obs2 = demo_batch['obs2']
        demo_act = demo_batch['act']
        demo_rew = demo_batch['rew']
        demo_done = demo_batch['done']

        rollout_obs = rollout_batch['obs']
        rollout_obs2 = rollout_batch['obs2']
        rollout_act = rollout_batch['act']
        rollout_rew = rollout_batch['rew']
        rollout_done = rollout_batch['done']

        mixed_obs = np.concatenate((demo_obs, rollout_obs))
        mixed_obs2 = np.concatenate((demo_obs2, rollout_obs2))
        mixed_act = np.concatenate((demo_act, rollout_act))
        mixed_rew = np.concatenate((demo_rew, rollout_rew))
        mixed_done = np.concatenate((demo_done, rollout_done))

        res_dict = {'obs': mixed_obs, 'obs2':mixed_obs2, 'act':mixed_act, 'rew':mixed_rew, 'done':mixed_done}

        return {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k, v in res_dict.items()}
    
    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        self.writer.add_scalar(tag='loss/q1_loss', scalar_value=loss_q1,
                               global_step=self.total_step)
        self.writer.add_scalar(tag='loss/q2_loss', scalar_value=loss_q2,
                               global_step=self.total_step)
        self.writer.add_scalar(tag='loss/total_q_loss', scalar_value=loss_q,
                               global_step=self.total_step)

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info
    
    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        o = data['obs']

        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        v_pi = torch.min(q1_pi, q2_pi)

        # beta = 2
        beta = 0.3
        q1_old_actions = self.ac.q1(o, data['act'])
        q2_old_actions = self.ac.q2(o, data['act'])
        q_old_actions = torch.min(q1_old_actions, q2_old_actions)

        adv_pi = q_old_actions - v_pi
        weights = F.softmax(adv_pi / beta, dim=0)
        policy_logpp = self.ac.pi.get_logprob(o, data['act'])
        loss_pi = (-policy_logpp * len(weights) * weights.detach()).mean()

        self.writer.add_scalar(tag='loss/pi_loss', scalar_value=loss_pi,
                               global_step=self.total_step)

        # Useful info for logging
        pi_info = dict(LogPi=policy_logpp.cpu().detach().numpy())

        return loss_pi, pi_info
    
    def update(self, data, update_timestep):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Record things
        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, deterministic=False):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(self.device), deterministic)
    
    def test_agent(self, task_name, evaluation_res=None, success_evaluation_res=None, test_all_models=False,participant_code=None, model_step=None):
        
        test_episode_reward = 0.0
        total_success_num = 0.0

        # Initialize a dictionary to store the results
        test_results = {
            'participant_code': participant_code,
            'model_step': model_step,
            'average_episode_reward': None,
            'average_success_rate': None,
            'episode_rewards': [],
            'successes': []
        }

        if task_name in ['Phase1TaskCentre', 'Phase1TaskRight', 'Phase1TaskLeft']:
            goal_pos = self.test_env.task.goal_position
            object_positions = self.test_env.task.predefined_obj_positions

            for obj_pos in object_positions:
                state_ = self.test_env.reset(goal_pos=goal_pos, object_pos=obj_pos)
                done_ = False
                episode_reward = 0.0
                whether_success = 0.0

                while not done_:
                    state_ = reconstruct_state(state_)
                    action_ = self.get_action(state_, deterministic=True)
                    # action_ = self.get_action(state_)
                    next_state_, reward_, done_, info_ = self.test_env.step(action_)
                    episode_reward += reward_
                    state_ = next_state_

                if info_['is_success']:
                    total_success_num += 1.0
                    whether_success = 1.0

                test_results['episode_rewards'].append(episode_reward)
                test_results['successes'].append(whether_success)

        num_test_episodes = len(test_results['episode_rewards'])
        test_results['average_episode_reward'] = np.mean(test_results['episode_rewards'])
        test_results['average_success_rate'] = total_success_num / num_test_episodes if num_test_episodes > 0 else 0.0

        # log to TensorBoard
        self.writer.add_scalar(tag='episode_reward/train/per_env_step',
                            scalar_value=test_results['average_episode_reward'],
                            global_step=self.total_step)
        self.writer.add_scalar(tag='success_rate/train/per_env_step',
                            scalar_value=test_results['average_success_rate'],
                            global_step=self.total_step)

        print(f"[{self.total_step} environment steps finished]: Average episode reward is {test_results['average_episode_reward']}, Success rate is {test_results['average_success_rate']}")

        return test_results
    
    def estimate_traj_uncertainty_td(self, traj):
        """
        Estimates the uncertainty of a trajectory by calculating the mean temporal difference (TD) error over the trajectory.
        The TD error is a measure of the difference between the predicted value and the actual return of a state-action pair.

        - traj: A trajectory, which is a list of tuples. Each tuple contains (state, action, reward, next_state, done),
        representing the steps taken in an episode.
        """
        # zero cumulative TD error
        cumulated_td_error = 0.0
        # number of steps in the trajectory
        episode_length = len(traj)

        # each step in the trajectory
        for state, action, reward, next_state, done in traj:
            # Convert the state to a tensor, reshape it to add a batch dimension, and move it to the correct device
            state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state.reshape(1, -1)).to(self.device)
            action_tensor = torch.FloatTensor(action.reshape(1, -1)).to(self.device)

            # Get the action for the next state from the target actor
            next_action_tuple = self.ac_targ.pi(next_state_tensor)
            if isinstance(next_action_tuple, tuple):
                next_action = next_action_tuple[0]
            else:
                next_action = next_action_tuple
            next_action = next_action.detach() 

            # Q-values for the next state-action pair from the target critics
            next_q1_value = self.ac_targ.q1(next_state_tensor, next_action).detach()
            next_q2_value = self.ac_targ.q2(next_state_tensor, next_action).detach()
            # next_q_value = torch.mean(torch.stack([next_q1_value, next_q2_value])).item()
            next_q_value = torch.min(next_q1_value, next_q2_value)

            # Same equation as before
            curr_return = reward + self.gamma * next_q_value * (1 - done)

            # Q-values for the current state-action pair from the critics
            curr_q1_value = self.ac.q1(state_tensor, action_tensor).detach()
            curr_q2_value = self.ac.q2(state_tensor, action_tensor).detach()
            # curr_q_value = torch.mean(torch.stack([curr_q1_value, curr_q2_value])).item()
            curr_q_value = torch.min(curr_q1_value, curr_q2_value)

            # Compute the TD error
            td_error = curr_return - curr_q_value

            # Accumulate the absolute TD error
            cumulated_td_error += abs(td_error)

        # Calculate the average TD error (trajectory uncertainty)
        traj_uncertainty = cumulated_td_error / episode_length
        return traj_uncertainty
    
    def initialize_starting_position_uncertainties(self, num_runs=10):
        predefined_positions = self.env.task.predefined_obj_positions
        traj_uncertainty_per_position = {idx: [] for idx in range(len(predefined_positions))}
        
        for idx, position in enumerate(predefined_positions):
            object_color = self.env.unwrapped.task.colors[idx % len(self.env.unwrapped.task.colors)]
            for _ in range(num_runs):
                state = self.env.reset(object_pos=position, object_color=object_color)
                done = False
                rollout_traj = []

                while not done:
                    reshaped_state = reconstruct_state(state)
                    action = self.get_action(reshaped_state, deterministic=False)
                    next_state, reward, done, _ = self.env.step(action)
                    reshaped_next_state = reconstruct_state(next_state)
                    rollout_traj.append((reshaped_state, action, reward, reshaped_next_state, float(done)))
                    state = next_state

                traj_uncertainty = self.estimate_traj_uncertainty_td(rollout_traj)
                logged_uncertainty = np.log2(float(traj_uncertainty))
                traj_uncertainty_per_position[idx].append(logged_uncertainty)
                # print(f"Initialized uncertainty for position {idx}: {logged_uncertainty}")

        return traj_uncertainty_per_position

    def run(self, 
            task_name, 
            method, 
            max_demo_num=10, 
            model_saving_interval=5000,
            base_runs_per_position=1,
            human_input_rollout_interval=10,
            max_queries=5,
            log_data = None,
            participant_code=None,
            testing_flow=False,
            training_flow=False,
            roll_demos_in_training=False,
            group=None,
            load_pretrained=False,
            pretrained_model_step=None,
            pretrained_model_path=None,
            show_uncertainty_categories=False,
            show_uncertainty_category_colors=False,
            condition='blind',
            continue_experiment_training=False,
            selection_strategy='Expectation-oriented'):
        
        """
        PARTICIPANT SELECTIONS
        """
        P1_blind = [9, 8, 7, 8, 7]
        P1_visible = [4, 5, 4, 0, 2]
        P2_blind = [9, 8, 7, 3, 6]
        P2_visible = [7, 5, 6, 8, 5]
        P3_blind = [0, 6, 7, 4, 9]
        P3_visible = [7, 8, 9, 3, 9]
        P4_blind = [9, 7, 6, 6, 6]
        P4_visible = [6, 7, 9, 0, 4]
        P5_blind = [7, 8, 6, 7, 9]
        P5_visible = [5, 5, 8, 8, 6]
        P6_blind = [7, 7, 6, 0, 6]
        P6_visible = [1, 5, 5, 0, 0]
        P7_blind = [9, 9, 9, 9, 9]
        P7_visible = [6, 6, 5, 5, 0]
        P8_blind = [8, 6, 6, 6, 6]
        P8_visible = [0, 5, 5, 0, 4]
        P9_blind = [1, 3, 4, 0, 5]
        P9_visible = [0, 3, 1, 2, 4]
        P10_blind = [9, 6, 7, 8, 6]
        P10_visible = [3, 0, 4, 5, 5]
        P11_blind = [9, 7, 8, 8, 9]
        P11_visible = [9, 8, 7, 8, 7]
        P12_blind = [5, 4, 4, 6, 6]
        P12_visible = [5, 6, 6, 6, 5]

        participant_positions = {
            'P1_blind': P1_blind,
            'P1_visible': P1_visible,
            'P2_blind': P2_blind,
            'P2_visible': P2_visible,
            'P3_blind': P3_blind,
            'P3_visible': P3_visible,
            'P4_blind': P4_blind,
            'P4_visible': P4_visible,
            'P5_blind': P5_blind,
            'P5_visible': P5_visible,
            'P6_blind': P6_blind,
            'P6_visible': P6_visible,
            'P7_blind': P7_blind,
            'P7_visible': P7_visible,
            'P8_blind': P8_blind,
            'P8_visible': P8_visible,
            'P9_blind': P9_blind,
            'P9_visible': P9_visible,
            'P10_blind': P10_blind,
            'P10_visible': P10_visible,
            'P11_blind': P11_blind,
            'P11_visible': P11_visible,
            'P12_blind': P12_blind,
            'P12_visible': P12_visible,
        }
        
        # Set display options based on the condition
        if condition == 'blind':
            show_uncertainties = True
            show_uncertainty_categories = False
            show_success_labels = True
        elif condition == 'visible':
            # In 'visible' condition, categories are shown
            show_uncertainties = True
            show_uncertainty_categories = True
            show_success_labels = True
        else:
            # Default settings
            show_uncertainties = True
            show_uncertainty_categories = True
            show_success_labels = True

        # Always show category colors
        show_uncertainty_category_colors = True

        # no_more_demos = False
        continuous_demo_mode=False

        """
        LOAD IN THE PRE-TRAINED MODEL (IF REQUIRED)
        """
        if load_pretrained and not testing_flow and not training_flow:
            if pretrained_model_path is None:
                model_path = PARENT_DIR + '/models/' + task_name + '/' + method + '/max_demo_' + str(max_demo_num)
            else:
                model_path = pretrained_model_path
            
            # Checking if the step number is provided and loading the model
            if pretrained_model_step is not None:
                model_file = model_path + '/' + str(pretrained_model_step) + '_actor_critic.pth'
                print(f"Loading pre-trained model from: {model_file}")
                self.ac.load_state_dict(torch.load(model_file))
                self.ac_targ.load_state_dict(self.ac.state_dict())
                self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
                self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.p_lr, weight_decay=1e-4)
                self.q_optimizer = Adam(self.q_params, lr=self.lr)
            else:
                raise ValueError('pretrained_model_step must be specified when load_pretrained is True')
        # Set the start step based on the pre-trained model step
        if pretrained_model_step is not None:
            start_step = int(pretrained_model_step)
        else:
            start_step = 0

        # Initalising total_step before the call to test_agent
        self.total_step = start_step

        if load_pretrained and not testing_flow and not training_flow:
            if participant_code is not None:
                model_path = os.path.join(PARENT_DIR, 'models', task_name, f'Phase1_{participant_code}_{condition}')
            else:
                model_path = os.path.join(PARENT_DIR, 'models', task_name, 'Phase1' + condition)
        else:
            model_path = os.path.join(PARENT_DIR, 'models', task_name, method, f'max_demo_{max_demo_num}_{condition}')
        # # Saving the model
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        print("Testing policy right after loading the anchor policy")
        test_results = self.test_agent(task_name=task_name, participant_code=participant_code, model_step=start_step)
        print(f"Average success rate after loading model: {test_results['average_success_rate']}")

        # variables for demonstration and exploration tracking
        query_count = 0
        rollout_count = 0
        evaluate_res_per_step = []
        success_evaluate_res_per_step = []

        # Getting base env and disabling line drawing (for visualization)
        base_env = get_base_env(self.env)
        base_env.enable_line_drawing = False

        # Initialize uncertainties + logging them + setting default failure lable
        traj_uncertainty_per_position = self.initialize_starting_position_uncertainties(num_runs=base_runs_per_position)
        latest_logged_uncertainties = [traj_uncertainty_per_position[pos][-1] for pos in range(10)]
        success_per_position = [False] * 10

        # Updating the environment with the initial uncertainties
        if show_uncertainties:
            if show_uncertainty_categories:
                # Categorising the uncertainties
                categories = categorize_uncertainties(latest_logged_uncertainties)
                self.env.update_uncertainties(categories, success_per_position, show_categories=True, show_uncertainty_category_colors=show_uncertainty_category_colors, show_success_labels=show_success_labels)
            else:
                self.env.update_uncertainties(latest_logged_uncertainties, success_per_position, show_categories=False, show_uncertainty_category_colors=False, show_success_labels=show_success_labels)
        # Clearing lines after initial uncertainty rollouts
        base_env.enable_line_drawing = True

        # Quick reset before the experiment - to show the gripper in the correct starting position so it does not block the env visualization
        self.env.reset()
        # A DELAY BEFORE STARTING THE EXPERIMENT
        # input("Press Enter to start the experiment...") 

        # Countdown before starting exploration
        self.env.display_countdown(count=5)
        self.env.remove_exploration_text()
        self.env.add_exploration_text()
        self.env.add_demos_remaining_text(demos_remaining=max_queries - query_count, total_demos=max_queries)

        # Prepare for interaction with environment
        total_steps = self.epochs * self.steps_per_epoch
        # start_time = time.time()
        rollout_traj = []
        # Random starting position or preset to first position
        # chosen_position = choose_random_position(self.env)
        chosen_position = 0
        choose_starting_position(self.env, chosen_position)
        object_position = self.env.object_position
        # Reset to the chosen position
        state = self.env.reset(object_pos=object_position, object_color=self.env.object_color)
        done = True
        num_train_episodes = 0
        test_results_list = []

        """
        MAIN LOOP
            - Collect experience in the environment
            - Update the model
            - Log data
        """
        rollout_count = 0
        for t in range(start_step, total_steps):
            self.total_step = t + 1
            self.env.update_exploration_text_position()
            # Checks if the current episode has ended and ensures the rest is executed only after the start_step
            if done and t > start_step:
                rollout_count += 1

                if rollout_count == 10:
                    print("Testing policy after 10 rollouts, before first demo query")
                    test_results = self.test_agent(task_name=task_name, participant_code=participant_code, model_step=self.total_step)
                    print(f"Average success rate after 10 rollouts: {test_results['average_success_rate']}")
                """
                ANCHOR POLICY CONTINUATION: 
                    - load_pretrained: code path is taken only if a pretrained model is loaded
                    - testing_flow: Skips demo logic if the function is in testing mode (Needs adjustment here and in main.py) DONT USE
                    - training_flow: Skips demo logic if the function is in training mode (Needs adjustment here and in main.py) DONT USE
                """
                if continue_experiment_training and not testing_flow and not training_flow:
                    # provide_demo = to check if we continue providing demos in an 'online' fashion.
                    provide_demo = False
                    """
                    continuous_demo_mode:
                        - Controls whether demonstrations are provided continuously or based on a query count and interval.
                    """
                    if continuous_demo_mode:
                        if rollout_count % human_input_rollout_interval == 0:
                            provide_demo = True
                    else:
                        if query_count < max_queries and rollout_count % human_input_rollout_interval == 0:
                            provide_demo = True
                    """
                    HANDLING DEMONSTRATIONS:
                        - provide_demo: flag to check if a demonstration is provided in the current rollout.
                    """
                    if provide_demo:
                        self.env.reset(object_pos=object_position, object_color=object_color)
                        self.env.set_background_color("demonstration")
                        self.env.remove_exploration_text()
                        print(f"Current rollout count: {rollout_count}")
                        self.env.add_provide_start_state_text()
                        """
                        DEMONSTRATION LOGIC:
                            - continue_experiment_training: 
                                When True, the code focuses on selecting positions based on training criteria, e.g: selecting Highly Unexpected Success/Failure positions.
                                Otherwise: it defaults to choosing a starting position
                        """
                        if continue_experiment_training:
                            if selection_strategy in participant_positions:
                                positions_list = participant_positions[selection_strategy]
                                if query_count < len(P1_visible):
                                    chosen_position = positions_list[query_count]
                                    print(f"Selected position {chosen_position} from {selection_strategy} at query_count {query_count}")
                            combined_categories = []
                            for idx, cat in enumerate(categories):
                                success_text = 'Success' if success_per_position[idx] else 'Failure'
                                combined_category = f"{cat} {success_text}"
                                combined_categories.append(combined_category)

                            if selection_strategy == "Expectation-oriented":
                                # Select the position with the highest TD error
                                max_uncertainty = max(latest_logged_uncertainties)
                                positions_with_max_uncertainty = [idx for idx, val in enumerate(latest_logged_uncertainties) if val == max_uncertainty]
                                print(f"ALL TD ERRORS CHECK: {latest_logged_uncertainties}")
                                chosen_position = np.random.choice(positions_with_max_uncertainty)
                                print(f"Selected position {chosen_position} with highest TD error: {max_uncertainty}, (Category: {combined_categories[chosen_position]})")
                            elif selection_strategy == "Result-oriented":
                                # Select any starting state that failed
                                failed_positions = [idx for idx, success in enumerate(success_per_position) if not success]
                                if failed_positions:
                                    chosen_position = np.random.choice(failed_positions)
                                    print(f"Selected failed position {chosen_position}, TD error: {latest_logged_uncertainties[chosen_position]}, (Category: {combined_categories[chosen_position]})")
                                else:
                                    print("No failed positions available, selecting a random position.")
                                    chosen_position = np.random.choice(range(10))
                            elif selection_strategy == "Lowest TD error":
                                # Select the position with the lowest TD error
                                min_uncertainty = min(latest_logged_uncertainties)
                                positions_with_min_uncertainty = [idx for idx, val in enumerate(latest_logged_uncertainties) if val == min_uncertainty]
                                chosen_position = np.random.choice(positions_with_min_uncertainty)
                                print(f"Selected position {chosen_position} with lowest TD error: {min_uncertainty}, (Category: {combined_categories[chosen_position]})")
                        else:
                            # Default starting position selection
                            chosen_position = choose_starting_position(self.env)
                        
                        self.env.clear_lines()
                        self.env.clear_lines_gripper()
                        self.env.remove_provide_start_state_text()

                        # Load the correct demonstration trajectories by task
                        joystick_demo_path = os.path.join(ACTIVE_PANDA_DIR, 'demo_data', 'joystick_demo', task_name)
                        demo_state_trajs = np.genfromtxt(os.path.join(joystick_demo_path, 'demo_state_trajs.csv'), delimiter=' ')
                        demo_action_trajs = np.genfromtxt(os.path.join(joystick_demo_path, 'demo_action_trajs.csv'), delimiter=' ')
                        trajectories = trajectories_for_position(self, chosen_position, demo_state_trajs, demo_action_trajs)

                        query_count += 1
                        if not continue_experiment_training:
                            # query_count += 1
                            self.env.update_demos_remaining_text(demos_remaining=max_queries - query_count)
                        # Roll out the selected demo trajectory
                        for traj in trajectories:
                            for (state_demo, action_demo, reward_demo, next_state_demo, done_demo) in traj:
                                self.demo_buffer.store(state_demo, action_demo, reward_demo, next_state_demo, done_demo)
                                if done_demo:
                                    self.env.clear_lines_gripper()
                                    break
                        if log_data is not None:
                            latest_logged_uncertainties = [traj_uncertainty_per_position[pos][-1] for pos in traj_uncertainty_per_position]

                            categories = categorize_uncertainties(latest_logged_uncertainties)

                            selected_state_uncertainty = latest_logged_uncertainties[chosen_position]
                            selected_state_category = categories[chosen_position]
                            selected_state_success = success_per_position[chosen_position]

                            selected_state_category_with_success = f"{selected_state_category} {'Success' if selected_state_success else 'Failure'}"
                            categories_with_success = [f"{category} {'Success' if success_per_position[idx] else 'Failure'}" for idx, category in enumerate(categories)]

                            if testing_flow or training_flow:
                                log_entry = [None, condition, task_name, latest_logged_uncertainties, chosen_position,
                                                selected_state_uncertainty, selected_state_category_with_success, categories_with_success]
                            else:
                                log_entry = [participant_code, condition, task_name, latest_logged_uncertainties, chosen_position,
                                                selected_state_uncertainty, selected_state_category_with_success, categories_with_success, selection_strategy]
                            print(f"Logging data: {log_entry}")
                            log_data.append(log_entry)
                            
                        
                        # # UPDATE RIGHT AFTER DEMO
                        # batch = self.sample_mixed_batch()
                        # self.update(data=batch, update_timestep=t)

                        demo_rollout_traj = []
                        for traj in trajectories:
                            demo_rollout_traj.extend(traj)

                        # Estimate uncertainty from the demo rollout
                        traj_uncertainty = self.estimate_traj_uncertainty_td(demo_rollout_traj)
                        logged_traj_uncertainty = np.log2(float(traj_uncertainty))
                        traj_uncertainty_per_position[chosen_position].append(logged_traj_uncertainty)
                        chosen_position = 9

                        if show_uncertainties:
                            latest_logged_uncertainties = [traj_uncertainty_per_position[pos][-1] for pos in traj_uncertainty_per_position]
                            if show_uncertainty_categories:
                                categories = categorize_uncertainties(latest_logged_uncertainties)
                                self.env.update_uncertainties(categories, success_per_position, show_categories=True, show_uncertainty_category_colors=show_uncertainty_category_colors, show_success_labels=show_success_labels)
                            else:
                                self.env.update_uncertainties(latest_logged_uncertainties, success_per_position, show_categories=False, show_uncertainty_category_colors=False, show_success_labels=show_success_labels)
                        
                        if query_count >= max_queries:
                            """
                            continue_experiment_training:
                                 When False, the script does not continue training, and updates are performed and logged immediately
                            """
                            if not continue_experiment_training:
                                print("-------Max queries reached, ending trial.-----")
                                # Perform any remaining updates
                                if t > self.update_after:
                                    for _ in range(self.update_every):
                                        batch = self.sample_mixed_batch()
                                        self.update(data=batch, update_timestep=t)
                                # Save experiment model
                                if participant_code is not None:
                                    save_step = self.total_step
                                    if continue_experiment_training:
                                        model_filename = f'{participant_code}_{save_step}_actor_critic.pth'
                                        save_path = os.path.join(model_path, model_filename)
                                    else:
                                        save_path = os.path.join(model_path, f'{save_step}_actor_critic.pth')
                                    torch.save(self.ac.state_dict(), save_path)
                                    print(f"Final model saved at: {save_path}")
                                else:
                                    print("Participant code not provided; model not saved.")

                                self.env.remove_demos_remaining_text()

                """
                START NEW EPISODE
                """
                predefined_positions = self.env.task.predefined_obj_positions
                self.env.set_background_color("training")
                self.env.remove_exploration_text()
                self.env.add_exploration_text()
                if query_count >= max_queries:
                    # After initial demos, randomize starting positions
                    chosen_position = choose_random_position(self.env)
                else:
                    # During initial demos, keep starting positions ordered
                    chosen_position = (chosen_position + 1) % 10
                choose_starting_position(self.env, chosen_position)

                object_position = self.env.object_position
                object_color = self.env.colors[chosen_position % len(self.env.colors)]
                
                # # UPDATE RIGHT AFTER EXPLORATION ROLLOUT
                # batch = self.sample_mixed_batch()
                # self.update(data=batch, update_timestep=t)
                
                state = self.env.reset(object_pos=object_position, object_color=object_color)
                done = False
                rollout_traj = []
                # episode_score = 0
                num_train_episodes += 1

            # Collect experience
            reshaped_state = reconstruct_state(state)
            act = self.get_action(reshaped_state , deterministic=False)
            # self.env.add_exploration_text()
            # self.env.update_exploration_text_position()
            next_state, rew, done, info = self.env.step(act)
            reshaped_next_state = reconstruct_state(next_state)

            self.replay_buffer.store(reshaped_state.copy(), act, rew, reshaped_next_state.copy(), done)
            state = next_state

            # Append to trajectory
            rollout_traj.append((reshaped_state, act, rew, reshaped_next_state, float(done)))
            # episode_score += rew

            """
            UPDATE MODEL
            """
            if t > self.update_after and t % self.update_every == 0:
                for _ in range(self.update_every):
                    # batch = self.replay_buffer.sample_batch(self.batch_size)
                    batch = self.sample_mixed_batch()
                    self.update(data=batch, update_timestep=t)

            """
            END OF EPISODE
            """
            if done:
                self.env.clear_lines_gripper()

                success = info.get('is_success', False)
                success_per_position[chosen_position] = success

                traj_uncertainty = self.estimate_traj_uncertainty_td(rollout_traj)
                logged_traj_uncertainty = np.log2(float(traj_uncertainty))
                traj_uncertainty_per_position[chosen_position].append(logged_traj_uncertainty)
                categories = categorize_uncertainties(latest_logged_uncertainties)

                if show_uncertainties:
                    latest_logged_uncertainties = [traj_uncertainty_per_position[pos][-1] for pos in traj_uncertainty_per_position]
                    if show_uncertainty_categories:
                        categories = categorize_uncertainties(latest_logged_uncertainties)
                        self.env.update_uncertainties(categories, success_per_position, show_categories=True, show_uncertainty_category_colors=show_uncertainty_category_colors, show_success_labels=show_success_labels)
                    else:
                        self.env.update_uncertainties(latest_logged_uncertainties, success_per_position, show_categories=False, show_uncertainty_category_colors=False, show_success_labels=show_success_labels)
                
                rollout_traj = []


            """
            END OF EPOCH
            """
            if (t + 1) % self.steps_per_epoch == 0:
                epoch = (t + 1) // self.steps_per_epoch

                model_step = self.total_step
                test_results = self.test_agent(
                    task_name=task_name,
                    evaluation_res=evaluate_res_per_step,
                    success_evaluation_res=success_evaluate_res_per_step,
                    participant_code=participant_code,
                    model_step=model_step
                )
                test_results_list.append(test_results)
                results_df = pd.DataFrame(test_results_list)
                results_df.to_csv(f'test_results_{participant_code}.csv', index=False)
                
            if (t + 1) % model_saving_interval == 0:
                save_step = self.total_step
                if participant_code is not None:
                    model_filename = f'{participant_code}_{save_step}_actor_critic.pth'
                    save_path = os.path.join(model_path, model_filename)
                else:
                    save_path = os.path.join(model_path, f'{save_step}_actor_critic.pth')
                torch.save(self.ac.state_dict(), save_path)
                print(f"Model saved at: {save_path}")

        self.env.close()
        self.test_env.close()
        results_df = pd.DataFrame(test_results_list)
        results_df.to_csv(f'test_results_{participant_code}.csv', index=False)

        return evaluate_res_per_step, success_evaluate_res_per_step
