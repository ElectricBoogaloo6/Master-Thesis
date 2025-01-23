import os
import copy
import random
from collections import deque
from typing import Deque, Dict, List, Tuple
import time

import gym
from gym import Wrapper
import panda_gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3 import SAC, DDPG
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import TQC
import argparse
from datetime import datetime
import sys
from utils import ReplayBuffer, OUNoise, Actor, Critic, ActionNormalizer, ResetWrapper, TimeFeatureWrapper, TimeLimitWrapper, reconstruct_state, save_results
from phase_1_func import choose_starting_position, CustomResetWrapper, choose_task, choose_random_position, trajectories_for_position

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
ACTIVE_PANDA_DIR = os.path.dirname(CURRENT_DIR)
ENVS_DIR = os.path.join(ACTIVE_PANDA_DIR, 'envs')
sys.path.append(ENVS_DIR)
from task_envs import ReachWithObstacleV0Env, ReachWithObstacleV1Env, PushWithObstacleV0Env, PushWithObstacleV1Env, Phase1TaskCentreEnv, Phase1TaskLeftEnv, Phase1TaskRightEnv



class Phase1_DDPGfD_BC_Agent:
    def __init__(
            self,
            env: gym.Env, # openAI Gym env
            test_env: gym.Env,
            task_name,
            method,
            demo_pool,
            demo_pool_inits,
            memory_size: int,
            batch_size: int,
            demo_batch_size: int,
            max_demo_num: int,
            ou_noise_theta: float,
            ou_noise_sigma: float,
            writer,
            demo_info = None,
            gamma: float = 0.99,
            tau: float = 0.005,
            initial_random_steps: int = 1e4,
            # loss parameters
            lambda1: float = 1e-3,
            lambda2: float = 1.0,
            mode='scratch',
            current_condition=None,
            log_data=None,
            participant_code=None,
            testing_flow=False,
            training_flow=False,
    ):
        """
        Initialization of the agent
        """
        # KM: dimension of the observation space and action space.
        obs_dim = env.observation_space['observation'].shape[0] + 3  # exclude the time feature and add goal position
        action_dim = env.action_space.shape[0]

        self.env = env
        self.test_env = test_env
        self.task_name = task_name
        self.method = method
        self.batch_size = batch_size
        self.demo_batch_size = demo_batch_size
        self.max_demo_num = max_demo_num
        self.gamma = gamma
        self.tau = tau
        self.initial_random_steps = initial_random_steps

        self.demo_pool = demo_pool # a list of trajectories, each trajectory itself is a list of tuple
        self.demo_pool_inits = demo_pool_inits # a list of initial states of each demo trajectory, each initial state is a 1d np array

        # loss parameters
        self.lambda1 = lambda1
        self.lambda2 = lambda2 / demo_batch_size

        # Condition tracking
        self.current_condition = current_condition

        # buffer
        self.memory = ReplayBuffer(obs_dim, action_dim, memory_size, batch_size)

        # demo buffer
        demo_memory_size = max_demo_num * 1000 # calculate size of the demo replay buffer
        # initialize the buffer object with size and other params
        self.demo_memory = ReplayBuffer(obs_dim, action_dim, demo_memory_size, demo_batch_size)
        # KM: loading up warm-up demos if provided
        if demo_info is not None:
            print("Going to load warm-up demo into demo replay buffer")
            demo = demo_info['demo'] # a list of tuples (state, action, reward, next_state, done)
            print("loaded warm-up demo steps: {}".format(len(demo)))
            self.teacher_demo_num = demo_info['demo_num']
            self.demo_memory.extend(demo)
            print("after loading: demo memory size is {}".format(len(self.demo_memory)))
        else:
            print("No warm-up demo provided")
            self.teacher_demo_num = 0

        # noise
        self.noise = OUNoise(
            action_dim,
            theta=ou_noise_theta,
            sigma=ou_noise_sigma
        )

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # networks
        self.mode = mode
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        print("Actor initialized successfully.")

        self.critic = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        print("Critic initialized successfully.")

        # optimizer for actor and critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3, weight_decay=1e-5)

        # transition to store in memory
        self.transition = list()

        # logger to track training process
        self.writer = writer

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select an action from the input state
        """
        # KM: condition checks if the total number of steps taken is less than number of initial random steps and if the agent is not in test mode.
        if self.total_step < self.initial_random_steps and not self.is_test:
            # print("Taking random action")
            # KM: if the condition is true, the agent selects a random action from the action space.
            selected_action = np.random.uniform(self.env.action_space.low, self.env.action_space.high)
            # print(f"Random action selected: {selected_action}")
        else:
            # KM: Policy based action - if the initial random steps are completed OR the agent is in test_mode.
                # passes the state through the actor to get the action. converts the state to a PyTorch tensor, moves it to device, passes it through the actor, detaches the result from computation graph, moves it back to device
                    # AND converts it to a numpy array.
            # print("Taking policy based action")
            selected_action = self.actor(torch.FloatTensor(state).to(self.device)).detach().cpu().numpy()

        # add noise for exploration during training
        if not self.is_test:
            noise = self.noise.sample()
            selected_action = np.clip(selected_action + noise, -1.0, 1.0)
            # print(f"Action with noise: {selected_action}")

        # KM: line stores the current state and the selected action in self.transition. This is useful for storing the transition later in the replay buffer.
        self.transition = [state, selected_action]
        # print(f"TRANSITION IN STEP FUNCTION: {self.transition}")

        return selected_action
    
    def step(self, action: np.ndarray, role='learner') -> Tuple[np.ndarray, np.float64, bool]:
        """
        Take an action and return the response of the environment.
        Tuple: 
            np.ndarray = numpy representing next state of env after action
            np.float = reward received after taking action
            bool = boolean indicating if the episode is done.
        """
        # print(f"Taking step with action: {action}")
        next_state, reward, done, info = self.env.step(action)
        # print(f"Step result -> next_state: {next_state}, reward: {reward}, done: {done}, info: {info}")
        # KM: reconstructs the state to have be an numpy array of [obs, goal]
        reshaped_next_state = reconstruct_state(next_state)
        # print(f"Reshaped next_state in step: {reshaped_next_state}")
        # print(f"Original next_state shape: {next_state['observation'].shape} in STEP")
        # print(f"Reshaped next_state shape: {reshaped_next_state.shape} in STEP")

        # KM: only if agent is in training mode, not testing
        if not self.is_test:
            # # KM: POTENTIAL FIX: the initial observation is also reshaped to match the shape of the next state
            # reshaped_obs = reconstruct_state({'observation': self.transition[0], 'desired_goal': next_state['desired_goal']})
            # # KM: The reward, reshaped next state, and done flag are added to self.transition.
            # # self.transition += [reward, reshaped_next_state, done]
            # self.transition = [reshaped_obs, action, reward, reshaped_next_state, done]
            # print("Before storing transition")
            self.transition += [reward, reshaped_next_state, done]
            # print(f"Transition in step: {self.transition}")
            if role == 'learner':
                self.memory.store(*self.transition)
                # print("Stored transition in memory.")
            else:
                # self.demo_memory.store_priority(self.teacher_demo_num + 1)
                self.demo_memory.store(*self.transition)
            # print("Stored transition in memory.")


            # ######### KM: BELOW IS CAUSING AN ERROR, STORAGE TO MEMORY NEEDS TO BE FIXED ACROSS THE MODEL #########: 
            #         # ValueError: could not broadcast input array from shape (18,) into shape (21,) 
            # if role == 'learner':
            #     # KM: the transition is stored in self.memory, which is the agent's main replay buffer.
            #     print(f"Storing transition with observation shape: {self.transition[0].shape}")
            #     self.memory.store(*self.transition)
            # else:
            #     # KM: if the role is teacher, the transition is stored in self.demo_memory, which is the buffer for demonstration data
            #     self.demo_memory.store(*self.transition)
        
        if done:
            if role == 'teacher':
                if info['is_success']:
                    print("teacher demo: success!")
                else:
                    print("teacher demo: failure!")
            else:
                if info['is_success']:
                    print(f"rollout done: success! {reward}")
                elif info['whether_collision']:
                    print(f"rollout done: collision! {reward}")
                else:
                    print(f"rollout done: time out! {reward}")

        return next_state, reward, done
    
    # def update_model(self) -> torch.Tensor:
    #     device = self.device

    #     # Sample from replay buffer
    #     samples = self.memory.sample_batch()
    #     state = torch.FloatTensor(samples["obs"]).to(device)
    #     next_state = torch.FloatTensor(samples["next_obs"]).to(device)
    #     action = torch.FloatTensor(samples["acts"]).to(device)
    #     reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
    #     done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

    #     # Initialize lists for combining data
    #     all_states = [state]
    #     all_next_states = [next_state]
    #     all_actions = [action]
    #     all_rewards = [reward]
    #     all_dones = [done]

    #     # Sample from demo buffer if available
    #     if len(self.demo_memory) >= self.demo_batch_size:
    #         d_samples = self.demo_memory.sample_batch()
    #         d_state = torch.FloatTensor(d_samples["obs"]).to(device)
    #         d_next_state = torch.FloatTensor(d_samples["next_obs"]).to(device)
    #         d_action = torch.FloatTensor(d_samples["acts"]).to(device)
    #         d_reward = torch.FloatTensor(d_samples["rews"].reshape(-1, 1)).to(device)
    #         d_done = torch.FloatTensor(d_samples["done"].reshape(-1, 1)).to(device)

    #         # Append to combined lists
    #         all_states.append(d_state)
    #         all_next_states.append(d_next_state)
    #         all_actions.append(d_action)
    #         all_rewards.append(d_reward)
    #         all_dones.append(d_done)

    #         # Compute behavior cloning loss without masking
    #         pred_action = self.actor(d_state)
    #         bc_loss = F.mse_loss(pred_action, d_action)
    #     else:
    #         bc_loss = torch.zeros(1, device=device)

    #     # Combine all data
    #     all_states = torch.cat(all_states, dim=0)
    #     all_next_states = torch.cat(all_next_states, dim=0)
    #     all_actions = torch.cat(all_actions, dim=0)
    #     all_rewards = torch.cat(all_rewards, dim=0)
    #     all_dones = torch.cat(all_dones, dim=0)
    #     all_masks = 1 - all_dones

    #     # Compute target values
    #     next_actions = self.actor_target(all_next_states)
    #     next_values = self.critic_target(all_next_states, next_actions)
    #     curr_returns = all_rewards + self.gamma * next_values * all_masks
    #     curr_returns = curr_returns.detach()

    #     # Critic loss
    #     values = self.critic(all_states, all_actions)
    #     critic_loss = F.mse_loss(values, curr_returns)

    #     # Update critic
    #     self.critic_optimizer.zero_grad()
    #     critic_loss.backward()
    #     self.critic_optimizer.step()

    #     # Policy gradient loss
    #     pg_loss = -self.critic(state, self.actor(state)).mean()

    #     # Total actor loss
    #     actor_loss = self.lambda1 * pg_loss + self.lambda2 * bc_loss

    #     # Update actor
    #     self.actor_optimizer.zero_grad()
    #     actor_loss.backward()
    #     self.actor_optimizer.step()

    #     # Soft update of target networks
    #     self._target_soft_update()

    #     return actor_loss.data, critic_loss.data






    def update_model(self) -> torch.Tensor:
        """
        Updates the Actor and Critic networks using gradient descent 
        based on samples from the replay buffer and demo buffer.
        """
        device = self.device  # for shortening the following lines
        # KM: sample a batch of transitions from the replay buffer
        # KM: retrieves a batch of experiences from replay buffer.
        samples = self.memory.sample_batch()
        # Each component (state, next_state, action, reward, done) is extracted from samples and converted to PyTorch tensors.
            # .to(device): Moves the tensors to the device (CPU or GPU).
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.FloatTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # sample from demo buffer (if available)
        # KM: Ensures there are enough samples in the demo buffer.
        if len(self.demo_memory) >= self.demo_batch_size:
            # print(f"Demo memory contains {len(self.demo_memory)} samples, sampling a batch of size {self.demo_batch_size}.")
            # KM: retrieves a batch from the demo buffer
            d_samples = self.demo_memory.sample_batch()
            # Each component (state, next_state, action, reward, done) is extracted from samples and converted to PyTorch tensors.
            # .to(device): Moves the tensors to the device (CPU or GPU).
            d_state = torch.FloatTensor(d_samples["obs"]).to(device)
            d_next_state = torch.FloatTensor(d_samples["next_obs"]).to(device)
            d_action = torch.FloatTensor(d_samples["acts"]).to(device)
            d_reward = torch.FloatTensor(d_samples["rews"].reshape(-1, 1)).to(device)
            d_done = torch.FloatTensor(d_samples["done"].reshape(-1, 1)).to(device)
        # else:
        #     print(f"Demo memory contains {len(self.demo_memory)} samples, which is less than demo_batch_size ({self.demo_batch_size}). Skipping demo sampling.")

        # KM: masks to handle non-terminal states.
        masks = 1 - done
        # KM: predicts the next action using the target actor.
        next_action = self.actor_target(next_state)
        # KM: evaluates the value of the next state-action pair.
        next_value = self.critic_target(next_state, next_action)
        # KM: calculates the target value (return) for the current state-action pair.
        curr_return = reward + self.gamma * next_value * masks
        # KM: detaches curr_return from the computation graph to prevent backpropagation through it.
        curr_return = curr_return.to(device).detach()

        # KM: Train critic network
        # KM: predicts the value of the current state-action pair.
        values = self.critic(state, action)
        # KM: computes the mean squared error loss between the predicted values and the target values.
        critic_loss = F.mse_loss(values, curr_return)
        # KM: logs the critic loss for visualization.
        self.writer.add_scalar(tag='loss/critic_loss', scalar_value=critic_loss,
                               global_step=self.total_step)
        # KM: Resets the gradients.
        self.critic_optimizer.zero_grad()
        # KM: Computes the gradients.
        critic_loss.backward()
        # KM: Updates the critic network parameters.
        self.critic_optimizer.step()

        # Train actor network
        # KM: computes the loss based on the predicted action values, encouraging actions that increase the predicted value.
        pg_loss = -self.critic(state, self.actor(state)).mean()

        # Behavior Cloning (BC) Loss
        # KM: Ensures there are enough samples in the demo buffer.
        if len(self.demo_memory) >= self.demo_batch_size:
            # KM: predicts actions for the demo states.
            pred_action = self.actor(d_state)
            # KM: mask that identifies where the predicted actions are worse than the demo actions.
            # KM: Compares Q-values of demo actions and predicted actions.
            qf_mask = torch.gt(self.critic(d_state, d_action), self.critic(d_state, pred_action),).to(device)
            # KM: Converts the mask to a float tensor.
            qf_mask = qf_mask.float()
            # KM: Counts the number of positive entries in the mask.
            n_qf_mask = int(qf_mask.sum().item())
            """
            BC Loss:
            """
            # KM: If there are no positive entries, set bc_loss to zero.
            if n_qf_mask == 0:
                bc_loss = torch.zeros(1, device=device)
            # KM: Otherwise, calculate bc_loss as the mean squared error between the masked predicted actions and demo actions.
            else:
                bc_loss = (
                            torch.mul(pred_action, qf_mask) - torch.mul(d_action, qf_mask)
                          ).pow(2).sum() / n_qf_mask
                
            # print(f"Behavior Cloning Loss (bc_loss): {bc_loss.item()} at step {self.total_step}")
        # KM: If there are not enough samples in the demo buffer, set bc_loss to zero.
        else:
            bc_loss = torch.zeros(1, device=device)

        # KM: combines the policy gradient loss and behavior cloning loss.
        actor_loss = self.lambda1 * pg_loss + self.lambda2 * bc_loss

        # KM: Logs the behavior cloning loss, policy gradient loss, and combined actor loss.
        self.writer.add_scalar(tag='loss/bc_loss', scalar_value=self.lambda2 * bc_loss,
                               global_step=self.total_step)
        self.writer.add_scalar(tag='loss/policy_gradient_loss', scalar_value=self.lambda1 * pg_loss,
                               global_step=self.total_step)
        self.writer.add_scalar(tag='loss/actor_loss', scalar_value=actor_loss,
                               global_step=self.total_step)
        
        # KM: Backward Pass and Optimization:
        # KM: Resets the gradients.
        self.actor_optimizer.zero_grad()
        # KM: Computes the gradients.
        actor_loss.backward()
        # KM: Updates the actor network parameters.
        self.actor_optimizer.step()

        # KM: Soft update of the target networks.
            # softly update the target networks using the current networks' parameters.
        self._target_soft_update()

        return actor_loss.data, critic_loss.data
    
    def train(self,
              num_frames: int,
              plotting_interval: int =200,
              num_eval_episodes: int = 10,
              model_saving_interval=5000,
              base_runs_per_position=1,
              human_input_rollout_interval=10,
              satisfactory_score=200,
              max_queries=5,
              current_condition="Baseline",
              log_data = None,
              participant_code=None,
              testing_flow=False,
              training_flow=False,):
        
        print(f"Current condition: ", current_condition) 
        # KM: MODIFY ENV BEHAVIOUR WITH DIFFERENT CONDITIONS
        if current_condition == "Baseline":
            show_uncertainties = True
        elif current_condition == "Condition 1":
            show_uncertainties = True
        elif current_condition == "Condition 2":
            show_uncertainties = True
        elif current_condition == None:
            show_uncertainties = True


        """
        Train the agent
        """
        self.is_test = False
        query_count = 0

        # KM: lists for storing evaluation results, uncertainty history, initial state history, and demonstration positions.
        evaluate_res_per_step = []
        success_evaluate_res_per_step = []
        # traj_uncertainty_per_step = []  # List to store TD errors per rollout
        actor_losses, critic_losses, scores = [], [], []
        score = 0

        # Initialize base uncertainties
        # dict with each position a list of uncertanties
        traj_uncertainty_per_position = self.initialize_starting_position_uncertainties(num_runs=base_runs_per_position)
        for position, uncertainties in traj_uncertainty_per_position.items():
            avg_uncertainty = np.mean(uncertainties) if uncertainties else float('inf')
            print(f"Starting position {position} - Average Uncertainty: {avg_uncertainty}")

        if show_uncertainties:
            latest_logged_uncertainties = [traj_uncertainty_per_position[pos][-1] for pos in range(10)]
            self.env.update_uncertainties(latest_logged_uncertainties)

        # Initialize max and min uncertainties per position
        max_uncertainty_per_position = {pos: -float('inf') for pos in range(10)}
        min_uncertainty_per_position = {pos: float('inf') for pos in range(10)}

        demo_path = PARENT_DIR + '/' + 'demo_data/visualization/' + self.task_name + '/' + self.method + '/max_demo_' + str(
            self.max_demo_num) + '/' + self.mode + '/'
        if not os.path.exists(demo_path):
            os.makedirs(demo_path)
        # saving path of the models
        current_dir = os.getcwd()
        parent_dir = os.path.dirname(current_dir)
        model_path = parent_dir + '/' + 'models/target_tasks/' + self.task_name + '/' + self.method + '/max_demo_' + str(
            self.max_demo_num) + '/' + self.mode + '/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)


        """
        Main training loop:
        """
        rollout_count = 0 

        while self.total_step <= num_frames:
            # print(f"Step: {self.total_step}")
            # KM: Render the environment
            self.env.render("rgb_array")

            if testing_flow == False and training_flow == False:
                if query_count < max_queries and rollout_count % human_input_rollout_interval == 0:
                    print(f"Current rollout count: {rollout_count}")
                    chosen_position = choose_starting_position(self.env)

                    # Load the correct demonstration trajectories by task
                    joystick_demo_path = os.path.join(ACTIVE_PANDA_DIR, 'demo_data', 'joystick_demo', self.task_name)
                    demo_state_trajs = np.genfromtxt(os.path.join(joystick_demo_path, 'demo_state_trajs.csv'), delimiter=' ')
                    demo_action_trajs = np.genfromtxt(os.path.join(joystick_demo_path, 'demo_action_trajs.csv'), delimiter=' ')
                    trajectories = trajectories_for_position(self, chosen_position, demo_state_trajs, demo_action_trajs)

                    query_count += 1

                    # Roll out the selected demo trajectory
                    for traj in trajectories:
                        for (state, action, reward, next_state, done) in traj:
                            self.demo_memory.store(state, action, reward, next_state, done)
                            if done:
                                break

                    if log_data is not None:
                        # log
                        # avg_uncertainties = [np.mean(traj_uncertainty_per_position[pos]) if traj_uncertainty_per_position[pos] else float('inf') for pos in range(10)]
                        logged_uncertainties = [traj_uncertainty_per_position[pos][-1] for pos in range(10)]
                        # selected_state_uncertainty = avg_uncertainties[chosen_position]
                        selected_state_uncertainty = logged_uncertainties[chosen_position]

                        if testing_flow or training_flow:
                            log_entry = [None, self.task_name, current_condition, logged_uncertainties, chosen_position, selected_state_uncertainty]
                        else:
                            log_entry = [participant_code, self.task_name, current_condition, logged_uncertainties, chosen_position, selected_state_uncertainty]

                        print(f"Logging data: {log_entry}")
                        log_data.append(log_entry)

                    if query_count >= max_queries:
                        print("-------Max queries reached, ending trial.-----")
                        break
                else:
                    chosen_position = choose_random_position(self.env)
                    choose_starting_position(self.env, chosen_position)

                object_position = self.env.object_position
            
            elif training_flow:
                if rollout_count % 5 == 0:
                    print(f"Current rollout count: {rollout_count}")
                    chosen_position = np.random.randint(0, 10)
                    chosen_position = choose_starting_position(self.env, chosen_position)
                
                    # Load the correct demonstration trajectories by task
                    joystick_demo_path = os.path.join(ACTIVE_PANDA_DIR, 'demo_data', 'joystick_demo', self.task_name)
                    demo_state_trajs = np.genfromtxt(os.path.join(joystick_demo_path, 'demo_state_trajs.csv'), delimiter=' ')
                    demo_action_trajs = np.genfromtxt(os.path.join(joystick_demo_path, 'demo_action_trajs.csv'), delimiter=' ')
                    trajectories = trajectories_for_position(self, chosen_position, demo_state_trajs, demo_action_trajs)

                    # Roll out the selected demo trajectory
                    for traj in trajectories:
                        for (state, action, reward, next_state, done) in traj:
                            self.demo_memory.store(state, action, reward, next_state, done)
                            if done:
                                break

                else:
                    chosen_position = choose_random_position(self.env)
                    choose_starting_position(self.env, chosen_position)

                object_position = self.env.object_position

            # start a new episode
            state = self.env.reset()
            # print(f"Initial state: {state}")
            # print(f"Initial state shape: {state['observation'].shape} in TRAIN")
            # Copies the initial state for later use.
            init_state = state.copy()
            # Sets done to False to start the episode.
            done = False
            # Initializes a list to store the trajectory of the rollout.
            rollout_traj = []
            episode_score = 0
            """
            Rollout the Agent Policy:
            """
            while not done:
                # Reshapes the state.
                reshaped_state = reconstruct_state(state)
                # print(f"Reshaped state shape: {reshaped_state.shape} in TRAIN")
                # Selects an action based on the reshaped state.
                action = self.select_action(reshaped_state)
                # print(f"Action selected: {action}")
                # Takes a step in the environment, storing the transition in the replay buffer.
                # print("Before taking step in environment")
                next_state, reward, done = self.step(action) # transition was stored in the replay buffer inside the function
                # print(f"Step taken. Next state: {next_state}, Reward: {reward}, Done: {done}")
                # print("Before reshaping next state")
                reshaped_next_state = reconstruct_state(next_state)
                # print(f"Reshaped next state: {reshaped_next_state}")
                # print(f"Reshaped next_state shape: {reshaped_next_state.shape}")
                # Updates the state and adds the transition to the rollout trajectory.
                rollout_traj.append((reshaped_state, action, reward, reshaped_next_state, float(done)))
                # print("Before assigning next state")
                state = next_state
                # print(f"Next state assigned: {state}")
                episode_score += reward
                """
                Training and evaluation:
                """
                # KM: If the replay buffer has enough samples and the initial random steps are completed, updates the model.
                if (len(self.memory) >= self.batch_size
                    and self.total_step > self.initial_random_steps):
                    # print("Updating model at step:", self.total_step)
                    actor_loss, critic_loss = self.update_model()
                    # print(f"Actor loss: {actor_loss}, Critic loss: {critic_loss}")
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)


                # plotting
                if self.total_step % plotting_interval == 0:
                    # KM: Evaluates the policy and stores the results.
                    average_episode_reward, evaluate_res_per_step, average_success_rate, success_evaluate_res_per_step = self.evaluate_policy(
                        evaluation_res=evaluate_res_per_step, success_evaluation_res=success_evaluate_res_per_step,
                        env_step=self.total_step, test_episode_num=1, total_seed_num=20)
                    # average_episode_reward, evaluate_res_per_step, average_success_rate, success_evaluate_res_per_step = self.evaluate_policy(
                    # evaluate_res_per_step, success_evaluate_res_per_step, self.total_step, num_eval_episodes
                    # )
                    print(f"Average episode reward: {average_episode_reward}, Average success rate: {average_success_rate}")
                    self.writer.add_scalar(tag='episode_reward/train/per_env_step', scalar_value=average_episode_reward,
                                        global_step=self.total_step)
                    self.writer.add_scalar(tag='success_rate/train/per_env_step', scalar_value=average_success_rate,
                                        global_step=self.total_step)
                    print("[{} environment steps finished]: Average episode reward is {}".format(self.total_step,
                                                                                                average_episode_reward))
                    # # Plot training progress
                    # self._plot(self.total_step, scores, actor_losses, critic_losses)

                # KM: If the total step is a multiple of the model saving interval, saves the model.
                if self.total_step % model_saving_interval == 0:
                    torch.save(self.actor.state_dict(), model_path + str(self.total_step) + '_' + 'actor' + '.pth')
                    torch.save(self.actor_target.state_dict(), model_path + str(self.total_step) + '_' + 'actor_target' + '.pth')
                    torch.save(self.critic.state_dict(), model_path + str(self.total_step) + '_' + 'critic' + '.pth')
                    torch.save(self.critic_target.state_dict(), model_path + str(self.total_step) + '_' + 'critic_target' + '.pth')

                self.total_step += 1
                # print(f"Next step: {self.total_step}")
                scores.append(episode_score)  # Append episode score to scores list

            # Increment rollout count after each rollout
            rollout_count += 1
            # print(f"Rollout count: {rollout_count}")

            # After each rollout, calculate the TD error for the trajectory
            traj_uncertainty = self.estimate_traj_uncertainty_td(rollout_traj)
            # Convert the TD error to a log scale
            logged_traj_uncertainty = np.log2(float(traj_uncertainty))
            # Append the logged uncertainty to the list for the chosen position
            traj_uncertainty_per_position[chosen_position].append(logged_traj_uncertainty)
            # Uncertainty for the current trajectory, appended to uncertainties for the chosen position

            # # Update max and min uncertainties
            # if traj_uncertainty > max_uncertainty_per_position[chosen_position]:
            #     max_uncertainty_per_position[chosen_position] = traj_uncertainty
            # if traj_uncertainty < min_uncertainty_per_position[chosen_position]:
            #     min_uncertainty_per_position[chosen_position] = traj_uncertainty

            # Update max and min uncertainties based on the logged value
            if logged_traj_uncertainty > max_uncertainty_per_position[chosen_position]:
                max_uncertainty_per_position[chosen_position] = logged_traj_uncertainty
            if logged_traj_uncertainty < min_uncertainty_per_position[chosen_position]:
                min_uncertainty_per_position[chosen_position] = logged_traj_uncertainty

            # Print uncertainties every 1000 rollouts
            if rollout_count % 1000 == 0:
                print("Upper and Lower bounds of uncertainties for each starting state:")
                for pos in range(10):
                    print(f"Position {pos}: Upper bound = {max_uncertainty_per_position[pos]}, Lower bound = {min_uncertainty_per_position[pos]}")

            if show_uncertainties:
                # Update the uncertainties displayed in the environment
                # avg_uncertainties = [np.mean(traj_uncertainty_per_position[pos]) if traj_uncertainty_per_position[pos] else float('inf') for pos in range(10)]
                latest_logged_uncertainties = [traj_uncertainty_per_position[pos][-1] for pos in range(10)]
                self.env.update_uncertainties(latest_logged_uncertainties)
                # self.env.update_uncertainties(avg_uncertainties)



        # Save the final upper and lower bounds to a file
        bounds_path = model_path + 'uncertainty_bounds.txt'
        with open(bounds_path, 'w') as f:
            for pos in range(10):
                f.write(f"Position {pos}: Upper bound = {max_uncertainty_per_position[pos]}, Lower bound = {min_uncertainty_per_position[pos]}\n")

        self.env.close()
        return evaluate_res_per_step, success_evaluate_res_per_step, traj_uncertainty_per_position
    
    def evaluate_policy(self,
                        evaluation_res,
                        success_evaluation_res,
                        env_step,
                        test_episode_num=1,
                        total_seed_num=20):
        """
        Evaluates the performance of the agent by running it through multiple test episodes 
        and computing the average episode reward and success rate.
        - evaluation_res: A list to store the evaluation results.
        - success_evaluation_res: A list to store the success rates of the evaluation.
        - The current environment step, used to track evaluation progress.
        - test_episode_num: The number of test episodes to run for each seed (default is 1).
        - total_seed_num: The number of different seeds to use for evaluation (default is 20).
        """
        # KM: Accumulates the total reward over all test episodes.
        test_episode_reward = 0.0
        # KM: Counts the number of successful episodes.
        total_success_num = 0.0
        # KM: Lists to store the rewards and success statuses. They start with the current env_step
        res = []
        success_res = []
        res.append(env_step)
        success_res.append(env_step)

        """
        Generating Goal and Object Positions:
        """
        ############ KM: CONSIDER HOW THIS AFFECT THE OBJECT POSITIONS IN CUSTOM TASKS ############
        # KM: Loops over total_seed_num different seeds.
        print(f"Starting policy evaluation")
        goal_position = self.test_env.task.goal_position
        object_positions = self.test_env.task.predefined_obj_positions

        # Iterate over all seeds
        for seed in range(total_seed_num):
            np.random.seed(seed)  # Ensure reproducibility for each seed
            # Shuffle the object positions to test different positions in each evaluation round
            np.random.shuffle(object_positions)

            # # Iterate over the predefined object positions
            # for object_position in object_positions:  # Ensure every position is encountered
            #     # Reset the environment with the current goal and object positions
            #     state_ = self.test_env.reset(whether_random=False, goal_pos=goal_position, object_pos=object_position)
            #     done_ = False
            #     episode_reward = 0.0
            #     whether_success = 0.0

            # Iterate over the predefined object positions
            for i in range(min(test_episode_num, len(object_positions))):
                object_position = object_positions[i]  # Select the i-th object position
                # Reset the environment with the current goal and object positions
                state_ = self.test_env.reset(whether_random=False, goal_pos=goal_position, object_pos=object_position)
                done_ = False
                episode_reward = 0.0
                whether_success = 0.0

                while not done_:
                    state_ = reconstruct_state(state_)
                    action_ = self.actor(torch.FloatTensor(state_).to(self.device)).detach().cpu().numpy()
                    next_state_, reward_, done_, info_ = self.test_env.step(action_)
                    episode_reward += reward_
                    test_episode_reward += reward_
                    state_ = next_state_

                if info_['is_success']:
                    total_success_num += 1.0
                    whether_success = 1.0

                res.append(episode_reward)
                success_res.append(whether_success)

        # Appends the results (res and success_res) to the evaluation lists.
        evaluation_res.append(res)
        # Computes the average_episode_reward over all test episodes and seeds.
        average_episode_reward = test_episode_reward / (test_episode_num * total_seed_num)
        success_evaluation_res.append(success_res)
        average_success_rate = total_success_num / (test_episode_num * total_seed_num)

        return average_episode_reward, evaluation_res, average_success_rate, success_evaluation_res

    def estimate_traj_uncertainty_td(self, traj):
        """
        estimates the uncertainty of a trajectory
          by calculating the mean temporal difference (TD) error over the trajectory. 
        The TD error is a measure of the difference between the predicted value and the actual
          return of a state-action pair.
        - traj: A trajectory, which is a list of tuples. Each tuple contains (state, action, reward, next_state, done), representing the steps taken in an episode.
        """
        # Accumulates the absolute TD errors over the trajectory.
        cumulated_td_error = 0.0
        #  The number of steps in the trajectory.
        episode_length = len(traj)
        # Loop through each step in the trajectory:
        for state, action, reward, next_state, done in traj:
            # Each step is unpacked into its components
               # The state, action and next_state are reshaped and converted to PyTorch tensors and moved to the device (CPU/GPU).
            state = torch.FloatTensor(state.reshape(-1, state.shape[0])).to(self.device)
            next_state = torch.FloatTensor(next_state.reshape(-1, next_state.shape[0])).to(self.device)
            action = torch.FloatTensor(action.reshape(-1, action.shape[0])).to(self.device)

            # Calculate next action using the target actor
            next_action = self.actor_target(next_state)
            # Estimate the value of the next state-action pair using the target critic
            next_value = self.critic_target(next_state, next_action).detach().item()
            # The return is calculated as the immediate reward plus the discounted value of the next_state (self.gamma is the discount factor).
                # If the episode is done, (done is True), the value of the next_state is not added ((1 - done) becomes 0).
            curr_return = reward + self.gamma * next_value * (1 - done)
            # Estimate the value of the current state-action pair using the critic
            values = self.critic(state, action).detach().item()
            # Calculate the TD error
            td_error = curr_return - values
            # Accumulate the absolute TD error
            cumulated_td_error += abs(td_error)
        # Calculate the average TD error (trajectory uncertainty)
        traj_uncertainty = cumulated_td_error / episode_length

        return traj_uncertainty
    
    def initialize_starting_position_uncertainties(self, num_runs=10):
        """
        Initialize base uncertainties for each starting position by running multiple rollouts.
        """
        traj_uncertainty_per_position = {pos: [] for pos in range(10)}
        for position in range(10):
            for _ in range(num_runs):
                choose_starting_position(self.env, position)
                state = self.env.reset()
                done = False
                rollout_traj = []

                while not done:
                    reshaped_state = reconstruct_state(state)
                    action = self.select_action(reshaped_state)
                    next_state, reward, done = self.step(action)
                    reshaped_next_state = reconstruct_state(next_state)
                    rollout_traj.append((reshaped_state, action, reward, reshaped_next_state, float(done)))
                    state = next_state

                traj_uncertainty = self.estimate_traj_uncertainty_td(rollout_traj)
                logged_uncertainty = np.log2(float(traj_uncertainty))
                traj_uncertainty_per_position[position].append(logged_uncertainty)
        
        # self.env.close()
        return traj_uncertainty_per_position


    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau

        for t_param, l_param in zip(
                self.actor_target.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        for t_param, l_param in zip(
                self.critic_target.parameters(), self.critic.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

    def _plot(
            self,
            frame_idx: int,
            scores: List[float],
            actor_losses: List[float],
            critic_losses: List[float],
    ):
        """Plot the training progresses."""

        def subplot(loc: int, title: str, values: List[float]):
            plt.subplot(loc)
            plt.title(title)
            plt.plot(values)

        subplot_params = [
            (131, f"frame {frame_idx}. score: {np.mean(scores[-10:])}", scores),
            (132, "actor_loss", actor_losses),
            (133, "critic_loss", critic_losses),
        ]

        plt.figure(figsize=(30, 5))
        for loc, title, values in subplot_params:
            subplot(loc, title, values)
        plt.show()


    def test(self):
            """Test the agent."""
            self.is_test = True

            state = self.env.reset()
            done = False
            score = 0

            frames = []
            while not done:
                frames.append(self.env.render(mode="rgb_array"))
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward

            print("score: ", score)
            self.env.close()

            return frames
    
    
    