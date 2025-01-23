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

class CustomResetWrapper(Wrapper):
    """
    To ensure compatibility with Stable Baselines3.
    """
    def __init__(self, env):
        super().__init__(env)
        # needs to be initialized so it can store the chosen object position. (needs to hold the object pos across calls in the wrapper)
        self.object_position = None
        self.goal_pos = None
        self.resetting_position = False
        self.object_color = [1, 1, 1]  # Default color
        if hasattr(self.env.unwrapped, 'set_wrapper_env'):
            self.env.unwrapped.set_wrapper_env(self)
        # KM: Observation space and action space
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, **kwargs):
        """
        Observations look like they should always be in form of numpy arrays, either directly or within a dictionary.
        """
        self.resetting_position = True

        if self.object_position is not None:
            object_pos = self.object_position
        else:
            object_pos = kwargs.get('object_pos')

        object_pos = kwargs.pop('object_pos', self.object_position)
        goal_pos = kwargs.pop('goal_pos', self.goal_pos)
        robot_state = kwargs.pop('robot_state', None)
        object_color = kwargs.pop('object_color', None)

        if object_color is not None:
            self.object_color = object_color

        reset_kwargs = kwargs.copy()
        if object_pos is not None:
            reset_kwargs['object_pos'] = object_pos
        if goal_pos is not None:
            reset_kwargs['goal_pos'] = goal_pos
        if robot_state is not None:
            reset_kwargs['robot_state'] = robot_state
        if self.object_color is not None:
            reset_kwargs['object_color'] = self.object_color

        obs = self.env.reset(**reset_kwargs)

        self.resetting_position = False
        # print(f"CustomResetWrapper: reset() completed, resetting_position set to False, object_color: {self.object_color}")
        # return formatted_result, {}
        return obs
    
    def step(self, action):
        # KM: THIS IS FOR USING PHASE 1 DDPGfD BC AGENT:
        # self.env.object_color = self.object_color  # Pass the color to the inner environment
        self.env.unwrapped.object_color = self.object_color
        obs, reward, done, info = self.env.step(action)
        if info.get('is_success', False):
            done = True
        return obs, reward, done, info


def choose_starting_position(wrapper_env, position=None):
    """
    Sets 'object_position' directly in the wrapper environment.
    """
    if position is None:
        position = int(input("Choose a starting position (0-9):"))

    wrapper_env.resetting_position = True

    base_env = wrapper_env.env
    wrapper_env.object_color = base_env.task.colors[position % len(base_env.task.colors)]
    while hasattr(base_env, 'env'):
        base_env = base_env.env

    wrapper_env.object_position = base_env.task.predefined_obj_positions[position]
    object_color = base_env.task.colors[position % len(base_env.task.colors)]
    wrapper_env.object_color = object_color

    # Build reset arguments
    reset_kwargs = {
        'object_pos': wrapper_env.object_position,
        'object_color': wrapper_env.object_color
    }

    # Call reset with kwargs
    wrapper_env.reset(
        object_pos=wrapper_env.object_position,
        object_color=wrapper_env.object_color
    )

    wrapper_env.resetting_position = False
    return position

def choose_task():
    print("Choose a task type (1=left, 2=centre, 3=right):")
    task_type = int(input("ENTER TASK TYPE: "))
    if task_type not in [1, 2, 3]:
        print("Invalid task type. Please choose '1=left', '2=centre', or '3=right'.")
        return choose_task()
    return task_type

def choose_model_step():
    print("Choose a model step (must be between 5000 and 1000000, in increments of 5000):")
    print("COPY AND PASTE example: 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000")
    model_step = int(input("ENTER STEP NUMBER: "))
    if model_step % 5000 != 0 or model_step < 5000 or model_step > 1000000:
        print("Invalid step number. Please choose a step number between 5000 and 1000000, in increments of 5000.")
        return choose_model_step()
    return model_step


def choose_random_position(wrapper_env):
        """
        Function to choose a random starting position for the object.
        """
        num_positions = 10
        return np.random.choice(num_positions)


def get_participant_configuration(participant_code: str) -> Tuple[str, List[str], int]:
    participant_code = participant_code.lower()
    if participant_code.startswith('p'):
        participant_num = int(participant_code[1:])
    else:
        raise ValueError("Participant code should start with 'P' or 'p' followed by a number.")

    # Determine the group based on even or odd participant number
    group = 1 if participant_num % 2 == 0 else 2

    # Define all possible tasks
    tasks = ["Left", "Centre", "Right"]

    # Randomly shuffle tasks for this participant
    random.seed(participant_num)  # Seed with participant number for reproducibility
    random.shuffle(tasks)

    return f"P{participant_num}", tasks, group


def apply_wrappers(env, test_env):
    # KM: Environment Wrappers
    env = ActionNormalizer(env)
    env = CustomResetWrapper(env=env)
    env = TimeFeatureWrapper(env=env, max_steps=120, test_mode=False)
    env = TimeLimitWrapper(env=env, max_steps=120)
    test_env = ActionNormalizer(test_env)
    test_env = CustomResetWrapper(env=test_env)
    test_env = TimeFeatureWrapper(env=test_env, max_steps=120, test_mode=False)
    test_env = TimeLimitWrapper(env=test_env, max_steps=120)
    return env, test_env

def initialize_envs_for_condition(task_map, task_key):
    if task_key in task_map:
        env_class = task_map[task_key]
        env = env_class(render=True, reward_type='modified_sparse', control_type='ee')
        test_env = env_class(render=False, reward_type='modified_sparse', control_type='ee')
        env, test_env = apply_wrappers(env, test_env)
        
        # Determine the task_name based on the task_key
        if "Centre" in task_key:
            task_name = "Phase1TaskCentre"
        elif "Left" in task_key:
            task_name = "Phase1TaskLeft"
        elif "Right" in task_key:
            task_name = "Phase1TaskRight"
        else:
            task_name = "UnknownTask"

        return env, test_env, task_name
    else:
        raise ValueError(f"Invalid task type: {task_key}")
    
def trajectories_for_position(self, chosen_position, demo_state_trajs, demo_action_trajs):
    """
    This function loads the correct trajectories for a given starting position.
    
    Parameters:
    - chosen_position (int): The starting position selected by the user (between 0-9).
    - demo_state_trajs (np.array): Loaded state trajectories from the CSV file.
    - demo_action_trajs (np.array): Loaded action trajectories from the CSV file.
    
    Returns:
    - demo_pool (list): A list of trajectories corresponding to the chosen position.
    """

    # Identify the indices where new episodes start by looking for rows where the first element is inf
    starting_ids = [i for i in range(demo_state_trajs.shape[0]) if demo_state_trajs[i][0] == float('inf')]
    
    # Ensure that the chosen position is within the range of available starting IDs
    if chosen_position < 0 or chosen_position >= len(starting_ids):
        raise ValueError(f"Chosen position {chosen_position} is out of range. It must be between 0 and {len(starting_ids) - 1}.")
    
    demo_pool = []  

    # Get the starting index for the chosen position
    starting_id = starting_ids[chosen_position]
    
    # Determine the end index for this rollout
    end_id = starting_ids[chosen_position + 1] if chosen_position < len(starting_ids) - 1 else demo_state_trajs.shape[0]

    base_env = get_base_env(self.env)
    base_env.in_demo_rollout = True

    # Retrieve the initial state and reset the environment
    init_state = demo_state_trajs[starting_id + 1]
    goal_pos = init_state[-3:]
    object_pos = init_state[6:9]
    state = self.env.reset(goal_pos=goal_pos, object_pos=object_pos)

    # Add the demonstration text above the robot arm
    self.env.add_human_demo_text()
    # Add the hand image above the robot arm
    # self.env.add_human_demo_image()
    
    # Process the trajectory
    step_idx = starting_id + 1
    episode_length = end_id - starting_id - 1
    
    done = False
    step = 0
    traj = []
    while not done and step < episode_length:
        # time.sleep(0.01)
        action = demo_action_trajs[step_idx]
        next_state, reward, done, info = self.env.step(action)

        self.env.update_human_demo_text_position()
        # self.env.update_human_demo_image_position()

        reshaped_state = reconstruct_state(state)
        reshaped_next_state = reconstruct_state(next_state)
        traj.append((reshaped_state, action, reward, reshaped_next_state, done))

        state = next_state

        step_idx += 1
        step += 1

    base_env.in_demo_rollout = False
    base_env.clear_demo_lines()
    # Remove the demonstration text after the rollout
    self.env.remove_human_demo_text()
    # Remove the hand image after the rollout
    # self.env.remove_human_demo_image()
    print(f"Success: {info['is_success']}")
    demo_pool.append(traj)
    
    print(f"[trajectories_for_position]: Loaded trajectory for starting position {chosen_position}")
    return demo_pool

def load_pre_train_model(self, actor_path, critic_path, actor_target_path=None, critic_target_path=None):
    """
    Load model weights from the given paths.
    """
    self.actor.load_state_dict(torch.load(actor_path))
    self.critic.load_state_dict(torch.load(critic_path))
    
    if actor_target_path is not None:
        self.actor_target.load_state_dict(torch.load(actor_target_path))
    
    if critic_target_path is not None:
        self.critic_target.load_state_dict(torch.load(critic_target_path))
    
    print(f"Model loaded successfully from {actor_path} and {critic_path}")

def get_base_env(env):
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env
    return base_env

def categorize_uncertainties(uncertainties):
    min_val = min(uncertainties)
    max_val = max(uncertainties)
    normalized_uncertainties = [(u - min_val) / (max_val - min_val) for u in uncertainties]

    sorted_indices = np.argsort(normalized_uncertainties)
    
    categories = [''] * len(uncertainties)
    shocking_indices = sorted_indices[-2:]  # Top 2 are Shocking
    closer_to_expectation_indices = sorted_indices[:2]  # Bottom 2 are Closer to Expectation
    surprising_indices = sorted_indices[2:8]  # Middle 6 are Surprising

    for idx in shocking_indices:
        categories[idx] = "Highly Unexpected"
    for idx in closer_to_expectation_indices:
        categories[idx] = "Expected"
    for idx in surprising_indices:
        categories[idx] = "Unexpected"

    return categories