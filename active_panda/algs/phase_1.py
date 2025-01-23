import os
import copy
import random
from collections import deque
from typing import Deque, Dict, List, Tuple
import time
import csv

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
from phase_1_func import choose_starting_position, CustomResetWrapper, SlowRenderCallback, choose_task, get_participant_configuration, initialize_envs_for_condition, load_pre_train_model
from phase_1_agent import Phase1_DDPGfD_BC_Agent


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
ACTIVE_PANDA_DIR = os.path.dirname(CURRENT_DIR)
ENVS_DIR = os.path.join(ACTIVE_PANDA_DIR, 'envs')
RESULTS_DIR = os.path.join(ACTIVE_PANDA_DIR, "Phase_1_results")
sys.path.append(ENVS_DIR)
from task_envs import ReachWithObstacleV0Env, ReachWithObstacleV1Env, PushWithObstacleV0Env, PushWithObstacleV1Env, Phase1TaskCentreEnv, Phase1TaskLeftEnv, Phase1TaskRightEnv

# KM: Constants
SEED = 6
NUM_STEPS = 20
MAX_TIMESTEPS = 50

def standard_agent_run(env, test_env, task_name, parent_dir, timestamp, current_condition, log_data, participant_code, testing_flow, training_flow):
    max_demo_num = 20
    warm_up_demo_num = 10
    warm_up_demo_info = {}
    warm_up_demo = []  # a list of tuples (state, action, reward, next_state, done)

    print("Loading demonstration data.")
    # KM: This loads the demonstration data (state and action trajectories) from CSV files.
    joystick_demo_path = os.path.join(parent_dir, 'demo_data', 'joystick_demo', task_name)
    # print(f"DEMO PATH: {joystick_demo_path}")
    loaded_demo_state_trajs = np.genfromtxt(os.path.join(joystick_demo_path, 'demo_state_trajs.csv'), delimiter=' ')
    loaded_demo_action_trajs = np.genfromtxt(os.path.join(joystick_demo_path, 'demo_action_trajs.csv'), delimiter=' ')

    """
    Process Demonstration Data
    """
    # This identifies the indices where new episodes start by looking for rows where the first element is inf.
    starting_ids = [i for i in range(loaded_demo_state_trajs.shape[0]) if loaded_demo_state_trajs[i][0] == np.inf]

    # Processing each demonstration:
    demo_id = 0
    demo_pool = []  # a list of trajectories, each trajectory itself is a list of tuples
    demo_pool_inits = []  # a list of initial states of each demo trajectory
    for i in range(len(starting_ids)):
        # time.sleep(1)
        starting_id = starting_ids[i]
        init_state = loaded_demo_state_trajs[starting_id + 1]
        goal_pos = init_state[-3:]
        demo_pool_inits.append(init_state.copy())
        object_pos = init_state[6:9]
        state = test_env.reset(whether_random=False, goal_pos=goal_pos, object_pos=object_pos)
        step_idx = starting_id + 1
        episode_length = (starting_ids[i + 1] - starting_ids[i] - 1) if i < len(starting_ids) - 1 else (loaded_demo_state_trajs.shape[0] - starting_ids[i] - 1)

        done = False
        step = 0
        traj = []
        while not done and step < episode_length:
            # time.sleep(0.01)
            action = loaded_demo_action_trajs[step_idx]
            next_state, reward, done, info = test_env.step(action)
            reshaped_state = reconstruct_state(state)
            reshaped_next_state = reconstruct_state(next_state)
            traj.append((reshaped_state, action, reward, reshaped_next_state, done))

            state = next_state

            step_idx += 1
            step += 1

        demo_id += 1
        print(f"[Active DDPGfD-BC]: demo {demo_id} is collected, whether success: {info['is_success']}")
        demo_pool.append(traj)
    
    """
    Warm-Up Demonstration Sampling
    """
    sampled_demo_ids = np.random.choice(demo_id, warm_up_demo_num, replace=False)
    for id in sampled_demo_ids:
        sampled_traj = demo_pool[id].copy()
        for data in sampled_traj:
            warm_up_demo.append(data)
    warm_up_demo_info['demo'] = warm_up_demo
    warm_up_demo_info['demo_num'] = warm_up_demo_num

    """
    Agent Initialization
    """
    # KM: agent parameters
    method = 'active_ddpgfd_bc_warmstart_transfer_learning'
    memory_size = 100000
    batch_size = 1024
    ou_noise_theta = 1.0
    ou_noise_sigma = 0.1
    initial_random_steps = 10000 # 10000
    mode = 'scratch'
    max_env_steps = int(394 * 1e3)
    demo_batch_size = 64

    # Logger
    writer = SummaryWriter(os.path.join(parent_dir, 'logs', task_name, method, f'max_demo_{max_demo_num}', mode, timestamp))

    # Loading pre-trained model:
    # agent.load_pre_train_model(
    #     actor_path="path_to_saved_actor.pth",
    #     critic_path="path_to_saved_critic.pth",
    #     actor_target_path="path_to_saved_actor_target.pth",
    #     critic_target_path="path_to_saved_critic_target.pth"
    # )
    
    agent = Phase1_DDPGfD_BC_Agent(
        env,
        test_env=test_env,
        task_name=task_name,
        method=method,
        demo_pool=demo_pool,
        demo_pool_inits=demo_pool_inits,
        memory_size=memory_size,
        batch_size=batch_size,
        demo_batch_size=demo_batch_size,
        max_demo_num=max_demo_num,
        ou_noise_theta=ou_noise_theta,
        ou_noise_sigma=ou_noise_sigma,
        writer=writer,
        demo_info=warm_up_demo_info,
        initial_random_steps=initial_random_steps,
        mode=mode,
        current_condition=current_condition,
        log_data=log_data,
        participant_code=participant_code,
        testing_flow=testing_flow,
        training_flow=training_flow
    )
    print("Agent initialized successfully.")

    print("Starting training step.")
    # Call the train function
    plotting_interval = 1000
    model_saving_interval = 5000
    evaluate_res_per_step, success_evaluate_res_per_step, updated_traj_uncertainty_per_position = agent.train(num_frames=max_env_steps, plotting_interval=plotting_interval, model_saving_interval=model_saving_interval, current_condition=current_condition, log_data=log_data, participant_code=participant_code, testing_flow=testing_flow, training_flow=training_flow)
    
    for position, uncertainties in updated_traj_uncertainty_per_position.items():
        avg_uncertainty = np.mean(uncertainties) if uncertainties else float('inf')
        print(f"Starting position {position} - Average Uncertainty: {avg_uncertainty}")


def main():
    # where to store the logs
    log_data = []

    # Set this to True for testing the center task and condition 1
    TESTING = False
    TRAINING = True
    # for logging purposes
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    # KM: It initializes the seed for PyTorch, NumPy, and Python’s random module.
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # custom parameters for the environment - learning rate (Not sure if its needed)
    custom_objects = {
    "learning_rate": 0.0,
    "lr_schedule": lambda _: 0.0,
    "clip_range": lambda _: 0.0,
    }

    # Get the participant code from the user or test with a hardcoded value
    if TESTING:
        TRAINING = False
        participant_code = "test"
        task_type = choose_task()
        if task_type == 1:
            env = Phase1TaskLeftEnv(render=True, reward_type="modified_sparse", control_type='ee')
            task_name = "Phase1TaskLeft"
            test_env = Phase1TaskLeftEnv(render=False, reward_type="modified_sparse", control_type='ee')
        elif task_type == 2:
            env = Phase1TaskCentreEnv(render=True, reward_type="modified_sparse", control_type='ee')
            task_name = "Phase1TaskCentre"
            test_env = Phase1TaskCentreEnv(render=False, reward_type="modified_sparse", control_type='ee')
        elif task_type == 3:
            env = Phase1TaskRightEnv(render=True, reward_type="modified_sparse", control_type='ee')
            task_name = "Phase1TaskRight"
            test_env = Phase1TaskRightEnv(render=False, reward_type="modified_sparse", control_type='ee')
        env = ActionNormalizer(env)
        env = TimeLimitWrapper(env=env, max_steps=120)
        env = CustomResetWrapper(env=env)
        test_env = ActionNormalizer(test_env)
        test_env = TimeLimitWrapper(env=test_env, max_steps=120)
        test_env = ResetWrapper(env=test_env)
        print("Environment initialized successfully.")
        current_condition = None
        standard_agent_run(env, test_env, task_name, PARENT_DIR, TIMESTAMP, current_condition=current_condition, participant_code=participant_code, log_data=log_data, testing_flow=TESTING, training_flow=TRAINING)
    elif TRAINING:
        TESTING = False
        participant_code = "train"
        task_type = choose_task()
        if task_type == 1:
            env = Phase1TaskLeftEnv(render=True, reward_type="modified_sparse", control_type='ee')
            task_name = "Phase1TaskLeft"
            test_env = Phase1TaskLeftEnv(render=False, reward_type="modified_sparse", control_type='ee')
        elif task_type == 2:
            env = Phase1TaskCentreEnv(render=True, reward_type="modified_sparse", control_type='ee')
            task_name = "Phase1TaskCentre"
            test_env = Phase1TaskCentreEnv(render=False, reward_type="modified_sparse", control_type='ee')
        elif task_type == 3:
            env = Phase1TaskRightEnv(render=True, reward_type="modified_sparse", control_type='ee')
            task_name = "Phase1TaskRight"
            test_env = Phase1TaskRightEnv(render=False, reward_type="modified_sparse", control_type='ee')
        env = ActionNormalizer(env)
        env = TimeLimitWrapper(env=env, max_steps=120)
        env = CustomResetWrapper(env=env)
        test_env = ActionNormalizer(test_env)
        test_env = TimeLimitWrapper(env=test_env, max_steps=120)
        test_env = ResetWrapper(env=test_env)
        print("Environment initialized successfully.")
        current_condition = None
        standard_agent_run(env, test_env, task_name, PARENT_DIR, TIMESTAMP, current_condition=current_condition, participant_code=participant_code, log_data=log_data, testing_flow=TESTING, training_flow=TRAINING)
    else:
        # Get the participant code from the user or test with a hardcoded value
        participant_code = input("Enter the participant code (e.g., 'p1', 'P1'): ")
        participant, baseline_task, condition1_task, condition2_task = get_participant_configuration(participant_code)
        print(f"Participant: {participant}, Baseline: {baseline_task}, Condition 1: {condition1_task}, Condition 2: {condition2_task}")

    log_file = os.path.join(RESULTS_DIR, f"{participant_code.upper()}_experiment_logs.csv")

    # Task selection mapping for Baseline, Condition 1, and Condition 2
    task_map = {
        "Left": Phase1TaskLeftEnv,
        "Centre": Phase1TaskCentreEnv,
        "Right": Phase1TaskRightEnv
    }

    # to keep track of completed tasks
    completed_tasks = []

    if TESTING == False and TRAINING == False:
        # Initialize and run the environment for Baseline if not completed
        if "Baseline" not in completed_tasks:
            current_condition = "Baseline"
            env, test_env, task_name = initialize_envs_for_condition(task_map, baseline_task)
            print(f"Running Baseline with task: {task_name}")
            # Run Baseline training/testing here
            standard_agent_run(env, test_env, task_name, PARENT_DIR, TIMESTAMP, current_condition, log_data=log_data, participant_code=participant_code, testing_flow=TESTING, training_flow=TRAINING)
            completed_tasks.append("Baseline")

        # Initialize and run the environment for Condition 1 if Baseline is completed and Condition 1 is not completed
        if "Baseline" in completed_tasks and "Condition 1" not in completed_tasks:
            current_condition = "Condition 1"
            env, test_env, task_name = initialize_envs_for_condition(task_map, condition1_task)
            print(f"Running Condition 1 with task: {task_name}")
            # Run Condition 1 training/testing here
            standard_agent_run(env, test_env, task_name, PARENT_DIR, TIMESTAMP, current_condition, log_data=log_data, participant_code=participant_code, testing_flow=TESTING, training_flow=TRAINING)
            completed_tasks.append("Condition 1")

        # Initialize and run the environment for Condition 2 if Condition 1 is completed and Condition 2 is not completed
        if "Condition 1" in completed_tasks and "Condition 2" not in completed_tasks:
            current_condition = "Condition 2"
            env, test_env, task_name = initialize_envs_for_condition(task_map, condition2_task)
            print(f"Running Condition 2 with task: {task_name}")
            # Run Condition 2 training/testing here
            standard_agent_run(env, test_env, task_name, PARENT_DIR, TIMESTAMP, current_condition, log_data=log_data, participant_code=participant_code, testing_flow=TESTING, training_flow=TRAINING)
            completed_tasks.append("Condition 2")

        if len(completed_tasks) == 3:
            print("All conditions have been completed.")
            ## SAVE RESULTS ETC

    with open(log_file, 'w', newline='') as file:
        writer = csv.writer(file)
        header = ["Participant Code", "Task", "Condition", "Uncertainties", "Selected Starting State", "Selected State Uncertainty"]
        writer.writerow(header)
        for row in log_data:
            writer.writerow(row)


if __name__ == "__main__":
    main()