import os
import random
from collections import deque
from typing import Deque, Dict, List, Tuple
import time
from datetime import datetime
import csv
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

seed = 6
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

import gym
from gym import Wrapper
import panda_gym
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim


# \active_panda\algs
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# \active_panda
PARENT_DIR = os.path.dirname(CURRENT_DIR)
# \active_panda
ACTIVE_PANDA_DIR = os.path.dirname(CURRENT_DIR)
# \active_panda\envs
ENVS_DIR = os.path.join(ACTIVE_PANDA_DIR, 'envs')
# \active_panda\Phase_1_results
RESULTS_DIR = os.path.join(ACTIVE_PANDA_DIR, "Phase_1_results")
sys.path.append(ENVS_DIR)
sys.path.append(PARENT_DIR + '/utils/')

from phase_1_awac_agent import AWACAgent
from utils_awac import ActionNormalizer, ResetWrapper, TimeFeatureWrapper, TimeLimitWrapper, RealRobotWrapper, save_results
from phase_1_func import CustomResetWrapper, choose_task, get_participant_configuration, initialize_envs_for_condition
from task_envs import Phase1TaskCentreEnv, Phase1TaskLeftEnv, Phase1TaskRightEnv

# Set the seed for reproducibility
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

def prepare_demo_pool(demo_path):
    """
    Prepares a pool of demonstration episodes by loading trajectory data from the specified demo_path
    and splitting it into individual episodes based on inf markers in the state trajectory.

    Takes in demo_path: Expected files: 'state_traj.csv', 'action_traj.csv', 
                         'next_state_traj.csv', 'reward_traj.csv', 'done_traj.csv'.
    Returns a list of dicts where each dict contains an episode:
        Each contains the following keys:
              - 'state_trajectory': The sequence of states for the episode.
              - 'action_trajectory': The sequence of actions for the episode.
              - 'next_state_trajectory': The sequence of next states for the episode.
              - 'reward_trajectory': The sequence of rewards for the episode.
              - 'done_trajectory': The sequence indicating if the episode is finished.
    """
    state_traj = np.genfromtxt(demo_path + 'state_traj.csv', delimiter=' ')
    action_traj = np.genfromtxt(demo_path + 'action_traj.csv', delimiter=' ')
    next_state_traj = np.genfromtxt(demo_path + 'next_state_traj.csv', delimiter=' ')
    reward_traj = np.genfromtxt(demo_path + 'reward_traj.csv', delimiter=' ')
    done_traj = np.genfromtxt(demo_path + 'done_traj.csv', delimiter=' ')

    reward_traj = np.reshape(reward_traj, (-1, 1))
    done_traj = np.reshape(done_traj, (-1, 1))

    print("reward traj shape: {}".format(reward_traj.shape))
    print("done traj shape: {}".format(done_traj.shape))

    # first go through the loaded trajectory to get the index of episode starting signs
    starting_ids = []
    for i in range(state_traj.shape[0]):
        if state_traj[i][0] == np.inf:
            starting_ids.append(i)
    total_demo_num = len(starting_ids)

    demos = []
    for i in range(total_demo_num):
        if i < total_demo_num - 1:
            start_step_id = starting_ids[i]
            end_step_id = starting_ids[i + 1]
        else:
            start_step_id = starting_ids[i]
            end_step_id = state_traj.shape[0]

        states = state_traj[(start_step_id + 1):end_step_id, :]
        actions = action_traj[(start_step_id + 1):end_step_id, :]
        next_states = next_state_traj[(start_step_id + 1):end_step_id, :]
        rewards = reward_traj[(start_step_id + 1):end_step_id, :]
        dones = done_traj[(start_step_id + 1):end_step_id, :]
        demo = {'state_trajectory': states.copy(),
                'action_trajectory': actions.copy(),
                'next_state_trajectory': next_states.copy(),
                'reward_trajectory': rewards.copy(),
                'done_trajectory': dones.copy()}
        demos.append(demo)

    return demos

def standard_agent_run(env, test_env, task_name, parent_dir, timestamp, log_data, participant_code, testing_flow, training_flow, group, roll_demos_in_training, load_pretrained=False, pretrained_model_step=None, pretrained_model_path=None, show_uncertainty_categories=False, show_uncertainty_category_colors=False, condition='blind', continue_experiment_training=False, selection_strategy='Expectation-oriented'):
    max_demo_num = 10

    base_demo_pool_path = PARENT_DIR + '/' + 'demo_data/' + '/' + 'AWAC_' + task_name + '/'
    base_demo_pool = prepare_demo_pool(demo_path=base_demo_pool_path)

    # sample 10 base demos
    sampled_demos = []
    sample_num = max_demo_num
    demo_ids = np.random.choice(len(base_demo_pool),sample_num, replace=False)
    for demo_id in demo_ids:
        sampled_base_demo = base_demo_pool[demo_id].copy()
        sampled_demos.append(sampled_base_demo)

    # training parameters
    if task_name == 'Phase1TaskCentre' or task_name == 'Phase1TaskRight' or task_name == 'Phase1TaskLeft':
        # max_env_steps = int(1000 * 1e3)
        max_env_steps = int(700000)
    
    # training hyperparameters
    method = 'awac_no_layernorm'
    memory_size = 100000
    batch_size = 1024

     # logger
    if selection_strategy == 'Expectation-oriented':
        writer = SummaryWriter(
            PARENT_DIR + '/' + 'logs/' + task_name + '/' + method + '_' + str(selection_strategy) + '_' + str(pretrained_model_step))
    elif selection_strategy == 'Result-oriented':
        writer = SummaryWriter(
            PARENT_DIR + '/' + 'logs/' + task_name + '/' + method  + str(selection_strategy) + str(pretrained_model_step))
    elif selection_strategy == 'Lowest TD error':
        writer = SummaryWriter(
            PARENT_DIR + '/' + 'logs/' + task_name + '/' + method + str(selection_strategy) + str(pretrained_model_step))
    elif selection_strategy == 'P1_blind':
        writer = SummaryWriter(
            PARENT_DIR + '/' + 'logs/' + task_name + '/' + method + str(selection_strategy) + str(pretrained_model_step))
    elif selection_strategy == 'P1_visible':
        writer = SummaryWriter(
            PARENT_DIR + '/' + 'logs/' + task_name + '/' + method + str(selection_strategy) + str(pretrained_model_step))
    elif selection_strategy == 'P2_blind':
        writer = SummaryWriter(
            PARENT_DIR + '/' + 'logs/' + task_name + '/' + method + str(selection_strategy) + str(pretrained_model_step))
    elif selection_strategy == 'P2_visible':
        writer = SummaryWriter(
            PARENT_DIR + '/' + 'logs/' + task_name + '/' + method + str(selection_strategy) + str(pretrained_model_step))
    elif selection_strategy == 'P3_blind':
        writer = SummaryWriter(
            PARENT_DIR + '/' + 'logs/' + task_name + '/' + method + str(selection_strategy) + str(pretrained_model_step))
    elif selection_strategy == 'P3_visible':
        writer = SummaryWriter(
            PARENT_DIR + '/' + 'logs/' + task_name + '/' + method + str(selection_strategy) + str(pretrained_model_step))
    elif selection_strategy == 'P4_blind':
        writer = SummaryWriter(
            PARENT_DIR + '/' + 'logs/' + task_name + '/' + method + str(selection_strategy) + str(pretrained_model_step))
    elif selection_strategy == 'P4_visible':
        writer = SummaryWriter(
            PARENT_DIR + '/' + 'logs/' + task_name + '/' + method + str(selection_strategy) + str(pretrained_model_step))
    elif selection_strategy == 'P5_blind':
        writer = SummaryWriter(
            PARENT_DIR + '/' + 'logs/' + task_name + '/' + method + str(selection_strategy) + str(pretrained_model_step))
    elif selection_strategy == 'P5_visible':
        writer = SummaryWriter(
            PARENT_DIR + '/' + 'logs/' + task_name + '/' + method + str(selection_strategy) + str(pretrained_model_step))
    elif selection_strategy == 'P6_blind':
        writer = SummaryWriter(
            PARENT_DIR + '/' + 'logs/' + task_name + '/' + method + str(selection_strategy) + str(pretrained_model_step))
    elif selection_strategy == 'P6_visible':
        writer = SummaryWriter(
            PARENT_DIR + '/' + 'logs/' + task_name + '/' + method + str(selection_strategy) + str(pretrained_model_step))
    elif selection_strategy == 'P7_blind':
        writer = SummaryWriter(
            PARENT_DIR + '/' + 'logs/' + task_name + '/' + method + str(selection_strategy) + str(pretrained_model_step))
    elif selection_strategy == 'P7_visible':
        writer = SummaryWriter(
            PARENT_DIR + '/' + 'logs/' + task_name + '/' + method + str(selection_strategy) + str(pretrained_model_step))
    elif selection_strategy == 'P8_blind':
        writer = SummaryWriter(
            PARENT_DIR + '/' + 'logs/' + task_name + '/' + method + str(selection_strategy) + str(pretrained_model_step))
    elif selection_strategy == 'P8_visible':
        writer = SummaryWriter(
            PARENT_DIR + '/' + 'logs/' + task_name + '/' + method + str(selection_strategy) + str(pretrained_model_step))
    elif selection_strategy == 'P9_blind':
        writer = SummaryWriter(
            PARENT_DIR + '/' + 'logs/' + task_name + '/' + method + str(selection_strategy) + str(pretrained_model_step))
    elif selection_strategy == 'P9_visible':
        writer = SummaryWriter(
            PARENT_DIR + '/' + 'logs/' + task_name + '/' + method + str(selection_strategy) + str(pretrained_model_step))
    elif selection_strategy == 'P10_blind':
        writer = SummaryWriter(
            PARENT_DIR + '/' + 'logs/' + task_name + '/' + method + str(selection_strategy) + str(pretrained_model_step))
    elif selection_strategy == 'P10_visible':
        writer = SummaryWriter(
            PARENT_DIR + '/' + 'logs/' + task_name + '/' + method + str(selection_strategy) + str(pretrained_model_step))
    elif selection_strategy == 'P11_blind':
        writer = SummaryWriter(
            PARENT_DIR + '/' + 'logs/' + task_name + '/' + method + str(selection_strategy) + str(pretrained_model_step))
    elif selection_strategy == 'P11_visible':
        writer = SummaryWriter(
            PARENT_DIR + '/' + 'logs/' + task_name + '/' + method + str(selection_strategy) + str(pretrained_model_step))
    elif selection_strategy == 'P12_blind':
        writer = SummaryWriter(
            PARENT_DIR + '/' + 'logs/' + task_name + '/' + method + str(selection_strategy) + str(pretrained_model_step))
    elif selection_strategy == 'P12_visible':
        writer = SummaryWriter(
            PARENT_DIR + '/' + 'logs/' + task_name + '/' + method + str(selection_strategy) + str(pretrained_model_step))
    else:
        writer = SummaryWriter(
            PARENT_DIR + '/' + 'logs/' + task_name + '/' + method + '/max_demo_' + str(max_demo_num) + '/' + TIMESTAMP + f'_{condition}')

    # initialize the agent
    agent = AWACAgent(env=env,
                      test_env=test_env,
                      with_layer_norm=True,
                      task_name=task_name,
                      writer=writer,
                      batch_size=1024,
                      epochs=int(max_env_steps / 1000.0),
                      log_data=log_data,
                      participant_code=participant_code,
                      testing_flow=testing_flow,
                      training_flow=training_flow,
                      roll_demos_in_training=roll_demos_in_training,
                      load_pretrained=load_pretrained,
                      pretrained_model_step=pretrained_model_step,
                      pretrained_model_path=pretrained_model_path,
                      show_uncertainty_categories=show_uncertainty_categories,
                      show_uncertainty_category_colors=show_uncertainty_category_colors,
                      condition=condition)
    print("Agent initialized successfully.")
    
    # agent.populate_replay_buffer(demos_list=sampled_demos)
    # print("Replay buffer populated successfully.")

    print("Starting training.")
    reward_res_per_step, success_res_per_step = agent.run(
        task_name=task_name,
        method=method,
        max_demo_num=max_demo_num,
        model_saving_interval=5000,
        log_data=log_data,
        participant_code=participant_code,
        testing_flow=testing_flow,
        training_flow=training_flow,
        roll_demos_in_training=roll_demos_in_training,
        group=group,
        load_pretrained=load_pretrained,
        pretrained_model_step=pretrained_model_step,
        pretrained_model_path=pretrained_model_path,
        show_uncertainty_categories=show_uncertainty_categories,
        show_uncertainty_category_colors=show_uncertainty_category_colors,
        condition=condition,
        continue_experiment_training=continue_experiment_training,
        selection_strategy=selection_strategy)

    # FOR TREATING UNCERTAINTIES
    # for position, uncertainties in updated_traj_uncertainty_per_position.items():
    #     avg_uncertainty = np.mean(uncertainties) if uncertainties else float('inf')
    #     print(f"Starting position {position} - Average Uncertainty: {avg_uncertainty}")

    save_results(method, task_name, reward_res_per_step, success_res_per_step, max_demo_num, mode=None)
    writer.close()
    print("[AWAC]: Results are saved! All finished")
    print("************************")

def main():
    # Initialize log data
    log_data = []

    # Load pre-trained models
    load_pretrained=True

    # Display settings for uncertainty categories
    show_uncertainty_categories = True
    show_uncertainty_category_colors = True

    """
    FLAGS to control the flow of the script:
        - Only one of these should be True at a time.
    """
    # Continue training with an anchor policy
    CONTINUE_EXPERIMENT_TRAINING = True
    # Test all participant models
    TEST_ALL_MODELS = False
    # Run in testing mode
    TESTING = False
    # Run in training mode
    TRAINING = False
    # Roll in demonstrations during training
    ROLL_DEMO_IN_TRAINING = False
    # Do all tasks instead of a single task
    DO_ALL_TASKS = False

    # LOGGING
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    method = 'awac_no_layernorm'

    # Only one main flag is True
    flags = [CONTINUE_EXPERIMENT_TRAINING, TEST_ALL_MODELS, TESTING, TRAINING]
    if sum([int(flag) for flag in flags]) > 1:
        raise ValueError("Only one of CONTINUE_EXPERIMENT_TRAINING, TEST_ALL_MODELS, TESTING, TRAINING can be True at a time.")
    
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


    participants = ['P12_blind', 'P12_visible']
    
    """
    EXPERIMENT START
    """
    if CONTINUE_EXPERIMENT_TRAINING:
        # participant code and other parameters
        task_name = 'Phase1TaskLeft'
        # selection_strategy = 'P2_visible'
        pretrained_model_steps = ['600000']

        for p in participants:
            selection_strategy = p
            for pretrained_model_step in pretrained_model_steps:
                env = Phase1TaskLeftEnv(render=True, reward_type="modified_sparse", control_type='ee')
                test_env = Phase1TaskLeftEnv(render=False, reward_type="modified_sparse", control_type='ee')
                # env = Phase1TaskCentreEnv(render=False, reward_type="modified_sparse", control_type='ee')
                # test_env = Phase1TaskCentreEnv(render=False, reward_type="modified_sparse", control_type='ee')
                # env = Phase1TaskRightEnv(render=False, reward_type="modified_sparse", control_type='ee')
                # test_env = Phase1TaskRightEnv(render=False, reward_type="modified_sparse", control_type='ee')
                env = ActionNormalizer(env)
                env = CustomResetWrapper(env=env)
                env = TimeFeatureWrapper(env=env, max_steps=120, test_mode=False)
                env = TimeLimitWrapper(env=env, max_steps=120)
                test_env = ActionNormalizer(test_env)
                test_env = CustomResetWrapper(env=test_env)
                test_env = TimeFeatureWrapper(test_env, max_steps=120, test_mode=False)
                test_env = TimeLimitWrapper(test_env, max_steps=120)
                print("Environment initialized successfully for continue experiment training.")

                if pretrained_model_step is not None:
                    load_pretrained = True
                    pretrained_model_step_str = str(pretrained_model_step)
                    participant_code = f"{task_name}_{selection_strategy}_{pretrained_model_step_str}"
                    pretrained_model_path = os.path.join(PARENT_DIR, 'models', task_name, method, 'max_demo_10')
                    print(f"Starting training with pretrained model at step {pretrained_model_step}")
                else:
                    # Starting from scratch
                    load_pretrained = False
                    pretrained_model_step_str = 'from_scratch'
                    participant_code = f"{task_name}_{selection_strategy}_{pretrained_model_step_str}"
                    pretrained_model_path = None
                    print("Starting training from scratch (no pretrained model)")

                print(f"Running task: {task_name} for {participant_code} with pretrained model step: {pretrained_model_step}")

                log_data = []
                log_file = os.path.join(RESULTS_DIR, f"{participant_code}_experiment_logs.csv")
            
                standard_agent_run(
                    env,
                    test_env,
                    task_name,
                    PARENT_DIR,
                    TIMESTAMP,
                    log_data=log_data,
                    participant_code=participant_code,
                    testing_flow=False,
                    training_flow=False,
                    group=None,
                    roll_demos_in_training=False,
                    load_pretrained=load_pretrained,
                    pretrained_model_step=pretrained_model_step,
                    pretrained_model_path=pretrained_model_path,
                    show_uncertainty_categories=show_uncertainty_categories,
                    show_uncertainty_category_colors=show_uncertainty_category_colors,
                    condition='profile_training',
                    continue_experiment_training=True,
                    selection_strategy=selection_strategy)

                with open(log_file, 'w', newline='') as file:
                    log_file = os.path.join(RESULTS_DIR, f"{participant_code}_experiment_logs.csv")
                    writer_exp = csv.writer(file)
                    header = ["Participant Code", "Condition", "Task", "Uncertainties", "Selected Starting State", "Selected State Uncertainty", "Selected State Category", "Uncertainty Categories", "Selection Strategy"]
                    writer_exp.writerow(header)
                    for row in log_data:
                        writer_exp.writerow(row)
        return


    if TEST_ALL_MODELS:
        # Initialize the Left task environment
        env = Phase1TaskLeftEnv(render=False, reward_type="modified_sparse", control_type='ee')
        test_env = Phase1TaskLeftEnv(render=False, reward_type="modified_sparse", control_type='ee')
        env = ActionNormalizer(env)
        env = CustomResetWrapper(env=env)
        env = TimeFeatureWrapper(env=env, max_steps=120, test_mode=False)
        env = TimeLimitWrapper(env=env, max_steps=120)
        test_env = ActionNormalizer(test_env)
        test_env = CustomResetWrapper(env=test_env)
        test_env = TimeFeatureWrapper(test_env, max_steps=120, test_mode=False)
        test_env = TimeLimitWrapper(test_env, max_steps=120)

        print("Environment initialized successfully for Left task testing.")

        # Initialize the agent
        agent = AWACAgent(
            env=env,
            test_env=test_env,
            with_layer_norm=True,
            task_name='Phase1TaskLeft',
            writer=None,  
            batch_size=1024,
            epochs=0,
            log_data=None,
            participant_code=None,
            testing_flow=False,
            training_flow=False,
            roll_demos_in_training=False
        )
        print("Agent initialized successfully for testing.")
        print("TESTING all models for the participants.")
        # Call the test_agent function with test_all_models=True
        results = agent.test_agent(task_name='Phase1TaskLeft', test_all_models=True)
        results_df = pd.DataFrame(results)
        results_csv_path = os.path.join(RESULTS_DIR, 'Phase_1_participant_performance_test.csv')
        results_df.to_csv(results_csv_path, index=False)
        print(f"Results saved to {results_csv_path}")
        print(results_df.groupby('condition')[['average_reward', 'success_rate']].mean())
        # Exit the script after testing
        return


    if TESTING:
        TRAINING = False
        ROLL_DEMO_IN_TRAINING = False
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
        env = CustomResetWrapper(env=env)
        env = TimeFeatureWrapper(env=env, max_steps=120, test_mode=False)
        env = TimeLimitWrapper(env=env, max_steps=120)
        test_env = ActionNormalizer(test_env)
        test_env = CustomResetWrapper(env=test_env)
        test_env = TimeFeatureWrapper(env=test_env, max_steps=120, test_mode=False)
        test_env = TimeLimitWrapper(env=test_env, max_steps=120)
        print("Environment initialized successfully.")
        standard_agent_run(env, test_env, task_name, PARENT_DIR, TIMESTAMP, participant_code=participant_code, log_data=log_data, testing_flow=TESTING, training_flow=TRAINING, group=None, roll_demos_in_training=ROLL_DEMO_IN_TRAINING, continue_experiment_training=False)
    elif TRAINING:
        TESTING = False
        participant_code = "train"
        task_type = choose_task()
        if task_type == 1:
            env = Phase1TaskLeftEnv(render=False, reward_type="modified_sparse", control_type='ee')
            task_name = "Phase1TaskLeft"
            test_env = Phase1TaskLeftEnv(render=False, reward_type="modified_sparse", control_type='ee')
        elif task_type == 2:
            env = Phase1TaskCentreEnv(render=False, reward_type="modified_sparse", control_type='ee')
            task_name = "Phase1TaskCentre"
            test_env = Phase1TaskCentreEnv(render=False, reward_type="modified_sparse", control_type='ee')
        elif task_type == 3:
            env = Phase1TaskRightEnv(render=False, reward_type="modified_sparse", control_type='ee')
            task_name = "Phase1TaskRight"
            test_env = Phase1TaskRightEnv(render=False, reward_type="modified_sparse", control_type='ee')
        env = ActionNormalizer(env)
        env = CustomResetWrapper(env=env)
        env = TimeFeatureWrapper(env=env, max_steps=120, test_mode=False)
        env = TimeLimitWrapper(env=env, max_steps=120)
        test_env = ActionNormalizer(test_env)
        test_env = CustomResetWrapper(env=test_env)
        test_env = TimeFeatureWrapper(env=test_env, max_steps=120, test_mode=False)
        test_env = TimeLimitWrapper(env=test_env, max_steps=120)
        print("Environment initialized successfully.")
        standard_agent_run(env, test_env, task_name, PARENT_DIR, TIMESTAMP, participant_code=participant_code, log_data=log_data, testing_flow=TESTING, training_flow=TRAINING, group=None, roll_demos_in_training=ROLL_DEMO_IN_TRAINING, continue_experiment_training=False)
    else:
        # Get the participant code from the user or test with a hardcoded value
        participant_code = input("Enter the participant code (e.g., 'p1', 'P1'): ")
        participant, tasks, group = get_participant_configuration(participant_code)
        print(f"Participant: {participant}, Group: {group}, Tasks: {tasks}")

    log_file = os.path.join(RESULTS_DIR, f"{participant_code.upper()}_experiment_logs.csv")

    # Task selection mapping
    task_map = {
        "Left": Phase1TaskLeftEnv,
        "Centre": Phase1TaskCentreEnv,
        "Right": Phase1TaskRightEnv
    }

    # to keep track of completed tasks
    completed_tasks = []

    if TESTING == False and TRAINING == False and CONTINUE_EXPERIMENT_TRAINING == False:
        if DO_ALL_TASKS:
            for task_key in tasks:
                for condition in ["blind", "visible"]:
                    env_class = task_map[task_key]
                    env = env_class(render=True, reward_type='modified_sparse', control_type='ee')
                    test_env = env_class(render=False, reward_type='modified_sparse', control_type='ee')
                    env = ActionNormalizer(env)
                    env = CustomResetWrapper(env=env)
                    env = TimeFeatureWrapper(env=env, max_steps=120, test_mode=False)
                    env = TimeLimitWrapper(env=env, max_steps=120)
                    test_env = ActionNormalizer(test_env)
                    test_env = CustomResetWrapper(env=test_env)
                    test_env = TimeFeatureWrapper(test_env, max_steps=120, test_mode=False)
                    test_env = TimeLimitWrapper(test_env, max_steps=120)

                    if "Centre" in task_key:
                        task_name = "Phase1TaskCentre"
                        pretrained_model_step = '900000'
                    elif "Left" in task_key:
                        task_name = "Phase1TaskLeft"
                        pretrained_model_step = '300000'
                    elif "Right" in task_key:
                        task_name = "Phase1TaskRight"
                        pretrained_model_step = '100000'
                    else:
                        task_name = "UnknownTask"

                    print(f"Running task: {task_name} for participant {participant}")

                    # Anchor model for the task
                    load_pretrained = True
                    # pretrained_model_step = '200000'
                    pretrained_model_path = os.path.join(PARENT_DIR, 'models', task_name, method, 'max_demo_10')

                    # Determine whether to show uncertainties based on the condition
                    if condition == "blind":
                        show_uncertainty_categories = False
                        show_uncertainty_category_colors = False
                    elif condition == "visible":
                        show_uncertainty_categories = True
                        show_uncertainty_category_colors = True

                    standard_agent_run(env, test_env, task_name, PARENT_DIR, TIMESTAMP, log_data=log_data, participant_code=participant_code, testing_flow=TESTING, training_flow=TRAINING, group=group, roll_demos_in_training=ROLL_DEMO_IN_TRAINING, load_pretrained=load_pretrained, pretrained_model_step=pretrained_model_step, pretrained_model_path=pretrained_model_path, show_uncertainty_categories=show_uncertainty_categories, show_uncertainty_category_colors=show_uncertainty_category_colors, condition=condition, continue_experiment_training=False)
        
        else:
            # RUN SINGLE TASK:
            task_key = 'Left'
            task_name = "Phase1TaskLeft"
            pretrained_model_step = '350000'

            for condition in ["blind", "visible"]:
                env_class = task_map[task_key]
                env = env_class(render=True, reward_type='modified_sparse', control_type='ee')
                test_env = env_class(render=False, reward_type='modified_sparse', control_type='ee')
                env = ActionNormalizer(env)
                env = CustomResetWrapper(env=env)
                env = TimeFeatureWrapper(env=env, max_steps=120, test_mode=False)
                env = TimeLimitWrapper(env=env, max_steps=120)
                test_env = ActionNormalizer(test_env)
                test_env = CustomResetWrapper(env=test_env)
                test_env = TimeFeatureWrapper(test_env, max_steps=120, test_mode=False)
                test_env = TimeLimitWrapper(test_env, max_steps=120)

                print(f"Running task: {task_name} for participant {participant}, Condition: {condition}")
                # Anchor model for the task
                load_pretrained = True
                pretrained_model_path = os.path.join(PARENT_DIR, 'models', task_name, method, 'max_demo_10')
                # Determine whether to show uncertainties based on the condition
                if condition == "blind":
                    show_uncertainty_categories = False
                    show_uncertainty_category_colors = False
                elif condition == "visible":
                    show_uncertainty_categories = True
                    show_uncertainty_category_colors = True
                standard_agent_run(env, test_env, task_name, PARENT_DIR, TIMESTAMP, log_data=log_data, participant_code=participant_code, testing_flow=TESTING, training_flow=TRAINING, group=group, roll_demos_in_training=ROLL_DEMO_IN_TRAINING, load_pretrained=load_pretrained, pretrained_model_step=pretrained_model_step, pretrained_model_path=pretrained_model_path, show_uncertainty_categories=show_uncertainty_categories, show_uncertainty_category_colors=show_uncertainty_category_colors, condition=condition, continue_experiment_training=False)
                
    with open(log_file, 'w', newline='') as file:
        writer_exp = csv.writer(file)
        header = ["Participant Code", "Condition", "Task", "Uncertainties", "Selected Starting State", "Selected State Uncertainty", "Selected State Category", "Uncertainty Categories"]
        writer_exp.writerow(header)
        for row in log_data:
            writer_exp.writerow(row)


if __name__ == "__main__":
    main()