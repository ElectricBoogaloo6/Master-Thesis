import numpy as np
import gym
import panda_gym
import torch
import argparse
import os
import sys
from time import sleep
# import imageio
from PIL import Image
import pybullet as p

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

from utils_awac import ActionNormalizer, ResetWrapper, TimeFeatureWrapper, TimeLimitWrapper, reconstruct_state
from phase_1_func import CustomResetWrapper, choose_model_step, choose_starting_position, choose_task, get_participant_configuration, initialize_envs_for_condition
from task_envs import Phase1TaskCentreEnv, Phase1TaskLeftEnv, Phase1TaskRightEnv
from phase_1_awac_agent import AWACAgent

def save_gif(frames, filename, size=(512, 512)):
    gif_path = os.path.join(RESULTS_DIR, filename)

    # Resize frames to optimize the GIF
    resized_frames = [Image.fromarray(frame).resize(size) for frame in frames]
    imageio.mimsave(gif_path, resized_frames, fps=30)
    print(f"Saved GIF to {gif_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    RECORD_GIF = False

    task_type = choose_task()
    print(f"Chosen task to test: {task_type}")
    model_step = choose_model_step()
    model_step = str(model_step)
    print(f"Chosen model step to test: {model_step}")

    if task_type == 1:
        env = Phase1TaskLeftEnv(render=True, reward_type="modified_sparse", control_type='ee')
        task_name = "Phase1TaskLeft"
    elif task_type == 2:
        env = Phase1TaskCentreEnv(render=True, reward_type="modified_sparse", control_type='ee')
        task_name = "Phase1TaskCentre"
    elif task_type == 3:
        env = Phase1TaskRightEnv(render=True, reward_type="modified_sparse", control_type='ee')
        task_name = "Phase1TaskRight"
    else:
        raise ValueError(f"Unknown task name: {task_type}")
    
    env = ActionNormalizer(env)
    env = CustomResetWrapper(env=env)
    env = TimeFeatureWrapper(env=env, max_steps=120, test_mode=False)
    env = TimeLimitWrapper(env=env, max_steps=120)

    """
    LOAD ANCHOR MODEL
    """
    method = 'awac_no_layernorm'
    agent = AWACAgent(env=env,
                    test_env=env,
                    task_name=task_name,
                    with_layer_norm=True,
                    # current_condition=None,
                    log_data=None,
                    writer=None,
                    participant_code=None,
                    testing_flow=False,
                    training_flow=False,
                    roll_demos_in_training=False)
    
    model_path = PARENT_DIR + '/models/' + task_name + '/' + method + '/max_demo_10'
    agent.ac.load_state_dict(torch.load(model_path + '/' + model_step + '_actor_critic.pth'))

    predefined_positions = env.task.predefined_obj_positions
    goal_pos = env.task.goal_position

    # Initialize uncertainties
    # traj_uncertainty_per_position = agent.initialize_starting_position_uncertainties(num_runs=1)

    chosen_position = np.random.choice(len(predefined_positions))  
    chosen_position_coords = predefined_positions[chosen_position]

    test_episode_reward = 0.0
    total_success_num = 0.0
    test_num = 20

    if RECORD_GIF:
        frames = []
    for i in range(test_num):
        chosen_position = choose_starting_position(env)
        chosen_position_coords = predefined_positions[chosen_position]
        print(f"Chosen starting position: {chosen_position}")

        obj_pos = chosen_position_coords
        state_ = env.reset(goal_pos=goal_pos, object_pos=obj_pos)
        done_ = False
        episode_reward = 0.0

        # Show uncertainties in environment before starting each episode
        # latest_logged_uncertainties = [traj_uncertainty_per_position[pos][-1] for pos in traj_uncertainty_per_position]
        # env.update_uncertainties(latest_logged_uncertainties)

        # Run the test rollout for the current chosen position
        rollout_traj = []
        while not done_:
            state_ = reconstruct_state(state_)
            action_ = agent.get_action(state_, True)
            next_state_, reward_, done_, info_ = env.step(action_)
            # next_state_ = reconstruct_state(next_state_)
            episode_reward += reward_
            test_episode_reward += reward_
            rollout_traj.append((state_, action_, reward_, next_state_, float(done_)))
            state_ = next_state_

            if RECORD_GIF:
                frames.append(env.render(mode='rgb_array'))
            sleep(0.01)
            # camera_data = p.getDebugVisualizerCamera(physicsClientId=env.client_id)

            # # Unpack only the values you need
            # width = camera_data[0]
            # height = camera_data[1]
            # view_matrix = camera_data[2]
            # projection_matrix = camera_data[3]
            # camera_target_position = camera_data[5]
            # camera_distance = camera_data[10]
            # camera_yaw = camera_data[8]
            # camera_pitch = camera_data[9]

            # # Optional: Print the values to check if they're correct
            # print("Camera Target Position:", camera_target_position)
            # print("Camera Distance:", camera_distance)
            # print("Camera Yaw:", camera_yaw)
            # print("Camera Pitch:", camera_pitch)

        # Log trajectory uncertainty after the episode
        # traj_uncertainty = agent.estimate_traj_uncertainty_td(rollout_traj)
        # logged_traj_uncertainty = np.log2(float(traj_uncertainty))
        # traj_uncertainty_per_position[chosen_position].append(logged_traj_uncertainty)

        # Check if the task was successful
        if info_['is_success']:
            total_success_num += 1.0
            print(f"[Test {i + 1}]: success!")
        else:
            print(f"[Test {i + 1}]: failure.")

    if RECORD_GIF:
        gif_filename = f"{model_step}_{task_name}.gif"
        save_gif(frames, gif_filename, size=(512, 512))

    average_episode_reward = test_episode_reward / test_num
    average_success_rate = total_success_num / test_num

    print(f"All finished: average success rate: {average_success_rate}, average episode reward: {average_episode_reward}")

if __name__ == '__main__':
    main()

