import numpy as np
import gym
import panda_gym
import os
import sys
import argparse
import rospy
from time import sleep

CURRENT_DIR = os.getcwd()
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR + '/envs/')
from task_envs import ReachWithObstacleV0Env, ReachWithObstacleV1Env, PushWithObstacleV0Env, PushWithObstacleV1Env
sys.path.append(PARENT_DIR + '/utils/')
from utils import ReplayBuffer, OUNoise, Actor, Critic, ActionNormalizer, ResetWrapper, TimeFeatureWrapper, TimeLimitWrapper, reconstruct_state, save_results
from joystick_teleoperator import JoystickTele

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', help='name of task')
    # parser.add_argument('--mode', help='mode of learning, scratch or transfer')

    return parser.parse_args()

def setup_demo_env(env):
    joystick = JoystickTele()

    # camera view parameters
    camera_distance = 0.8
    camera_yaw = 90.0
    camera_pitch = -70.0
    camera_target_position = np.array([0.0, 0.0, 0.0])

    # adjust the camera view
    env.sim.physics_client.resetDebugVisualizerCamera(
        cameraDistance=camera_distance,
        cameraYaw=camera_yaw,
        cameraPitch=camera_pitch,
        cameraTargetPosition=camera_target_position,
    )

    return joystick

def env_reset(obj_pos, ee_pos, env, task_name):
    if task_name == 'PushWithObstacleV0' or task_name == 'PushWithObstacleV1':
        goal_pos = env.task.goal_range_center  # since the goal for these two tasks are fixed

        state = env.reset(whether_random=False, goal_pos=goal_pos, object_pos=obj_pos, ee_pos=ee_pos)

    return state


def collect_demo(env, task_name, joystick, obj_pos, ee_pos=None):
    success_demo = False

    while not success_demo:
        state = env_reset(obj_pos=obj_pos, ee_pos=ee_pos, env=env, task_name=task_name)
        reshaped_state = reconstruct_state(state)

        done = False
        state_traj = []  # a list of 1d np array
        action_traj = []  # a list of 1d np array

        while not done and not rospy.is_shutdown():
            action = joystick.ee_displacement

            if not env.robot.block_gripper:
                gripper_width = np.array([joystick.gripper_cmd])
                action = np.concatenate((action, gripper_width))

            action_copy = action.copy()
            next_state, reward, done, info = env.step(action)
            env.render()

            reshaped_state = reconstruct_state(state)
            reshaped_next_state = reconstruct_state(next_state)

            state_traj.append(reshaped_state)
            action_traj.append(action_copy)

            sleep(0.01)
            state = next_state

        if info['is_success']:
            success_demo = True
            print("[Demo]: succeeded")
        else:
            print("[Demo]: failed, going to do it again")
            sleep(2.0)

    state_traj = np.array(state_traj)  # 2d np array in the form of (total_episode_steps, state_dims)
    action_traj = np.array(action_traj)  # 2d np array in the form of (total_episode_steps, action_dims)

    return state_traj, action_traj

def whether_beyond_workspace(workspace_area, pos):
    if pos[0] >= workspace_area['min_x'] and pos[0] <= workspace_area['max_x'] \
            and pos[1] >= workspace_area['min_y'] and pos[1] <= workspace_area['max_y']:
        return False
    else:
        return True

def whether_in_collision_area(obstacle_area, pos):
    if pos[0] >= obstacle_area['min_x'] and pos[0] <= obstacle_area['max_x'] \
            and pos[1] >= obstacle_area['min_y'] and pos[1] <= obstacle_area['max_y']:
        return True
    else:
        return False


def sample_around_center_state(task_name, sample_num, center_state, object_sample_radius=0.05, ee_sample_radius=0.00, ee_sample_delta_z=0.02):
    if task_name == 'PushWithObstacleV0':
        object_center_pos = center_state[6:9]
        ee_center_pos = center_state[0:3]

        print("center object pos: {}".format(object_center_pos))

        workspace_area = {'min_x': -0.25, 'max_x': 0.25, 'min_y': -0.25, 'max_y': 0.25}

        obstacle_1_pos = np.array([-0.11, 0.0, 0.01])
        obstacle_1_dims = np.array([0.02, 0.1, 0.02])
        object_size = 0.04 / 2.0 + 0.005
        obstacle_1_area = {'min_x': obstacle_1_pos[0] - obstacle_1_dims[0] / 2.0 - object_size,
                           'max_x': obstacle_1_pos[0] + obstacle_1_dims[0] / 2.0 + object_size,
                           'min_y': obstacle_1_pos[1] - obstacle_1_dims[1] / 2.0 - object_size,
                           'max_y': obstacle_1_pos[1] + obstacle_1_dims[1] / 2.0 + object_size}

        # sample positions for the object
        sampled_obj_pos_list = []  # a list of 1d np array of shape (x,y,z)
        for i in range(sample_num):
            valid_sample = False

            while not valid_sample:
                r = np.random.uniform(0.0, object_sample_radius, 1)[0]
                theta = np.random.uniform(0.0, 2.0 * np.pi, 1)[0]
                delta_x = r * np.cos(theta)
                delta_y = r * np.sin(theta)
                sampled_obj_pos = np.array([object_center_pos[0] + delta_x,
                                            object_center_pos[1] + delta_y,
                                            0.04 / 2.0])

                beyond_workspace = whether_beyond_workspace(workspace_area, sampled_obj_pos)
                in_collision = whether_in_collision_area(obstacle_1_area, sampled_obj_pos)
                if (not beyond_workspace) and (not in_collision):
                    valid_sample = True
                else:
                    print("[Sample around center state]: invalid sample of object pos")

            sampled_obj_pos_list.append(sampled_obj_pos)

        # sample positions for the ee
        sampled_ee_pos_list = []  # a list of 1d np array of shape (x,y,z)
        for i in range(sample_num):
            valid_sample = False

            while not valid_sample:
                # r = np.random.uniform(0.0, ee_sample_radius, 1)[0]
                # theta = np.random.uniform(0.0, 2.0 * np.pi, 1)[0]
                # delta_z = np.random.uniform(0.03, 0.03 + ee_sample_delta_z, 1)[
                #     0]  # make it a little higher than the obstacle
                #
                # delta_x = r * np.cos(theta)
                # delta_y = r * np.sin(theta)
                # sampled_ee_pos = np.array([ee_center_pos[0] + delta_x,
                #                            ee_center_pos[1] + delta_y,
                #                            ee_center_pos[2] + delta_z])

                sampled_ee_pos = ee_center_pos.copy()
                if sampled_ee_pos[2] < 0.05:
                    sampled_ee_pos[2] = 0.05

                valid_sample = True

                # beyond_workspace = whether_beyond_workspace(workspace_area, sampled_ee_pos)
                # if not beyond_workspace:
                #     valid_sample = True
                # else:
                #     print("[Sample around center state]: invalid sample of ee pos")

            sampled_ee_pos_list.append(sampled_ee_pos)

        return sampled_obj_pos_list, sampled_ee_pos_list


def visualize_curricula(center_state_list, sampled_obj_list, sampled_ee_list, env):
    total_curricula_num = len(center_state_list)
    color_shade_list = np.linspace(0.1, 0.9, total_curricula_num)

    for i in range(total_curricula_num):

        # if i < total_curricula_num - 1:
        #     continue

        # visualize center state as green ghost cube, with different shade of green for different curricula
        env.sim.create_box(
            body_name="center_state_" + str(i + 1),
            half_extents=np.ones(3) * 0.01,
            mass=0.0,
            ghost=True,
            position=center_state_list[i],
            rgba_color=np.array([0.1, 0.9, 0.1, color_shade_list[i]]),
        )

        # visualize sampled object positions as blue ghost cubes, with different shade of blue for different curricula
        for k in range(len(sampled_obj_list[i])):
            obj_pos = sampled_obj_list[i][k]
            env.sim.create_box(
                body_name="obj_pos_" + str(i + 1) + '_sample_' + str(k + 1),
                half_extents=np.ones(3) * 0.005,
                mass=0.0,
                ghost=True,
                position=obj_pos,
                rgba_color=np.array([0.1, 0.1, 0.9, color_shade_list[i]]),
            )

        # visualize sampled ee positions as red ghost spheres, with different shades of red for different curricula
        print("total num of sampled ee pos: {}".format(len(sampled_ee_list[i])))
        for j in range(len(sampled_ee_list[i])):
            ee_pos = sampled_ee_list[i][j]
            env.sim.create_sphere(
                body_name="center_state_" + str(i + 1) + '_sample_' + str(j + 1),
                radius = 0.01,
                mass=0.0,
                ghost=True,
                position=ee_pos,
                rgba_color=np.array([0.9, 0.1, 0.1, color_shade_list[i]])
            )


def main():
    args = argparser()

    seed = 6
    np.random.seed(seed)

    task_name = args.task_name
    if task_name == 'PushWithObstacleV0':
        env = PushWithObstacleV0Env(render=True, reward_type='modified_sparse', control_type='ee')
        env = ActionNormalizer(env)
        env = ResetWrapper(env=env)
        env = TimeFeatureWrapper(env=env, max_steps=120, test_mode=False)
        env = TimeLimitWrapper(env=env, max_steps=120)

        print("created the env")


    base_demo = []
    init_obj_pos = np.array([-0.15, 0.0, 0.02])
    if len(base_demo) == 0:
        joystick = setup_demo_env(env)
        demo_state_traj, demo_action_traj = collect_demo(env=env,
                                                         task_name=task_name,
                                                         joystick=joystick,
                                                         obj_pos=init_obj_pos,
                                                         ee_pos=None)
        demo_dict = {"state_trajectory": demo_state_traj,
                     "action_trajectory": demo_action_traj}
        base_demo.append(demo_dict)
        print("[Base demo]: finished collecting")

    env.sim.physics_client.removeBody(env.sim._bodies_idx['object'])
    env.sim.physics_client.removeBody(env.sim._bodies_idx['target'])
    env.robot.reset()

    # simulate curriculum update process just along the base demo
    stage = 0
    delta_stage = 10
    score_points_num = 10
    stage += delta_stage
    base_demo_length = base_demo[0]["state_trajectory"].shape[0]
    center_state_list = []
    sampled_obj_pos_list = []
    sampled_ee_pos_list = []
    while stage <= base_demo_length:
        center_state = base_demo[0]['state_trajectory'][base_demo_length - stage].copy()
        print("***********************************")
        print("[Stage {}]: Going to sample score points".format(stage))
        init_obj_pos_list, init_ee_pos_list = sample_around_center_state(task_name=task_name,
                                                                         sample_num=score_points_num,
                                                                         center_state=center_state)

        center_state_list.append(center_state[6:9].copy())
        sampled_obj_pos_list.append(init_obj_pos_list.copy())
        sampled_ee_pos_list.append(init_ee_pos_list.copy())

        new_stage = stage + delta_stage

        if stage < base_demo_length and new_stage > base_demo_length:
            new_stage = base_demo_length

        stage = new_stage

    visualize_curricula(center_state_list, sampled_obj_pos_list, sampled_ee_pos_list, env)

    while not rospy.is_shutdown():
        # visualize_curricula(center_state_list, sampled_obj_pos_list, sampled_ee_pos_list, env)
        env.render()
        # continue


if __name__ == '__main__':
    main()



