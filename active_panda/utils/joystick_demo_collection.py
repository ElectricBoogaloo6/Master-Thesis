import numpy as np
import gym
import panda_gym
from time import sleep
import os
import sys
import rospy
import argparse

CURRENT_DIR = os.getcwd()
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR + '/envs/')
from task_envs import ReachWithObstacleV0Env, ReachWithObstacleV1Env, PushWithObstacleV0Env, PushWithObstacleV1Env

from active_panda.algs.utils_awac import ActionNormalizer, ResetWrapper, TimeFeatureWrapper, TimeLimitWrapper, reconstruct_state, whether_in_collision_area
from joystick_teleoperator import JoystickTele

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', help='name of task')
    parser.add_argument('--mode', help='mode of initial state space, or whole state space')

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


def main():
    args = argparser()

    task_name = args.task_name
    mode = args.mode

    state_traj = []  # a list of 1d np array
    action_traj = []  # a list of 1d np array
    next_state_traj = []
    reward_traj = []
    done_traj = []

    demo_path = PARENT_DIR + '/' + 'demo_data/demo_pool/' + mode + '/' + task_name + '/'
    if not os.path.exists(demo_path):
        os.makedirs(demo_path)

    if task_name == 'PushWithObstacleV0':
        env = PushWithObstacleV0Env(render=True, reward_type='modified_sparse', control_type='ee')
        env = ActionNormalizer(env)
        env = ResetWrapper(env=env)
        env = TimeFeatureWrapper(env=env, max_steps=120, test_mode=False)
        env = TimeLimitWrapper(env=env, max_steps=120)

        test_env = PushWithObstacleV0Env(render=False, reward_type='modified_sparse', control_type='ee')
        test_env = ActionNormalizer(test_env)
        test_env = ResetWrapper(env=test_env)
        test_env = TimeFeatureWrapper(env=test_env, max_steps=120, test_mode=False)
        test_env = TimeLimitWrapper(env=test_env, max_steps=120)

        joystick = setup_demo_env(env=env)

        goal_pos = env.task.goal_range_center

        obstacle_1_pos = np.array([-0.11, 0.0, 0.01])
        obstacle_1_dims = np.array([0.02, 0.1, 0.02])
        object_size = 0.04 / 2.0 + 0.005
        obstacle_1_area = {'min_x': obstacle_1_pos[0] - obstacle_1_dims[0] / 2.0 - object_size,
                           'max_x': obstacle_1_pos[0] + obstacle_1_dims[0] / 2.0 + object_size,
                           'min_y': obstacle_1_pos[1] - obstacle_1_dims[1] / 2.0 - object_size,
                           'max_y': obstacle_1_pos[1] + obstacle_1_dims[1] / 2.0 + object_size}
        obstacle_areas = []
        obstacle_areas.append(obstacle_1_area)
    elif task_name == 'PushWithObstacleV1':
        env = PushWithObstacleV1Env(render=True, reward_type='modified_sparse', control_type='ee')
        env = ActionNormalizer(env)
        env = ResetWrapper(env=env)
        env = TimeFeatureWrapper(env=env, max_steps=120, test_mode=False)
        env = TimeLimitWrapper(env=env, max_steps=120)

        test_env = PushWithObstacleV1Env(render=False, reward_type='modified_sparse', control_type='ee')
        test_env = ActionNormalizer(test_env)
        test_env = ResetWrapper(env=test_env)
        test_env = TimeFeatureWrapper(env=test_env, max_steps=120, test_mode=False)
        test_env = TimeLimitWrapper(env=test_env, max_steps=120)

        joystick = setup_demo_env(env=env)

        goal_pos = env.task.goal_range_center

        obstacle_areas = []
        obstacle_1_pos = np.array([-0.05, 0.01, 0.01])
        obstacle_1_dims = np.array([0.01, 0.1, 0.02])
        object_size = 0.04 / 2.0 + 0.005
        obstacle_1_area = {'min_x': obstacle_1_pos[0] - obstacle_1_dims[0] / 2.0 - object_size,
                           'max_x': obstacle_1_pos[0] + obstacle_1_dims[0] / 2.0 + object_size,
                           'min_y': obstacle_1_pos[1] - obstacle_1_dims[1] / 2.0 - object_size,
                           'max_y': obstacle_1_pos[1] + obstacle_1_dims[1] / 2.0 + object_size}
        obstacle_areas.append(obstacle_1_area)

        obstacle_2_pos = np.array([0.05, -0.1, 0.01])
        obstacle_2_dims = np.array([0.01, 0.1, 0.02])
        object_size = 0.04 / 2.0 + 0.005
        obstacle_2_area = {'min_x': obstacle_2_pos[0] - obstacle_2_dims[0] / 2.0 - object_size,
                           'max_x': obstacle_2_pos[0] + obstacle_2_dims[0] / 2.0 + object_size,
                           'min_y': obstacle_2_pos[1] - obstacle_2_dims[1] / 2.0 - object_size,
                           'max_y': obstacle_2_pos[1] + obstacle_2_dims[1] / 2.0 + object_size}
        obstacle_areas.append(obstacle_2_area)
    elif task_name == 'ReachWithObstacleV0':
        env = ReachWithObstacleV0Env(render=True, reward_type='modified_sparse', control_type='ee')
        env = ActionNormalizer(env)
        env = ResetWrapper(env=env)
        env = TimeFeatureWrapper(env=env, max_steps=100, test_mode=False)
        env = TimeLimitWrapper(env=env, max_steps=100)

        test_env = ReachWithObstacleV0Env(render=False, reward_type='modified_sparse', control_type='ee')
        test_env = ActionNormalizer(test_env)
        test_env = ResetWrapper(env=test_env)
        test_env = TimeFeatureWrapper(env=test_env, max_steps=100, test_mode=False)
        test_env = TimeLimitWrapper(env=test_env, max_steps=100)

        joystick = setup_demo_env(env=env)

        goal_pos = env.task.goal_range_center
        obstacle_areas = []
        obstacle_1_pos = np.array([-0.05, -0.1, 0.075])
        obstacle_1_dims = np.array([0.02, 0.2, 0.15])
        object_size = np.array([0.05, 0.1, 0.03])
        obstacle_1_area = {'min_x': obstacle_1_pos[0] - obstacle_1_dims[0] / 2.0 - object_size[0],
                           'max_x': obstacle_1_pos[0] + obstacle_1_dims[0] / 2.0 + object_size[0],
                           'min_y': obstacle_1_pos[1] - obstacle_1_dims[1] / 2.0 - object_size[1],
                           'max_y': obstacle_1_pos[1] + obstacle_1_dims[1] / 2.0 + object_size[1],
                           'min_z': 0.0,
                           'max_z': obstacle_1_pos[2] + obstacle_1_dims[2] / 2.0 + object_size[2]}
        obstacle_areas.append(obstacle_1_area)
    elif task_name == 'ReachWithObstacleV1':
        env = ReachWithObstacleV1Env(render=True, reward_type='modified_sparse', control_type='ee')
        env = ActionNormalizer(env)
        env = ResetWrapper(env=env)
        env = TimeFeatureWrapper(env=env, max_steps=100, test_mode=False)
        env = TimeLimitWrapper(env=env, max_steps=100)

        test_env = ReachWithObstacleV1Env(render=False, reward_type='modified_sparse', control_type='ee')
        test_env = ActionNormalizer(test_env)
        test_env = ResetWrapper(env=test_env)
        test_env = TimeFeatureWrapper(env=test_env, max_steps=100, test_mode=False)
        test_env = TimeLimitWrapper(env=test_env, max_steps=100)

        joystick = setup_demo_env(env=env)

        goal_pos = env.task.goal_range_center
        obstacle_areas = []
        obstacle_1_pos = np.array([-0.05, -0.1, 0.075])
        obstacle_1_dims = np.array([0.02, 0.2, 0.15])
        object_size = np.array([0.05, 0.1, 0.03])
        obstacle_1_area = {'min_x': obstacle_1_pos[0] - obstacle_1_dims[0] / 2.0 - object_size[0],
                           'max_x': obstacle_1_pos[0] + obstacle_1_dims[0] / 2.0 + object_size[0],
                           'min_y': obstacle_1_pos[1] - obstacle_1_dims[1] / 2.0 - object_size[1],
                           'max_y': obstacle_1_pos[1] + obstacle_1_dims[1] / 2.0 + object_size[1],
                           'min_z': 0.0,
                           'max_z': obstacle_1_pos[2] + obstacle_1_dims[2] / 2.0 + object_size[2]}
        obstacle_areas.append(obstacle_1_area)

        obstacle_2_pos = np.array([0.0, 0.08, 0.09])
        obstacle_2_dims = np.array([0.05, 0.05, 0.18])
        object_size = np.array([0.05, 0.1, 0.03])
        obstacle_2_area = {'min_x': obstacle_2_pos[0] - obstacle_2_dims[0] / 2.0 - object_size[0],
                           'max_x': obstacle_2_pos[0] + obstacle_2_dims[0] / 2.0 + object_size[0],
                           'min_y': obstacle_2_pos[1] - obstacle_2_dims[1] / 2.0 - object_size[1],
                           'max_y': obstacle_2_pos[1] + obstacle_2_dims[1] / 2.0 + object_size[1],
                           'min_z': 0.0,
                           'max_z': obstacle_2_pos[2] + obstacle_2_dims[2] / 2.0 + object_size[2]}
        obstacle_areas.append(obstacle_2_area)

    if mode == 'whole_space':
        if task_name == 'PushWithObstacleV0' or task_name == 'PushWithObstacleV1':
            obj_xs = np.arange(start=-0.15, stop=0.15 + 0.01, step=0.05)
            obj_ys = np.arange(start=-0.15, stop=0.15 + 0.01, step=0.05)
            demo_id = 0

            for obj_x in obj_xs:
                for obj_y in obj_ys:
                    obj_pos = np.array([obj_x, obj_y, 0.02])

                    in_collision = False
                    for obstacle_area in obstacle_areas:
                        if whether_in_collision_area(obstacle_area=obstacle_area, pos=obj_pos):
                            in_collision = True
                            print("Starting state is in collision area, going to record next demo")
                            break

                    if not in_collision:
                        success_demo = False

                        while not success_demo:
                            state = env.reset(whether_random=False, goal_pos=goal_pos, object_pos=obj_pos, ee_pos=None)
                            reshaped_state = reconstruct_state(state)

                            done = False

                            curr_state_traj = []
                            curr_action_traj = []
                            curr_next_state_traj = []
                            curr_reward_traj = []
                            curr_done_traj = []

                            curr_state_traj.append(np.ones(reshaped_state.shape[0]) * np.inf)
                            curr_action_traj.append(np.ones(env.action_space.shape[0]) * np.inf)
                            curr_next_state_traj.append(np.ones(reshaped_state.shape[0]) * np.inf)
                            curr_reward_traj.append(np.ones(1) * np.inf)
                            curr_done_traj.append(np.ones(1) * np.inf)

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

                                curr_state_traj.append(reshaped_state.copy())
                                curr_action_traj.append(action_copy)
                                curr_next_state_traj.append(reshaped_next_state.copy())
                                curr_reward_traj.append(np.array([reward]))
                                curr_done_traj.append(np.array([float(done)]))

                                sleep(0.01)
                                state = next_state

                            if info['is_success']:
                                success_demo = True
                                demo_id += 1
                                print("[Demo {}]: succeeded".format(demo_id))
                                sleep(2.0)

                                for i in range(len(curr_state_traj)):
                                    state_traj.append(curr_state_traj[i])
                                    action_traj.append(curr_action_traj[i])
                                    next_state_traj.append(curr_next_state_traj[i])
                                    reward_traj.append(curr_reward_traj[i])
                                    done_traj.append(curr_done_traj[i])

                                print("demo saved")
                            else:
                                print("[Demo {}]: failed, going to do it again".format(demo_id + 1))
                                sleep(2.0)
        elif task_name == 'ReachWithObstacleV0' or task_name == 'ReachWithObstacleV1':
            obj_xs = np.arange(start=-0.15, stop=0.15 + 0.01, step=0.05)
            obj_ys = np.arange(start=-0.15, stop=0.15 + 0.01, step=0.05)
            demo_id = 0

            for obj_x in obj_xs:
                for obj_y in obj_ys:
                    ee_pos = np.array([obj_x, obj_y, 0.02])

                    in_collision = False
                    for obstacle_area in obstacle_areas:
                        if whether_in_collision_area(obstacle_area=obstacle_area, pos=ee_pos):
                            in_collision = True
                            print("Starting state is in collision area, going to record next demo")
                            break

                    if not in_collision:
                        success_demo = False

                        while not success_demo:
                            state = env.reset(whether_random=False, goal_pos=goal_pos, object_pos=None, ee_pos=ee_pos)
                            reshaped_state = reconstruct_state(state)

                            done = False

                            curr_state_traj = []
                            curr_action_traj = []
                            curr_next_state_traj = []
                            curr_reward_traj = []
                            curr_done_traj = []

                            curr_state_traj.append(np.ones(reshaped_state.shape[0]) * np.inf)
                            curr_action_traj.append(np.ones(env.action_space.shape[0]) * np.inf)
                            curr_next_state_traj.append(np.ones(reshaped_state.shape[0]) * np.inf)
                            curr_reward_traj.append(np.ones(1) * np.inf)
                            curr_done_traj.append(np.ones(1) * np.inf)

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

                                curr_state_traj.append(reshaped_state.copy())
                                curr_action_traj.append(action_copy)
                                curr_next_state_traj.append(reshaped_next_state.copy())
                                curr_reward_traj.append(np.array([reward]))
                                curr_done_traj.append(np.array([float(done)]))

                                sleep(0.01)
                                state = next_state

                            if info['is_success']:
                                success_demo = True
                                demo_id += 1
                                print("[Demo {}]: succeeded".format(demo_id))
                                sleep(2.0)

                                for i in range(len(curr_state_traj)):
                                    state_traj.append(curr_state_traj[i])
                                    action_traj.append(curr_action_traj[i])
                                    next_state_traj.append(curr_next_state_traj[i])
                                    reward_traj.append(curr_reward_traj[i])
                                    done_traj.append(curr_done_traj[i])

                                print("demo saved")
                            else:
                                print("[Demo {}]: failed, going to do it again".format(demo_id + 1))
                                sleep(2.0)
        state_traj = np.array(state_traj)  # 2d np array in the form of (total_steps, state_dims)
        action_traj = np.array(action_traj)  # 2d np array in the form of (total_steps, action_dims)
        next_state_traj = np.array(next_state_traj)
        reward_traj = np.array(reward_traj)
        done_traj = np.array(done_traj)

        np.savetxt(demo_path + 'state_traj.csv', state_traj, delimiter=' ')
        np.savetxt(demo_path + 'action_traj.csv', action_traj, delimiter=' ')
        np.savetxt(demo_path + 'next_state_traj.csv', next_state_traj, delimiter=' ')
        np.savetxt(demo_path + 'reward_traj.csv', reward_traj, delimiter=' ')
        np.savetxt(demo_path + 'done_traj.csv', done_traj, delimiter=' ')

        print("[Online demo query]: All demo have been collected and saved, going to close the demo env")
    else:
        if task_name == 'PushWithObstacleV0' or task_name == 'PushWithObstacleV1':
            obj_ys = np.linspace(start=-0.15, stop=0.15, num=30)
            demo_id = 0

            for obj_y in obj_ys:
                obj_pos = np.array([-0.15, obj_y, 0.02])

                success_demo = False

                while not success_demo:
                    state = env.reset(whether_random=False, goal_pos=goal_pos, object_pos=obj_pos, ee_pos=None)
                    reshaped_state = reconstruct_state(state)

                    done = False

                    curr_state_traj = []
                    curr_action_traj = []
                    curr_next_state_traj = []
                    curr_reward_traj = []
                    curr_done_traj = []

                    curr_state_traj.append(np.ones(reshaped_state.shape[0]) * np.inf)
                    curr_action_traj.append(np.ones(env.action_space.shape[0]) * np.inf)
                    curr_next_state_traj.append(np.ones(reshaped_state.shape[0]) * np.inf)
                    curr_reward_traj.append(np.ones(1) * np.inf)
                    curr_done_traj.append(np.ones(1) * np.inf)

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

                        curr_state_traj.append(reshaped_state.copy())
                        curr_action_traj.append(action_copy)
                        curr_next_state_traj.append(reshaped_next_state.copy())
                        curr_reward_traj.append(np.array([reward]))
                        curr_done_traj.append(np.array([float(done)]))

                        sleep(0.01)
                        state = next_state

                    if info['is_success']:
                        success_demo = True
                        demo_id += 1
                        print("[Demo {}]: succeeded".format(demo_id))
                        sleep(2.0)

                        for i in range(len(curr_state_traj)):
                            state_traj.append(curr_state_traj[i])
                            action_traj.append(curr_action_traj[i])
                            next_state_traj.append(curr_next_state_traj[i])
                            reward_traj.append(curr_reward_traj[i])
                            done_traj.append(curr_done_traj[i])

                        print("demo saved")
                    else:
                        print("[Demo {}]: failed, going to do it again".format(demo_id + 1))
                        sleep(2.0)
        elif task_name == 'ReachWithObstacleV0' or task_name == 'ReachWithObstacleV1':
            obj_ys = np.linspace(start=-0.15, stop=0.15, num=30)
            demo_id = 0

            for obj_y in obj_ys:
                ee_pos = np.array([0.15, obj_y, 0.02])

                success_demo = False

                while not success_demo:
                    state = env.reset(whether_random=False, goal_pos=goal_pos, object_pos=None, ee_pos=ee_pos)
                    reshaped_state = reconstruct_state(state)

                    done = False

                    curr_state_traj = []
                    curr_action_traj = []
                    curr_next_state_traj = []
                    curr_reward_traj = []
                    curr_done_traj = []

                    curr_state_traj.append(np.ones(reshaped_state.shape[0]) * np.inf)
                    curr_action_traj.append(np.ones(env.action_space.shape[0]) * np.inf)
                    curr_next_state_traj.append(np.ones(reshaped_state.shape[0]) * np.inf)
                    curr_reward_traj.append(np.ones(1) * np.inf)
                    curr_done_traj.append(np.ones(1) * np.inf)

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

                        curr_state_traj.append(reshaped_state.copy())
                        curr_action_traj.append(action_copy)
                        curr_next_state_traj.append(reshaped_next_state.copy())
                        curr_reward_traj.append(np.array([reward]))
                        curr_done_traj.append(np.array([float(done)]))

                        sleep(0.01)
                        state = next_state

                    if info['is_success']:
                        success_demo = True
                        demo_id += 1
                        print("[Demo {}]: succeeded".format(demo_id))
                        sleep(2.0)

                        for i in range(len(curr_state_traj)):
                            state_traj.append(curr_state_traj[i])
                            action_traj.append(curr_action_traj[i])
                            next_state_traj.append(curr_next_state_traj[i])
                            reward_traj.append(curr_reward_traj[i])
                            done_traj.append(curr_done_traj[i])

                        print("demo saved")
                    else:
                        print("[Demo {}]: failed, going to do it again".format(demo_id + 1))
                        sleep(2.0)

        state_traj = np.array(state_traj)  # 2d np array in the form of (total_steps, state_dims)
        action_traj = np.array(action_traj)  # 2d np array in the form of (total_steps, action_dims)
        next_state_traj = np.array(next_state_traj)
        reward_traj = np.array(reward_traj)
        done_traj = np.array(done_traj)

        np.savetxt(demo_path + 'state_traj.csv', state_traj, delimiter=' ')
        np.savetxt(demo_path + 'action_traj.csv', action_traj, delimiter=' ')
        np.savetxt(demo_path + 'next_state_traj.csv', next_state_traj, delimiter=' ')
        np.savetxt(demo_path + 'reward_traj.csv', reward_traj, delimiter=' ')
        np.savetxt(demo_path + 'done_traj.csv', done_traj, delimiter=' ')

        print("[Online demo query]: All demo have been collected and saved, going to close the demo env")


if __name__ == '__main__':
    main()
