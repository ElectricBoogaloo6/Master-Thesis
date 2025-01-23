import numpy as np
import gym
import panda_gym
from time import sleep
import os
import sys
import rospy

CURRENT_DIR = os.getcwd()
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR + '/envs/')
from task_envs import ReachWithObstacleV0Env, PickAndPlaceWithObstacleV0Env

from utils import ActionNormalizer, ResetWrapper, TimeFeatureWrapper, TimeLimitWrapper, reconstruct_state
from joystick_teleoperator import JoystickTele


def main():
    # task_name = 'ReachWithObstacleV0'
    task_name = 'PickAndPlaceWithObstacleV0'
    test_after_demo = True
    camera_distance = 0.8
    camera_yaw = 90.0
    camera_pitch = -70.0
    camera_target_position = np.array([0.0, 0.0, 0.0])

    joystick = JoystickTele()

    # create the environment
    if task_name == 'ReachWithObstacleV0':
        env = ReachWithObstacleV0Env(render=True, reward_type='modified_sparse', control_type='ee')
        env = ActionNormalizer(env)
        env = ResetWrapper(env=env)
        env = TimeFeatureWrapper(env=env, max_steps=50, test_mode=False)
        env = TimeLimitWrapper(env=env, max_steps=50)
        total_demo_num = 30
    elif task_name == 'PickAndPlaceWithObstacleV0':
        env = PickAndPlaceWithObstacleV0Env(render=True, reward_type='modified_sparse', control_type='ee')
        env = ActionNormalizer(env)
        env = ResetWrapper(env=env)
        env = TimeFeatureWrapper(env=env, max_steps=200, test_mode=False)
        env = TimeLimitWrapper(env=env, max_steps=200)
        total_demo_num = 60

    # adjust the camera view
    env.sim.physics_client.resetDebugVisualizerCamera(
        cameraDistance=camera_distance,
        cameraYaw=camera_yaw,
        cameraPitch=camera_pitch,
        cameraTargetPosition=camera_target_position,
    )

    goals = np.linspace(env.task.goal_range_low, env.task.goal_range_high, total_demo_num)
    demo_id = 0
    demo_state_trajs = []
    demo_action_trajs = []
    while demo_id < total_demo_num and not rospy.is_shutdown():
        goal_pos = goals[demo_id]
        if task_name == 'ReachWithObstacleV0':
            state = env.reset(whether_random=False, goal_pos=goal_pos, object_pos=None)
            reshaped_state = reconstruct_state(state)
        elif task_name == 'PickAndPlaceWithObstacleV0':
            object_pos = np.random.uniform(env.task.obj_range_low, env.task.obj_range_high)
            state = env.reset(whether_random=False, goal_pos=goal_pos, object_pos=object_pos)
            reshaped_state = reconstruct_state(state)

        done = False
        step = 0
        state_traj = []
        action_traj = []

        state_traj.append(np.ones(reshaped_state.shape[0]) * np.inf) # use np.inf as the sign for a new episode
        action_traj.append(np.ones(env.action_space.shape[0]) * np.inf)

        while not done and not rospy.is_shutdown():
            action = joystick.ee_displacement

            if not env.robot.block_gripper:
                gripper_width = np.array([joystick.gripper_cmd])
                action = np.concatenate((action, gripper_width))

            action_copy = action.copy()
            next_state, reward, done, info = env.step(action)
            env.render()

            reshaped_state = reconstruct_state(state)
            state_traj.append(reshaped_state)
            action_traj.append(action_copy)

            sleep(0.01)
            step += 1
            state = next_state
            print("step {}".format(step))

        if info['is_success']:
            demo_id += 1

            if demo_id == 1:
                demo_state_trajs = np.array(state_traj)
                demo_action_trajs = np.array(action_traj)
            else:
                demo_state_trajs = np.concatenate((demo_state_trajs, np.array(state_traj)), axis=0)
                demo_action_trajs = np.concatenate((demo_action_trajs, np.array(action_traj)), axis=0)
            print("[Demo {}]: finished".format(demo_id))
        else:
            print("[Demo {}]: failed, going to do it again".format(demo_id + 1))

        sleep(2.0)

    print("All demo have been collected!")
    env.close()

    joystick_demo_path = PARENT_DIR + '/demo_data/joystick_demo/' + task_name + '/'
    if not os.path.exists(joystick_demo_path):
        os.makedirs(joystick_demo_path)
    np.savetxt(joystick_demo_path + 'demo_state_trajs.csv', demo_state_trajs, delimiter=' ')
    np.savetxt(joystick_demo_path + 'demo_action_trajs.csv', demo_action_trajs, delimiter=' ')

    print('shape of demo state trajs: {}'.format(demo_state_trajs.shape))
    print('shape of demo action trajs: {}'.format(demo_action_trajs.shape))

    # test the performance of recorded demo
    if test_after_demo:
        if task_name == 'ReachWithObstacleV0':
            test_env = ReachWithObstacleV0Env(render=False, reward_type='modified_sparse', control_type='ee')
            test_env = ActionNormalizer(test_env)
            test_env = ResetWrapper(env=test_env)
            test_env = TimeFeatureWrapper(env=test_env, max_steps=50, test_mode=False)
            test_env = TimeLimitWrapper(env=test_env, max_steps=50)
        elif task_name == 'PickAndPlaceWithObstacleV0':
            test_env = PickAndPlaceWithObstacleV0Env(render=False, reward_type='modified_sparse', control_type='ee')
            test_env = ActionNormalizer(test_env)
            test_env = ResetWrapper(env=test_env)
            test_env = TimeFeatureWrapper(env=test_env, max_steps=200, test_mode=False)
            test_env = TimeLimitWrapper(env=test_env, max_steps=200)

        loaded_demo_state_trajs = np.genfromtxt(joystick_demo_path + 'demo_state_trajs.csv', delimiter=' ')
        loaded_demo_action_trajs = np.genfromtxt(joystick_demo_path + 'demo_action_trajs.csv', delimiter=' ')

        # first go through the loaded trajectory to get the index of episode starting signs
        starting_ids = []
        for i in range(loaded_demo_state_trajs.shape[0]):
            if loaded_demo_state_trajs[i][0] == np.inf:
                starting_ids.append(i)

        total_success_num = 0.0
        demo_id = 0
        for i in range(len(starting_ids)):
            starting_id = starting_ids[i]
            print("Demo {}: start at step index {}".format(demo_id + 1, starting_id))

            init_state = loaded_demo_state_trajs[starting_id + 1]
            goal_pos = init_state[-3:]
            if task_name == 'ReachWithObstacleV0':
                state = test_env.reset(whether_random=False, goal_pos=goal_pos, object_pos=None)
            elif task_name == 'PickAndPlaceWithObstacleV0':
                object_pos = init_state[7:10]
                state = test_env.reset(whether_random=False, goal_pos=goal_pos, object_pos=object_pos)

            step_idx = starting_id + 1
            if i < len(starting_ids) - 1:
                episode_length = starting_ids[i + 1] - starting_ids[i] - 1
            else:
                episode_length = loaded_demo_state_trajs.shape[0] - starting_ids[i]

            done = False
            step = 0
            while not done and step < episode_length:
                demo_action = loaded_demo_action_trajs[step_idx]
                _, _, done, info = test_env.step(demo_action)
                step_idx += 1
                step += 1

            demo_id += 1
            if info['is_success']:
                total_success_num += 1
                print("[Demo {}]: Success".format(demo_id))
            else:
                print("[Demo {}]: Failed".format(demo_id))

        print("Test finished: average success rate is {}".format(total_success_num / total_demo_num))

        # i = 0
        # total_success_num = 0.0
        # demo_id = 0
        # while i < loaded_demo_state_trajs.shape[0] and not rospy.is_shutdown():
        #     demo_state = loaded_demo_state_trajs[i]
        #
        #     if demo_state[0] == np.inf:
        #         i += 1
        #         init_state = loaded_demo_state_trajs[i]
        #         goal_pos = init_state[-3:]
        #         if task_name == 'ReachWithObstacleV0':
        #             state = test_env.reset(whether_random=False, goal_pos=goal_pos, object_pos=None)
        #         elif task_name == 'PickAndPlaceWithObstacleV0':
        #             object_pos = init_state[7:10]
        #             state = test_env.reset(whether_random=False, goal_pos=goal_pos, object_pos=object_pos)
        #
        #         done = False
        #         while not done:
        #             demo_action = loaded_demo_action_trajs[i]
        #             _, _, done, info = test_env.step(demo_action)
        #             i += 1
        #
        #         demo_id += 1
        #         if info['is_success']:
        #             total_success_num += 1
        #             print("[Demo {}]: Success".format(demo_id))
        #         else:
        #             print("[Demo {}]: Failed".format(demo_id))
        #
        # print("Test finished: average success rate is {}".format(total_success_num/total_demo_num))
        #




if __name__ == "__main__":
    main()
