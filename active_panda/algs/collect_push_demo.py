import numpy as np
import gym
import panda_gym
import os
import sys
from sb3_contrib import TQC
from time import sleep

CURRENT_DIR = os.getcwd()
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR + '/envs/')
from task_envs import PushWithObstacleV0Env, PushWithObstacleV1Env

from utils import ActionNormalizer, ResetWrapper, TimeFeatureWrapper, TimeLimitWrapper, reconstruct_state



def main():
    # task_name = 'PushWithObstacleV0'
    task_name = 'PushWithObstacleV1'
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    test_after_demo = True

    if task_name == 'PushWithObstacleV0':
        loader_env = gym.make("Panda" + 'Push' + "-v2", render=False)
        loader_env = ActionNormalizer(loader_env)
        loader_env = ResetWrapper(env=loader_env)
        loader_env = TimeFeatureWrapper(env=loader_env, max_steps=50, test_mode=False)
        loader_env = TimeLimitWrapper(env=loader_env, max_steps=50)

        env = PushWithObstacleV0Env(render=False, reward_type='modified_sparse', control_type='ee')
        env = ActionNormalizer(env)
        env = ResetWrapper(env=env)
        env = TimeFeatureWrapper(env=env, max_steps=50, test_mode=False)
        env = TimeLimitWrapper(env=env, max_steps=50)

        print("Load oracle for PushWithObstacle-V0...")
        oracle = TQC.load(PARENT_DIR + '/models/oracles/' + task_name + "/oracle_model",
                          custom_objects=custom_objects,
                          env=loader_env)
        oracle.set_env(env)

        total_demo_num = 30
    elif task_name == 'PushWithObstacleV1':
        loader_env = gym.make("Panda" + 'Push' + "-v2", render=False)
        loader_env = ActionNormalizer(loader_env)
        loader_env = ResetWrapper(env=loader_env)
        loader_env = TimeFeatureWrapper(env=loader_env, max_steps=50, test_mode=False)

        env = PushWithObstacleV1Env(render=False, reward_type='modified_sparse', control_type='ee')
        env = ActionNormalizer(env)
        env = ResetWrapper(env=env)
        env = TimeFeatureWrapper(env=env, max_steps=50, test_mode=False)
        env = TimeLimitWrapper(env=env, max_steps=50)

        print("Load oracle for PushWithObstacle-V1...")
        oracle = TQC.load(PARENT_DIR + '/models/oracles/' + task_name + "/oracle_model",
                          custom_objects=custom_objects, env=loader_env)
        oracle.set_env(env)

        total_demo_num = 60

    object_pos_list = np.linspace(env.task.obj_range_low, env.task.obj_range_high, total_demo_num)
    goal_pos = np.random.uniform(env.task.goal_range_low, env.task.goal_range_high)
    demo_id = 0
    demo_state_trajs = []
    demo_action_trajs = []
    current_step_idx = 0
    while demo_id < total_demo_num:
        object_pos = object_pos_list[demo_id]
        state = env.reset(whether_random=False, goal_pos=goal_pos, object_pos=object_pos)
        reshaped_state = reconstruct_state(state)

        done = False
        step = 0
        state_traj = []
        action_traj = []

        print("Demo {}: start at index {}".format(demo_id + 1, current_step_idx))
        state_traj.append(np.ones(reshaped_state.shape[0]) * np.inf)  # use np.inf as the sign for a new episode
        action_traj.append(np.ones(env.action_space.shape[0]) * np.inf)

        while not done:
            action, _ = oracle.predict(state, deterministic=True)
            action = np.array(action)
            next_state, reward, done, info = env.step(action)
            print(action)
            # env.render()

            reshaped_state = reconstruct_state(state)
            state_traj.append(reshaped_state)
            action_traj.append(action)

            # sleep(0.01)
            step += 1
            current_step_idx += 1
            state = next_state
            # print("step {}".format(step))

        current_step_idx += 1

        if info['is_success']:
            print("[Demo {}]: succeeded".format(demo_id + 1))
        else:
            print("[Demo {}]: failed".format(demo_id + 1))

        demo_id += 1
        if demo_id == 1:
            demo_state_trajs = np.array(state_traj)
            demo_action_trajs = np.array(action_traj)
        else:
            demo_state_trajs = np.concatenate((demo_state_trajs, np.array(state_traj)), axis=0)
            demo_action_trajs = np.concatenate((demo_action_trajs, np.array(action_traj)), axis=0)

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
        if task_name == 'PushWithObstacleV0':
            test_env = PushWithObstacleV0Env(render=False, reward_type='modified_sparse', control_type='ee')
            test_env = ActionNormalizer(test_env)
            test_env = ResetWrapper(env=test_env)
            test_env = TimeFeatureWrapper(env=test_env, max_steps=50, test_mode=False)
            test_env = TimeLimitWrapper(env=test_env, max_steps=50)
        elif task_name == 'PushWithObstacleV1':
            test_env = PushWithObstacleV1Env(render=False, reward_type='modified_sparse', control_type='ee')
            test_env = ActionNormalizer(test_env)
            test_env = ResetWrapper(env=test_env)
            test_env = TimeFeatureWrapper(env=test_env, max_steps=50, test_mode=False)
            test_env = TimeLimitWrapper(env=test_env, max_steps=50)

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
            object_pos = init_state[6:9]
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
                print(demo_action)
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


if __name__ == "__main__":
    main()
