import numpy as np
import gym
import panda_gym
import torch
import argparse
import os
import sys
from time import sleep


CURRENT_DIR = os.getcwd()
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR + '/envs/')
from task_envs import ReachWithObstacleV0Env, ReachWithObstacleV1Env, PushWithObstacleV0Env, PushWithObstacleV1Env
from active_panda.algs.utils_awac import ActionNormalizer, ResetWrapper, TimeFeatureWrapper, TimeLimitWrapper, reconstruct_state, save_results
sys.path.append(PARENT_DIR + '/algs/')
from active_curr_awac import Active_Curr_AWAC


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', help='name of task')
    parser.add_argument('--step', help='step of the model to load')

    return parser.parse_args()


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(device)

    args = argparser()

    task_name = args.task_name
    step = int(args.step) * int(1e3)
    step = str(step)
    if task_name == 'PushWithObstacleV0':
        env = PushWithObstacleV0Env(render=True, reward_type='modified_sparse', control_type='ee')
        env = ActionNormalizer(env)
        env = ResetWrapper(env=env)
        env = TimeFeatureWrapper(env=env, max_steps=120, test_mode=False)
        env = TimeLimitWrapper(env=env, max_steps=120)
    elif task_name == 'PushWithObstacleV1':
        env = PushWithObstacleV1Env(render=True, reward_type='modified_sparse', control_type='ee')
        env = ActionNormalizer(env)
        env = ResetWrapper(env=env)
        env = TimeFeatureWrapper(env=env, max_steps=120, test_mode=False)
        env = TimeLimitWrapper(env=env, max_steps=120)
    elif task_name == 'ReachWithObstacleV0':
        env = ReachWithObstacleV0Env(render=True, reward_type='modified_sparse', control_type='ee')
        env = ActionNormalizer(env)
        env = ResetWrapper(env=env)
        env = TimeFeatureWrapper(env=env, max_steps=100, test_mode=False)
        env = TimeLimitWrapper(env=env, max_steps=100)
    elif task_name == 'ReachWithObstacleV1':
        env = ReachWithObstacleV1Env(render=True, reward_type='modified_sparse', control_type='ee')
        env = ActionNormalizer(env)
        env = ResetWrapper(env=env)
        env = TimeFeatureWrapper(env=env, max_steps=100, test_mode=False)
        env = TimeLimitWrapper(env=env, max_steps=100)

    method = 'active_curr_awac'
    agent = Active_Curr_AWAC(env=env,
                             test_env=env,
                             demo_env=env,
                             task_name=task_name,
                             method=method,
                             base_demo_pool=None,
                             demo_pool=None,
                             max_demo_num=None,
                             writer=None,
                             load_mode=True)
    model_path = PARENT_DIR + '/' + 'models/' + task_name + '/' + method + '/max_demo_10'
    agent.awac_agent.ac.load_state_dict(torch.load(model_path + '/' + step + '_actor_critic.pth'))

    test_episode_reward = 0.0
    total_success_num = 0.0
    test_num = 20

    input("Press Enter to continue...")

    if task_name == 'PushWithObstacleV0' or task_name == 'PushWithObstacleV1':
        goal_pos = env.task.goal_range_center
        obj_ys = np.linspace(env.task.obj_range_low[1], env.task.obj_range_high[1], test_num)
        for i in range(obj_ys.shape[0]):
            object_pos = np.array([-0.15, obj_ys[i], 0.02])
            # state_ = env.reset(whether_random=False, goal_pos=goal_pos, object_pos=object_pos)

            state_ = env.reset()

            done_ = False
            episode_reward = 0.0
            whether_success = 0.0
            while not done_:
                state_ = reconstruct_state(state_)
                action_ = agent.awac_agent.get_action(state_, True)
                next_state_, reward_, done_, info_ = env.step(action_)
                episode_reward += reward_
                test_episode_reward += reward_
                state_ = next_state_

                sleep(0.01)

            if info_['is_success']:
                total_success_num += 1.0
                whether_success = 1.0
                print("[Test {}]: success!".format(i + 1))
            else:
                print("[Test {}]: failure!".format(i + 1))

        average_episode_reward = test_episode_reward / test_num
        average_success_rate = total_success_num / test_num
    elif task_name == 'ReachWithObstacleV0' or task_name == 'ReachWithObstacleV1':
        goal_pos = env.task.goal_range_center
        ee_ys = np.linspace(env.task.init_ee_range_low[1], env.task.init_ee_range_high[1], test_num)
        for i in range(ee_ys.shape[0]):
            ee_pos = np.array([0.15, ee_ys[i], 0.02])
            object_pos = None
            state_ = env.reset(whether_random=False, goal_pos=goal_pos, object_pos=object_pos, ee_pos=ee_pos)

            done_ = False
            episode_reward = 0.0
            whether_success = 0.0
            while not done_:
                state_ = reconstruct_state(state_)
                action_ = agent.awac_agent.get_action(state_, True)
                next_state_, reward_, done_, info_ = env.step(action_)
                episode_reward += reward_
                test_episode_reward += reward_
                state_ = next_state_

                sleep(0.01)

            if info_['is_success']:
                total_success_num += 1.0
                whether_success = 1.0
                print("[Test {}]: success!".format(i + 1))
            else:
                print("[Test {}]: failure!".format(i + 1))

        average_episode_reward = test_episode_reward / test_num
        average_success_rate = total_success_num / test_num

    print("All finished: average success rate: {}, average episode reward: {}".format(average_success_rate,
                                                                                      average_episode_reward))



if __name__ == '__main__':
    main()
