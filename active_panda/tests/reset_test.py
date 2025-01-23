import numpy as np
import gym
import panda_gym
import os
import sys


CURRENT_DIR = os.getcwd()
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR + '/envs/')
from task_envs import ReachWithObstacleV0Env, ReachWithObstacleV1Env, PushWithObstacleV0Env, PushWithObstacleV1Env
sys.path.append(PARENT_DIR + '/utils/')
from utils import ReplayBuffer, OUNoise, Actor, Critic, ActionNormalizer, ResetWrapper, TimeFeatureWrapper, TimeLimitWrapper, reconstruct_state, save_results


def main():
    task_name = 'PushWithObstacleV0'
    env = PushWithObstacleV0Env(render=False, reward_type='modified_sparse', control_type='ee')
    env = ActionNormalizer(env)
    env = ResetWrapper(env=env)
    env = TimeFeatureWrapper(env=env, max_steps=120, test_mode=False)
    env = TimeLimitWrapper(env=env, max_steps=120)

    goal_pos = env.task.goal_range_center
    obj_pos = np.array([-0.15, 0.0, 0.02])
    state = env.reset(whether_random=False, goal_pos=goal_pos, object_pos=obj_pos, ee_pos=None)
    print("Reset ee to default pose")

    desired_ee_pos = np.array([-0.15, 0.0, 0.06])
    state = env.reset(whether_random=False, goal_pos=goal_pos, object_pos=obj_pos, ee_pos=desired_ee_pos)
    state = reconstruct_state(state)
    real_ee_pos = state[0:3]
    print("desired ee pos: {}".format(desired_ee_pos))
    print("real ee pos: {}".format(real_ee_pos))
    print("position distance: {}".format(np.linalg.norm((desired_ee_pos - real_ee_pos))))


if __name__ == '__main__':
    main()