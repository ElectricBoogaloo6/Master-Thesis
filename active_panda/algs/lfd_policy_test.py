import numpy as np
import random
import gym
import panda_gym
import torch
from time import sleep

import os
import argparse

CURRENT_DIR = os.getcwd()
PARENT_DIR = os.path.dirname(CURRENT_DIR)
from utils import  ActionNormalizer, ResetWrapper, TimeFeatureWrapper, TimeLimitWrapper, reconstruct_state, Actor


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', help='name of task')

    return parser.parse_args()



if __name__ == '__main__':
    args = argparser()

    seed = 6
    np.random.seed(seed)
    random.seed(seed)
    whether_render = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task_name = args.task_name
    if task_name == 'Reach':
        env = gym.make("Panda" + task_name + "-v2", render=whether_render)
        env = ActionNormalizer(env)
        env = ResetWrapper(env=env)
        env = TimeFeatureWrapper(env=env, max_steps=50, test_mode=False)
        env = TimeLimitWrapper(env=env, max_steps=50)

        obs_dim = env.observation_space['observation'].shape[0] + 3  # exclude the time feature and add goal position
        action_dim = env.action_space.shape[0]
        actor_model = Actor(obs_dim, action_dim).to(device)
        model_path = PARENT_DIR + '/' + 'models/source_tasks/' + task_name + '/'
        actor_model.load_state_dict(torch.load(model_path + '/' + 'source_actor.pth'))
    elif task_name == 'PickAndPlace':
        env = gym.make("Panda" + task_name + "-v2", render=whether_render)
        env = ActionNormalizer(env)
        env = ResetWrapper(env=env)
        env = TimeFeatureWrapper(env=env, max_steps=50, test_mode=False)
        env = TimeLimitWrapper(env=env, max_steps=50)

        obs_dim = env.observation_space['observation'].shape[0] + 3  # exclude the time feature and add goal position
        action_dim = env.action_space.shape[0]
        actor_model = Actor(obs_dim, action_dim).to(device)
        model_path = PARENT_DIR + '/' + 'models/source_tasks/' + task_name + '/'
        actor_model.load_state_dict(torch.load(model_path + '/' + 'source_actor.pth'))
    elif task_name == 'Push':
        env = gym.make("Panda" + task_name + "-v2", render=whether_render)
        env = ActionNormalizer(env)
        env = ResetWrapper(env=env)
        env = TimeFeatureWrapper(env=env, max_steps=50, test_mode=False)
        env = TimeLimitWrapper(env=env, max_steps=50)

        obs_dim = env.observation_space['observation'].shape[0] + 3  # exclude the time feature and add goal position
        action_dim = env.action_space.shape[0]
        actor_model = Actor(obs_dim, action_dim).to(device)
        model_path = PARENT_DIR + '/' + 'models/source_tasks/' + task_name + '/'
        actor_model.load_state_dict(torch.load(model_path + '/' + 'source_actor.pth'))


    test_num = 50
    success_num = 0.0
    for i in range(test_num):
        state = env.reset()
        done = False

        while not done:
            reshaped_state = reconstruct_state(state)
            action = actor_model(torch.FloatTensor(reshaped_state).to(device)).detach().cpu().numpy()
            next_state, reward, done, info = env.step(action)

            if whether_render:
                env.render()
                sleep(0.01)

            state = next_state

        if info['is_success']:
            print("test [{}]: success!".format(i + 1))
            success_num += 1.0
        else:
            print('test [{}]: failed!'.format(i + 1))

    print("All finished!")
    print("success ratio: {}".format(success_num / test_num))