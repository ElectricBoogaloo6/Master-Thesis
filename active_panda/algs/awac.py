from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import panda_gym
import time
import torch.nn.functional as F
import os

import sys
import argparse
from datetime import datetime
import random
from torch.utils.tensorboard import SummaryWriter


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
ACTIVE_PANDA_DIR = os.path.dirname(CURRENT_DIR)
ENVS_DIR = os.path.join(ACTIVE_PANDA_DIR, 'envs')
RESULTS_DIR = os.path.join(ACTIVE_PANDA_DIR, "Phase_1_results")
sys.path.append(ENVS_DIR)
from task_envs import ReachWithObstacleV0Env, ReachWithObstacleV1Env, PushWithObstacleV0Env, PushWithObstacleV1Env
sys.path.append(PARENT_DIR + '/utils/')
from utils_awac import ActionNormalizer, ResetWrapper, TimeFeatureWrapper, TimeLimitWrapper, reconstruct_state, save_results
import core as core


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size, device):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

        self.device = device

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32, idxs=None):
        if idxs is None:
            idxs = np.random.randint(0, self.size, size=batch_size)

        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k, v in batch.items()}

        # return {k: torch.FloatTensor(v).to(self.device) for k, v in batch.items()}

    def sample_batch_array(self, batch_size=32, idxs=None):
        if idxs is None:
            idxs = np.random.randint(0, self.size, size=batch_size)

        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])

        return batch


class AWAC:

    def __init__(self,
                 env,
                 test_env,
                 writer,
                 actor_critic=core.MLPActorCritic,
                 with_layer_norm=False,
                 steps_per_epoch=1000,
                 epochs=1000,
                 replay_size=int(2000000),
                 gamma=0.99,
                 polyak=0.995,
                 lr=3e-4,
                 p_lr=3e-4,
                 alpha=0.0,
                 batch_size=1024,
                 start_steps=10000,
                 update_after=0,
                 update_every=50,
                 num_test_episodes=10,
                 max_ep_len=1000,
                 logger_kwargs=dict(),
                 save_freq=1,
                 algo='SAC'):
        """
        Soft Actor-Critic (SAC)


        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.

            actor_critic: The constructor method for a PyTorch Module with an ``act``
                method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
                The ``act`` method and ``pi`` module should accept batches of
                observations as inputs, and ``q1`` and ``q2`` should accept a batch
                of observations and a batch of actions as inputs. When called,
                ``act``, ``q1``, and ``q2`` should return:

                ===========  ================  ======================================
                Call         Output Shape      Description
                ===========  ================  ======================================
                ``act``      (batch, act_dim)  | Numpy array of actions for each
                                            | observation.
                ``q1``       (batch,)          | Tensor containing one current estimate
                                            | of Q* for the provided observations
                                            | and actions. (Critical: make sure to
                                            | flatten this!)
                ``q2``       (batch,)          | Tensor containing the other current
                                            | estimate of Q* for the provided observations
                                            | and actions. (Critical: make sure to
                                            | flatten this!)
                ===========  ================  ======================================

                Calling ``pi`` should return:

                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                            | given observations.
                ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                            | actions in ``a``. Importantly: gradients
                                            | should be able to flow back into ``a``.
                ===========  ================  ======================================

            ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
                you provided to SAC.

            seed (int): Seed for random number generators.

            steps_per_epoch (int): Number of steps of interaction (state-action pairs)
                for the agent and the environment in each epoch.

            epochs (int): Number of epochs to run and train agent.

            replay_size (int): Maximum length of replay buffer.

            gamma (float): Discount factor. (Always between 0 and 1.)

            polyak (float): Interpolation factor in polyak averaging for target
                networks. Target networks are updated towards main networks
                according to:

                .. math:: \\theta_{\\text{targ}} \\leftarrow
                    \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

                where :math:`\\rho` is polyak. (Always between 0 and 1, usually
                close to 1.)

            lr (float): Learning rate (used for both policy and value learning).

            alpha (float): Entropy regularization coefficient. (Equivalent to
                inverse of reward scale in the original SAC paper.)

            batch_size (int): Minibatch size for SGD.

            start_steps (int): Number of steps for uniform-random action selection,
                before running real policy. Helps exploration.

            update_after (int): Number of env interactions to collect before
                starting to do gradient descent updates. Ensures replay buffer
                is full enough for useful updates.

            update_every (int): Number of env interactions that should elapse
                between gradient descent updates. Note: Regardless of how long
                you wait between updates, the ratio of env steps to gradient steps
                is locked to 1.

            num_test_episodes (int): Number of episodes to test the deterministic
                policy at the end of each epoch.

            max_ep_len (int): Maximum length of trajectory / episode / rollout.

            logger_kwargs (dict): Keyword args for EpochLogger.

            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.

            """

        self.env = env
        self.test_env = test_env

        self.obs_dim = env.observation_space['observation'].shape[0] + 3  # add goal position
        self.act_dim = env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # Create actor-critic module and target networks
        self.ac = actor_critic(self.obs_dim, self.env.action_space, device=self.device,
                               special_policy='awac', with_layer_norm=with_layer_norm)
        self.ac_targ = actor_critic(self.obs_dim, self.env.action_space, device=self.device,
                                    special_policy='awac', with_layer_norm=with_layer_norm)
        self.ac_targ.load_state_dict(self.ac.state_dict())
        self.gamma = gamma

        # tensorboard logger
        self.writer = writer

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                          size=replay_size, device=self.device)
        self.demo_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                          size=replay_size, device=self.device)

        self.algo = algo

        self.p_lr = p_lr
        self.lr = lr
        self.alpha = 0
        # # Algorithm specific hyperparams

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.p_lr, weight_decay=1e-4)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.update_after = update_after
        self.update_every = update_every
        self.batch_size = batch_size
        self.save_freq = save_freq
        self.polyak = polyak

        # Set up model saving
        print("Running Offline RL algorithm: {}".format(self.algo))

    def populate_replay_buffer(self, demos_list):
        for demo in demos_list:
            demo_length = demo['state_trajectory'].shape[0]
            for step in range(demo_length):
                state = demo['state_trajectory'][step].copy()
                action = demo['action_trajectory'][step].copy()
                next_state = demo['next_state_trajectory'][step].copy()
                reward = demo['state_trajectory'][step][0]
                done = demo['done_trajectory'][step][0]

                transition = [state, action, reward, next_state, done]
                self.demo_buffer.store(*transition)

        print("[Populate offline base demos]: Finished")

    def sample_mixed_batch(self):
        demo_batch_size = int(self.batch_size / 2)
        rollout_batch_size = int(self.batch_size / 2)

        demo_batch = self.demo_buffer.sample_batch_array(demo_batch_size)
        rollout_batch = self.replay_buffer.sample_batch_array(rollout_batch_size)

        demo_obs = demo_batch['obs']
        demo_obs2 = demo_batch['obs2']
        demo_act = demo_batch['act']
        demo_rew = demo_batch['rew']
        demo_done = demo_batch['done']

        rollout_obs = rollout_batch['obs']
        rollout_obs2 = rollout_batch['obs2']
        rollout_act = rollout_batch['act']
        rollout_rew = rollout_batch['rew']
        rollout_done = rollout_batch['done']

        mixed_obs = np.concatenate((demo_obs, rollout_obs))
        mixed_obs2 = np.concatenate((demo_obs2, rollout_obs2))
        mixed_act = np.concatenate((demo_act, rollout_act))
        mixed_rew = np.concatenate((demo_rew, rollout_rew))
        mixed_done = np.concatenate((demo_done, rollout_done))

        res_dict = {'obs': mixed_obs, 'obs2':mixed_obs2, 'act':mixed_act, 'rew':mixed_rew, 'done':mixed_done}

        return {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k, v in res_dict.items()}


    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        self.writer.add_scalar(tag='loss/q1_loss', scalar_value=loss_q1,
                               global_step=self.total_step)
        self.writer.add_scalar(tag='loss/q2_loss', scalar_value=loss_q2,
                               global_step=self.total_step)
        self.writer.add_scalar(tag='loss/total_q_loss', scalar_value=loss_q,
                               global_step=self.total_step)

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        o = data['obs']

        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        v_pi = torch.min(q1_pi, q2_pi)

        # beta = 2
        beta = 0.3
        q1_old_actions = self.ac.q1(o, data['act'])
        q2_old_actions = self.ac.q2(o, data['act'])
        q_old_actions = torch.min(q1_old_actions, q2_old_actions)

        adv_pi = q_old_actions - v_pi
        weights = F.softmax(adv_pi / beta, dim=0)
        policy_logpp = self.ac.pi.get_logprob(o, data['act'])
        loss_pi = (-policy_logpp * len(weights) * weights.detach()).mean()

        self.writer.add_scalar(tag='loss/pi_loss', scalar_value=loss_pi,
                               global_step=self.total_step)

        # Useful info for logging
        pi_info = dict(LogPi=policy_logpp.cpu().detach().numpy())

        return loss_pi, pi_info

    def update(self, data, update_timestep):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Record things
        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, deterministic=False):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(self.device), deterministic)

        # return self.ac.act(torch.FloatTensor(o).to(self.device), deterministic)

    def test_agent(self, task_name, evaluation_res, success_evaluation_res):
        test_episode_reward = 0.0
        total_success_num = 0.0

        res = []
        success_res = []
        res.append(self.total_step)
        success_res.append(self.total_step)

        if task_name == 'PushWithObstacleV0' or task_name == 'PushWithObstacleV1':
            goal_pos = self.test_env.task.goal_range_center
            obj_ys = np.linspace(self.test_env.task.obj_range_low[1], self.test_env.task.obj_range_high[1],
                                 self.num_test_episodes)
            for i in range(obj_ys.shape[0]):
                object_pos = np.array([-0.15, obj_ys[i], 0.02])
                state_ = self.test_env.reset(whether_random=False, goal_pos=goal_pos, object_pos=object_pos)

                done_ = False
                episode_reward = 0.0
                whether_success = 0.0
                while not done_:
                    state_ = reconstruct_state(state_)
                    action_ = self.get_action(state_, True)
                    next_state_, reward_, done_, info_ = self.test_env.step(action_)
                    episode_reward += reward_
                    test_episode_reward += reward_
                    state_ = next_state_

                if info_['is_success']:
                    total_success_num += 1.0
                    whether_success = 1.0

                res.append(episode_reward)
                success_res.append(whether_success)

        elif task_name == 'ReachWithObstacleV0' or task_name == 'ReachWithObstacleV1':
            goal_pos = self.test_env.task.goal_range_center
            init_ee_ys = np.linspace(self.test_env.task.init_ee_range_low[1], self.test_env.task.init_ee_range_high[1],
                                     self.num_test_episodes)
            for i in range(init_ee_ys.shape[0]):
                ee_pos = np.array([0.15, init_ee_ys[i], 0.02])
                state_ = self.test_env.reset(whether_random=False, goal_pos=goal_pos, object_pos=None, ee_pos=ee_pos)

                done_ = False
                episode_reward = 0.0
                whether_success = 0.0
                while not done_:
                    state_ = reconstruct_state(state_)
                    action_ = self.get_action(state_, True)
                    next_state_, reward_, done_, info_ = self.test_env.step(action_)
                    episode_reward += reward_
                    test_episode_reward += reward_
                    state_ = next_state_

                if info_['is_success']:
                    total_success_num += 1.0
                    whether_success = 1.0

                res.append(episode_reward)
                success_res.append(whether_success)

        evaluation_res.append(res)
        average_episode_reward = test_episode_reward / self.num_test_episodes

        success_evaluation_res.append(success_res)
        average_success_rate = total_success_num / self.num_test_episodes

        self.writer.add_scalar(tag='episode_reward/train/per_env_step',
                               scalar_value=average_episode_reward,
                               global_step=self.total_step)
        self.writer.add_scalar(tag='success_rate/train/per_env_step',
                               scalar_value=average_success_rate,
                               global_step=self.total_step)

        print("[{} environment steps finished]: Average episode reward is {}".format(
            self.total_step,
            average_episode_reward))

        return average_episode_reward, evaluation_res, average_success_rate, success_evaluation_res

    def env_reset(self, task_name):
        if task_name == 'PushWithObstacleV0' or task_name == 'PushWithObstacleV1':
            state = self.env.reset(whether_random=True)
        elif task_name == 'ReachWithObstacleV0' or task_name == 'ReachWithObstacleV1':
            ee_pos = np.random.uniform(self.env.task.init_ee_range_low, self.env.task.init_ee_range_high)
            goal_pos = self.env.task.goal_range_center
            state = self.env.reset(whether_random=False, goal_pos=goal_pos, object_pos=None, ee_pos=ee_pos)

        return state

    def run(self, task_name, method, max_demo_num=10, model_saving_interval=5000):
        evaluate_res_per_step = []
        success_evaluate_res_per_step = []

        model_path = PARENT_DIR + '/' + 'models/' + task_name + '/' + method + '/max_demo_' + str(
            max_demo_num) + '/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # Prepare for interaction with environment
        total_steps = self.epochs * self.steps_per_epoch
        start_time = time.time()

        state = self.env_reset(task_name)
        done = True
        num_train_episodes = 0

        # Main loop: collect experience in env and update/log each epoch
        for t in range(total_steps):
            self.env.render("rgb_array")
            self.total_step = t + 1

            # Reset stuff if necessary
            if done and t > 0:
                state = self.env_reset(task_name=task_name)
                num_train_episodes += 1

            # Collect experience
            reshaped_state = reconstruct_state(state)
            act = self.get_action(reshaped_state , deterministic=False)
            next_state, rew, done, info = self.env.step(act)
            reshaped_next_state = reconstruct_state(next_state)

            self.replay_buffer.store(reshaped_state.copy(), act, rew, reshaped_next_state.copy(), done)
            state = next_state

            # Update handling
            if t > self.update_after and t % self.update_every == 0:
                for _ in range(self.update_every):
                    # batch = self.replay_buffer.sample_batch(self.batch_size)
                    batch = self.sample_mixed_batch()
                    self.update(data=batch, update_timestep=t)

            # End of epoch handling
            if (t + 1) % self.steps_per_epoch == 0:
                epoch = (t + 1) // self.steps_per_epoch

                # Save model
                # if (epoch % self.save_freq == 0) or (epoch == self.epochs):
                #     self.logger.save_state({'env': self.env}, None)


                # Test the performance of the deterministic version of the agent.
                _, evaluate_res_per_step, _, success_evaluate_res_per_step = self.test_agent(task_name=task_name, evaluation_res=evaluate_res_per_step, success_evaluation_res=success_evaluate_res_per_step)

            if (t + 1) % model_saving_interval == 0:
                torch.save(self.ac.state_dict(),
                           model_path + str(self.total_step) + '_' + 'actor_critic' + '.pth')

        self.env.close()
        self.test_env.close()

        return evaluate_res_per_step, success_evaluate_res_per_step

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', help='name of task')
    # parser.add_argument('--mode', help='mode of learning, scratch or transfer')

    return parser.parse_args()


def prepare_demo_pool(demo_path):
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


def main():
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    task_name = 'PushWithObstacleV0'

    seed = 6
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # environment

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

        demo_env = PushWithObstacleV0Env(render=False, reward_type='modified_sparse', control_type='ee',
                                            distance_thres=0.045)
        demo_env = ActionNormalizer(demo_env)
        demo_env = ResetWrapper(env=demo_env)
        demo_env = TimeFeatureWrapper(env=demo_env, max_steps=120, test_mode=False)
        demo_env = TimeLimitWrapper(env=demo_env, max_steps=120)

        max_demo_num = 10
    

    # training parameters
    if task_name == 'PushWithObstacleV0':
        max_env_steps = int(1000 * 1e3)
  
    # training hyperparameters
    method = 'awac_no_layernorm'
    memory_size = 100000
    batch_size = 1024

    # load and prepare the pool for online demonstrations
    demo_pool_path = PARENT_DIR + '/' + 'demo_data/demo_pool/' + 'whole_space' + '/' + task_name + '/'
    demo_pool = prepare_demo_pool(demo_path=demo_pool_path)

    # load and prepare the pool for offline base demonstrations
    base_demo_pool_path = PARENT_DIR + '/' + 'demo_data/demo_pool/' + 'init_space' + '/' + task_name + '/'
    base_demo_pool = prepare_demo_pool(demo_path=base_demo_pool_path)

    # sample 10 base demos
    sampled_demos = []
    sample_num = max_demo_num
    demo_ids = np.random.choice(len(base_demo_pool),sample_num, replace=False)
    for demo_id in demo_ids:
        sampled_base_demo = base_demo_pool[demo_id].copy()
        sampled_demos.append(sampled_base_demo)


    # logger
    writer = SummaryWriter(
        PARENT_DIR + '/' + 'logs/' + task_name + '/' + method + '/max_demo_' + str(max_demo_num) + '/' + TIMESTAMP)

    awac_agent = AWAC(env=env,
                      test_env=test_env,
                      writer=writer,
                      batch_size=1024,
                      epochs=int(max_env_steps / 1000.0))

    awac_agent.populate_replay_buffer(demos_list=sampled_demos)
    reward_res_per_step, success_res_per_step = awac_agent.run(task_name=task_name, method=method, max_demo_num=max_demo_num, model_saving_interval=5000)

    save_results(method, task_name, reward_res_per_step, success_res_per_step, max_demo_num, mode=None)
    writer.close()
    print("[AWAC]: Results are saved! All finished")
    print("************************")


if __name__ == '__main__':
    main()

