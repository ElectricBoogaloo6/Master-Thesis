from typing import Deque, Dict, List, Tuple, Union
import numpy as np
import random
import copy
import gym
import panda_gym
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn

from collections import deque
import operator
from typing import Callable

import torch
from torch.nn import LayerNorm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from time import sleep

import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib import animation




class SegmentTree:
    """ Create SegmentTree.
    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    Attributes:
        capacity (int)
        tree (list)
        operation (function)
    """

    def __init__(self, capacity: int, operation: Callable, init_value: float):
        """Initialization.
        Args:
            capacity (int)
            operation (function)
            init_value (float)
        """
        assert (
            capacity > 0 and capacity & (capacity - 1) == 0
        ), "capacity must be positive and a power of 2."
        self.capacity = capacity
        self.tree = [init_value for _ in range(2 * capacity)]
        self.operation = operation

    def _operate_helper(
        self, start: int, end: int, node: int, node_start: int, node_end: int
    ) -> float:
        """Returns result of operation in segment."""
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(
                    self._operate_helper(start, mid, 2 * node, node_start, mid),
                    self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
                )

    def operate(self, start: int = 0, end: int = 0) -> float:
        """Returns result of applying `self.operation`."""
        if end <= 0:
            end += self.capacity
        end -= 1

        return self._operate_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val: float):
        """Set value in tree."""
        idx += self.capacity
        self.tree[idx] = val

        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx: int) -> float:
        """Get real value in leaf node of tree."""
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity + idx]


class SumSegmentTree(SegmentTree):
    """ Create SumSegmentTree.
    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    """

    def __init__(self, capacity: int):
        """Initialization.
        Args:
            capacity (int)
        """
        super(SumSegmentTree, self).__init__(
            capacity=capacity, operation=operator.add, init_value=0.0
        )

    def sum(self, start: int = 0, end: int = 0) -> float:
        """Returns arr[start] + ... + arr[end]."""
        return super(SumSegmentTree, self).operate(start, end)

    def retrieve(self, upperbound: float) -> int:
        """Find the highest index `i` about upper bound in the tree"""
        # TODO: Check assert case and fix bug
        assert 0 <= upperbound <= self.sum() + 1e-5, "upperbound: {}".format(upperbound)

        idx = 1

        while idx < self.capacity:  # while non-leaf
            left = 2 * idx
            right = left + 1
            if self.tree[left] > upperbound:
                idx = 2 * idx
            else:
                upperbound -= self.tree[left]
                idx = right
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    """ Create SegmentTree.
    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    """

    def __init__(self, capacity: int):
        """Initialization.
        Args:
            capacity (int)
        """
        super(MinSegmentTree, self).__init__(
            capacity=capacity, operation=min, init_value=float("inf")
        )

    def min(self, start: int = 0, end: int = 0) -> float:
        """Returns min(arr[start], ...,  arr[end])."""
        return super(MinSegmentTree, self).operate(start, end)


class ReplayBuffer_ddpglfd:
    """A numpy replay buffer with demonstrations."""

    def __init__(
            self,
            obs_dim: int,
            act_dim:int,
            size: int,
            batch_size: int = 32,
            gamma: float = 0.99,
            demo: list = None,
            n_step: int = 1,
    ):
        """Initialize."""
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size = 0, 0

        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

        # for demonstration
        self.demo_size = len(demo) if demo else 0
        self.demo = demo

        self.demo_step_ids = []

        if self.demo:
            self.ptr += self.demo_size
            self.size += self.demo_size
            for ptr, d in enumerate(self.demo):
                state, action, reward, next_state, done = d
                self.obs_buf[ptr] = state
                # self.acts_buf[ptr] = np.array(action)
                self.acts_buf[ptr] = action
                self.rews_buf[ptr] = reward
                self.next_obs_buf[ptr] = next_state
                self.done_buf[ptr] = done
                self.demo_step_ids.append(ptr)

    def store(
            self,
            obs: np.ndarray,
            act: np.ndarray,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
            role
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        """Store the transition in buffer."""
        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()

        # make a n-step transition
        rew, next_obs, done = self._get_n_step_info()
        obs, act = self.n_step_buffer[0][:2]

        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        if role == 'teacher':
            self.demo_step_ids.append(self.ptr)

        self.ptr += 1
        self.ptr = 0 if self.ptr % self.max_size == 0 else self.ptr
        while self.ptr in self.demo_step_ids:
            self.ptr += 1
            self.ptr = 0 if self.ptr % self.max_size == 0 else self.ptr

        # self.ptr += 1
        # self.ptr = self.demo_size if self.ptr % self.max_size == 0 else self.ptr
        self.size = min(self.size + 1, self.max_size)

        return self.n_step_buffer[0]

    def sample_batch(self, indices: List[int] = None) -> Dict[str, np.ndarray]:
        """Randomly sample a batch of experiences from memory."""
        assert len(self) >= self.batch_size

        if indices is None:
            indices = np.random.choice(
                len(self), size=self.batch_size, replace=False
            )

        return dict(
            obs=self.obs_buf[indices],
            next_obs=self.next_obs_buf[indices],
            acts=self.acts_buf[indices],
            rews=self.rews_buf[indices],
            done=self.done_buf[indices],
            # for N-step learning
            indices=indices,
        )

    def _get_n_step_info(self) -> Tuple[np.int64, np.ndarray, bool]:
        """Return n step rew, next_obs, and done."""
        # info of the last transition
        rew, next_obs, done = self.n_step_buffer[-1][-3:]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]

            rew = r + self.gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer_ddpglfd):
    """Prioritized Replay buffer with demonstrations."""

    def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            size: int,
            batch_size: int = 32,
            gamma: float = 0.99,
            alpha: float = 0.6,
            epsilon_d: float = 1.0,
            demo: list = None,
    ):
        """Initialize."""
        assert alpha >= 0

        super(PrioritizedReplayBuffer, self).__init__(
            obs_dim, act_dim, size, batch_size, gamma, demo, n_step=1
        )
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        self.epsilon_d = epsilon_d

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

        # for init priority of demo
        self.tree_ptr = self.demo_size
        for i in range(self.demo_size):
            self.sum_tree[i] = self.max_priority ** self.alpha
            self.min_tree[i] = self.max_priority ** self.alpha

    def store(
            self,
            obs: np.ndarray,
            act: np.ndarray,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
            role
    ):
        """Store experience and priority."""
        transition = super().store(obs, act, rew, next_obs, done, role)

        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha

            self.tree_ptr += 1
            if self.tree_ptr % self.max_size == 0:
                self.tree_ptr = self.demo_size

        return transition

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional()

        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        epsilon_d = np.array(
            [self.epsilon_d if i in self.demo_step_ids else 0.0 for i in indices]
        )
        # epsilon_d = np.array(
        #     [self.epsilon_d if i < self.demo_size else 0.0 for i in indices]
        # )

        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            epsilon_d=epsilon_d,
            indices=indices,
        )

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, act_dim: int, size: int, batch_size: int = 32):
        """Initialize."""
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

        self.priority_buf = np.zeros([size], dtype=np.float32)
        self.next_acts_buff = np.zeros([size, act_dim], dtype=np.float32)

    def store(
            self,
            obs: np.ndarray,
            act: np.ndarray,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
    ):
        """Store the transition in buffer."""
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def store_priority(self, demo_id):
        self.priority_buf[self.ptr] = demo_id / 60.0

    def store_next_action(self, next_act):
        self.next_acts_buff[self.ptr] = next_act

    def extend(
            self,
            transitions: List[Tuple],
    ):
        """Store the multi transitions in buffer."""
        for transition in transitions:
            self.store(*transition)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """Randomly sample a batch of experiences from memory."""
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def sample_batch_with_priority(self):
        # print("shape of a: {}".format(self.size))
        # print("shape of p: {}".format((self.priority_buf / np.sum(self.priority_buf)).shape))

        idxs = np.random.choice(self.size, size=self.batch_size, replace=False, p=self.priority_buf[:self.size] / np.sum(self.priority_buf[:self.size]))
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    priority=self.priority_buf[idxs])

    def sample_batch_with_next_action(self):
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    next_acts=self.next_acts_buff[idxs])

    def __len__(self) -> int:
        return self.size


class OUNoise:
    """Ornstein-Uhlenbeck process.
    Taken from Udacity deep-reinforcement-learning github repository:
    https://github.com/udacity/deep-reinforcement-learning/blob/master/
    ddpg-pendulum/ddpg_agent.py
    """

    def __init__(
        self,
        size: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
    ):
        """Initialize parameters and noise process."""
        self.state = np.float64(0.0)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

        # random.seed(seed)

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self) -> np.ndarray:
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
            [random.random() for _ in range(len(x))]
        )
        self.state = x + dx
        return self.state


class Actor(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            init_w: float = 3e-3,
    ):
        """Initialize."""
        super(Actor, self).__init__()

        self.hidden1 = nn.Linear(in_dim, 128)
        self.ln1 = LayerNorm(128)
        self.hidden2 = nn.Linear(128, 128)
        self.ln2 = LayerNorm(128)
        self.out = nn.Linear(128, out_dim)

        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        # x = F.relu(self.hidden1(state))
        x = self.hidden1(state)
        x = self.ln1(x)
        x = F.relu(x)
        # x = F.relu(self.hidden2(x))
        x = self.hidden2(x)
        x = self.ln2(x)
        x = F.relu(x)
        action = self.out(x).tanh()

        return action


class Critic(nn.Module):
    def __init__(
            self,
            in_dim: int,
            init_w: float = 3e-3,
    ):
        """Initialize."""
        super(Critic, self).__init__()

        self.hidden1 = nn.Linear(in_dim, 128)
        self.ln1 = LayerNorm(128)
        self.hidden2 = nn.Linear(128, 128)
        self.ln2 = LayerNorm(128)
        self.out = nn.Linear(128, 1)

        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(
            self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Forward method implementation."""
        x = torch.cat((state, action), dim=-1)
        # x = F.relu(self.hidden1(x))
        x = self.hidden1(x)
        x = self.ln1(x)
        x = F.relu(x)
        # x = F.relu(self.hidden2(x))
        x = self.hidden2(x)
        x = self.ln2(x)
        x = F.relu(x)
        value = self.out(x)

        return value


class ActionNormalizer(gym.ActionWrapper):
    """Rescale and relocate the actions."""

    def action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (-1, 1) to (low, high)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (low, high) to (-1, 1)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, -1.0, 1.0)

        return action


class TimeFeatureWrapper(gym.Wrapper):
    """
    Add remaining, normalized time to observation space for fixed length episodes.
    See https://arxiv.org/abs/1712.00378 and https://github.com/aravindr93/mjrl/issues/13.

    .. note::

        Only ``gym.spaces.Box`` and ``gym.spaces.Dict`` (``gym.GoalEnv``) 1D observation spaces
        are supported for now.

    :param env: Gym env to wrap.
    :param max_steps: Max number of steps of an episode
        if it is not wrapped in a ``TimeLimit`` object.
    :param test_mode: In test mode, the time feature is constant,
        equal to zero. This allow to check that the agent did not overfit this feature,
        learning a deterministic pre-defined sequence of actions.
    """

    def __init__(self, env: gym.Env, max_steps: int = 1000, test_mode: bool = False):
        assert isinstance(
            env.observation_space, (gym.spaces.Box, gym.spaces.Dict)
        ), "`TimeFeatureWrapper` only supports `gym.spaces.Box` and `gym.spaces.Dict` (`gym.GoalEnv`) observation spaces."

        # Add a time feature to the observation
        if isinstance(env.observation_space, gym.spaces.Dict):
            assert "observation" in env.observation_space.spaces, "No `observation` key in the observation space"
            obs_space = env.observation_space.spaces["observation"]
            assert isinstance(
                obs_space, gym.spaces.Box
            ), "`TimeFeatureWrapper` only supports `gym.spaces.Box` observation space."
            obs_space = env.observation_space.spaces["observation"]
        else:
            obs_space = env.observation_space

        assert len(obs_space.shape) == 1, "Only 1D observation spaces are supported"

        low, high = obs_space.low, obs_space.high
        low, high = np.concatenate((low, [0.0])), np.concatenate((high, [1.0]))
        self.dtype = obs_space.dtype

        if isinstance(env.observation_space, gym.spaces.Dict):
            env.observation_space.spaces["observation"] = gym.spaces.Box(low=low, high=high, dtype=self.dtype)
        else:
            env.observation_space = gym.spaces.Box(low=low, high=high, dtype=self.dtype)

        super(TimeFeatureWrapper, self).__init__(env)

        # Try to infer the max number of steps per episode
        try:
            self._max_steps = env.spec.max_episode_steps
            print("default max steps: {}".format(self._max_steps))
        except AttributeError:
            self._max_steps = None
            print("no default max steps")

        # Fallback to provided value
        if self._max_steps is None:
            self._max_steps = max_steps

        self._current_step = 0
        self._test_mode = test_mode

    def reset(self, **kwargs) -> GymObs:
        self._current_step = 0
        return self._get_obs(self.env.reset( **kwargs))

    def step(self, action: Union[int, np.ndarray]) -> GymStepReturn:
        self._current_step += 1
        obs, reward, done, info = self.env.step(action)

        # if self._current_step >= self._max_steps:
        #     done = True

        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs: Union[np.ndarray, Dict[str, np.ndarray]]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Concatenate the time feature to the current observation.

        :param obs:
        :return:
        """
        # Remaining time is more general
        time_feature = 1 - (self._current_step / self._max_steps)
        if self._test_mode:
            time_feature = 1.0
        time_feature = np.array(time_feature, dtype=self.dtype)

        if isinstance(obs, dict):
            obs["observation"] = np.append(obs["observation"], time_feature)
            return obs
        return np.append(obs, time_feature)



class TimeLimitWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    :param max_steps: (int) Max number of steps per episode
    """

    def __init__(self, env, max_steps=50):
        # Call the parent constructor, so we can access self.env later
        super(TimeLimitWrapper, self).__init__(env)
        self.max_steps = max_steps
        # Counter of steps per episode
        self.current_step = 0

    def reset(self, **kwargs):
        """
        Reset the environment
        """
        # print("TimeLimitWrapper: reset() called")
        # Reset the counter
        self.current_step = 0
        obs = self.env.reset(**kwargs)
        if hasattr(self.env, 'object_color'):
            self.object_color = self.env.object_color
        return obs

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, bool, dict) observation, reward, is the episode over?, additional informations
        """
        self.current_step += 1
        # print(f"TimeLimitWrapper: step() called, current_step={self.current_step}")
        obs, reward, done, info = self.env.step(action)
        # Overwrite the truncation signal when when the number of steps reaches the maximum
        if self.current_step >= self.max_steps:
            done = True
        # if hasattr(self.env, 'object_color'):
        #     self.object_color = self.env.object_color
        self.env.object_color = self.object_color
        return obs, reward, done, info


class ResetWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        # Call the parent constructor, so we can access self.env later
        super().__init__(env)
        self.object_color = [1, 1, 1]  # Default color

    def reset(self, **kwargs):
        self.env.reset()

        whether_random = kwargs.get('whether_random', True)
        with self.env.sim.no_rendering():
            if whether_random:
                self.env.robot.reset()
                self.env.task.reset()
            else:
                self.env.robot.reset()
                goal_pos = kwargs.get('goal_pos') # 1d array of the form (x, y, z)
                object_pos = kwargs.get('object_pos') # 1d array of the form (x, y, z)
                self.env.task.goal = goal_pos
                self.env.task.sim.set_base_pose("target", self.env.task.goal, [0, 0, 0, 1])

                if object_pos is None:
                    pass
                else:
                    self.env.task.sim.set_base_pose("object", object_pos, [0, 0, 0, 1])

        # get obs
        robot_obs = self.env.robot.get_obs()  # robot state
        task_obs = self.env.task.get_obs()  # object position, velococity, etc...
        observation = np.concatenate([robot_obs, task_obs])
        achieved_goal = self.env.task.get_achieved_goal()

        obs = {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": self.task.get_goal(),
               }
        self.env.object_color = self.object_color

        return obs

    def step(self, action):
        self.env.object_color = self.object_color
        obs, reward, done, info = self.env.step(action)

        if info['is_success']:
            done = True

        return obs, reward, done, info



class RealRobotWrapper(gym.Wrapper):
    def __init__(self, env):
        # Call the parent constructor, so we can access self.env later
        super().__init__(env)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        return obs

    def step(self, action):
        # resize the action for the controller of real robot
        action = action * 0.2

        obs, reward, done, info = self.env.step(action)

        return obs, reward, done, info


def reconstruct_state(state):
    obs = state['observation'] # 1d np array and we exclude the last time feature
    goal = state['desired_goal'] # 1d np array in the form of (x, y, z)
    state = np.concatenate((obs, goal))

    return state


def save_results(method, task_name, evaluation_res, success_evaluation_res, max_demo_num=60, mode=None):
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)

    res_per_step_path = parent_dir + '/' + 'evaluation_res/' + task_name + '/' + method + '/max_demo_' + str(
        max_demo_num) + '/'

    if mode is not None:
        res_per_step_path += mode + '/'

    if not os.path.exists(res_per_step_path):
        os.makedirs(res_per_step_path)

    np.savetxt(res_per_step_path + 'res_per_step.csv', evaluation_res, delimiter=' ')
    np.savetxt(res_per_step_path + 'success_res_per_step.csv', success_evaluation_res, delimiter=' ')



def draw_q_value_heatmap(critic_model, actor_model, device=None, save=False, name=None, vmin=None, vmax=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    points_per_row = 30 + 1

    # q_value_map = np.zeros((points_per_row, points_per_row))
    q_values = []
    for object_x in np.linspace(0.15, -0.15, points_per_row):
        q_values_row = []
        for object_y in np.linspace(0.15, -0.15, points_per_row):
            state = np.array([])

            object_x = np.round(object_x, 2)
            object_y = np.round(object_y, 2)

            ee_pos = np.array([object_x, object_y, 0.04])
            state = np.concatenate((state, ee_pos))

            ee_vel = np.array([0.0, 0.0, 0.0])
            state = np.concatenate((state, ee_vel))

            object_pos = np.array([object_x, object_y, 0.02])
            state = np.concatenate((state, object_pos))

            object_rot = np.array([0.0, 0.0, 0.0])
            state = np.concatenate((state, object_rot))

            object_vel = np.array([0.0, 0.0, 0.0])
            state = np.concatenate((state, object_vel))

            object_angular_vel = np.array([0.0, 0.0, 0.0])
            state = np.concatenate((state, object_angular_vel))

            time_feature = np.array([1.0])
            state = np.concatenate((state, time_feature))

            goal_pos = np.array([0.1, 0.0, 0.02])
            state = np.concatenate((state, goal_pos))

            state_tensor = torch.FloatTensor(state).to(device)
            action_tensor = actor_model(state_tensor).detach()
            q_value = critic_model(state_tensor, action_tensor).detach().cpu().numpy()

            q_values_row.append(q_value[0])

            # map_x_ind = int((0.15 - object_x) / 0.01)
            # map_y_ind = int((0.15 - object_y) / 0.01)
            # q_value_map[map_x_ind][map_y_ind] = q_value[0]

        q_values.append(q_values_row)

    x_ticks = np.linspace(0.15, -0.15, points_per_row)
    x_ticks = np.round(x_ticks, 4)
    y_ticks = np.linspace(0.15, -0.15, points_per_row)
    y_ticks = np.round(y_ticks, 4)

    q_values = np.array(q_values)

    # heatmap = sn.heatmap(data=q_value_map, xticklabels=x_ticks, yticklabels=y_ticks)

    if vmin is None:
        heatmap = sn.heatmap(data=q_values, xticklabels=x_ticks, yticklabels=y_ticks, square=True)
    else:
        heatmap = sn.heatmap(data=q_values, xticklabels=x_ticks, yticklabels=y_ticks, vmin=vmin, vmax=vmax, square=True)

    if save:
        plt.savefig(name)

    # plt.show()

    return heatmap


def draw_heatmap_animation(actor_model, critic_model, device, model_path, plot_name, mode='per_step', vmin=0.0, vmax=200.0):
    fig = plt.figure()
    def init():
        plt.clf()

        if mode == 'per_step':
            step = int(5 * 1e3)

            actor_model.load_state_dict(torch.load(model_path + '/' + str(step) + '_actor.pth'))
            critic_model.load_state_dict(torch.load(model_path + '/' + str(step) + '_critic.pth'))

            draw_q_value_heatmap(critic_model, actor_model, device, vmin=vmin, vmax=vmax)

    def animate(i):
        plt.clf()

        if mode == 'per_step':
            step = int(5*(i + 1) * 1e3)

            actor_model.load_state_dict(torch.load(model_path + '/' + str(step) + '_actor.pth'))
            critic_model.load_state_dict(torch.load(model_path + '/' + str(step) + '_critic.pth'))

            draw_q_value_heatmap(critic_model, actor_model, device, vmin=vmin, vmax=vmax)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=99, repeat=False, interval=100)

    # savefile = r"heatmap.gif"
    savefile = plot_name + '.gif'
    pillowwriter = animation.PillowWriter(fps=5)
    anim.save(savefile, writer=pillowwriter)


def draw_rollout_trajectories(actor_model, env, device=None, save=False, name=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    goal_position = np.array([0.1, 0.0, 0.02])
    object_ys = np.linspace(-0.15, 0.15, 31)
    for object_y in object_ys:
        object_trajectory = []
        object_position = np.array([-0.15, object_y, 0.02])
        object_trajectory.append(object_position)
        state = env.reset(whether_random=False, goal_pos=goal_position, object_pos=object_position)
        done = False

        while not done:
            reshaped_state = reconstruct_state(state)
            action = actor_model(torch.FloatTensor(reshaped_state).to(device)).detach().cpu().numpy()
            next_state, reward, done, info = env.step(action)

            state = next_state

            if env.robot.block_gripper:
                object_position = state['observation'][6:9].copy()
            else:
                object_position = state['observation'][7:10].copy()

            object_trajectory.append(object_position)

        object_trajectory = np.array(object_trajectory)
        if info['is_success']:
            color = 'green'
        elif len(object_trajectory) >= 50:
            color = 'orange'
        else:
            color = 'red'
        x_new = -object_trajectory[:, 1]
        y_new = object_trajectory[:, 0]

        plt.axis('scaled')
        plt.hlines(y=-0.11, xmin=-0.05, xmax=0.05, linewidth=2, color='black')
        plt.vlines(x=-0.185, ymin=-0.15, ymax=0.15, linewidth=2, color='black')
        plt.vlines(x=0.185, ymin=-0.15, ymax=0.15, linewidth=2, color='black')
        # plt.plot(0.0, 0.1, c='grey', marker='o')

        plt.plot(x_new, y_new, color=color)
        plt.xlim(-0.2, 0.2)
        plt.ylim(-0.2, 0.2)

    if save:
        plt.savefig(name)


def draw_rollout_animation(actor_model, env, device, model_path, plot_name):
    fig = plt.figure()

    def init():
        plt.clf()

        step = int(5 * 1e3)

        actor_model.load_state_dict(torch.load(model_path + '/' + str(step) + '_actor.pth'))

        draw_rollout_trajectories(actor_model, env, device)

    def animate(i):
        plt.clf()

        step = int(5*(i + 1) * 1e3)

        actor_model.load_state_dict(torch.load(model_path + '/' + str(step) + '_actor.pth'))

        draw_rollout_trajectories(actor_model, env, device)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=99, repeat=False, interval=100)

    # savefile = r"heatmap.gif"
    savefile = plot_name + '.gif'
    pillowwriter = animation.PillowWriter(fps=5)
    anim.save(savefile, writer=pillowwriter)

    # FFwriter = animation.FFMpegWriter(fps=10)
    # anim.save(plot_name + '.mp4', writer=FFwriter)


def draw_demo_trajectories(demo_data_path, env, oracle_model, device, save=False, name=None):
    initial_object_positions = np.genfromtxt(demo_data_path + 'demo_object_positions.csv', delimiter=' ')

    # xs = initial_object_positions[:, 0]
    object_ys = initial_object_positions[:, 1]
    goal_position = np.array([0.1, 0.0, 0.02])
    demo_id = 0
    alphas = np.linspace(0.1, 1, initial_object_positions.shape[0])
    for object_y in object_ys:
        demo_id += 1

        object_trajectory = []
        object_position = np.array([-0.15, object_y, 0.02])
        object_trajectory.append(object_position)
        state = env.reset(whether_random=False, goal_pos=goal_position, object_pos=object_position)
        done = False

        while not done:
            reshaped_state = reconstruct_state(state)
            action, _ = oracle_model.predict(state, deterministic=True)
            # action = actor_model(torch.FloatTensor(reshaped_state).to(device)).detach().cpu().numpy()
            next_state, reward, done, info = env.step(action)

            state = next_state

            if env.robot.block_gripper:
                object_position = state['observation'][6:9].copy()
            else:
                object_position = state['observation'][7:10].copy()

            object_trajectory.append(object_position)

        object_trajectory = np.array(object_trajectory)
        if info['is_success']:
            color = 'green'
        elif len(object_trajectory) >= 50:
            color = 'orange'
        else:
            color = 'red'
        x_new = -object_trajectory[:, 1]
        y_new = object_trajectory[:, 0]

        plt.axis('scaled')
        plt.hlines(y=-0.11, xmin=-0.05, xmax=0.05, linewidth=2, color='black')
        plt.vlines(x=-0.185, ymin=-0.15, ymax=0.15, linewidth=2, color='black')
        plt.vlines(x=0.185, ymin=-0.15, ymax=0.15, linewidth=2, color='black')
        # plt.plot(0.0, 0.1, c='grey', marker='o')

        plt.plot(x_new, y_new, color=color, linewidth=0.5, alpha=alphas[demo_id - 1])
        plt.xlim(-0.2, 0.2)
        plt.ylim(-0.2, 0.2)

        print("plot demo {}".format(demo_id))

    if save:
        plt.savefig(name)

