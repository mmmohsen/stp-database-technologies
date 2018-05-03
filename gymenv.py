import numpy as np

import gym
from gym import spaces
from gym.utils import seeding


class DatabaseIndexesEnv(gym.Env):
    def __init__(self, n):
        self.action_space = Dynamic(n)
        self.observation_space = spaces.Tuple([spaces.Discrete(2) for _ in range(n)])

    def reset(self):
        for index in self.observation_space.spaces:
            index.n = 0

    def render(self, mode='human'):
        pass

    def step(self, action):
        self.observation_space.spaces[action].n = 1
        self.action_space.disable_actions((action,))
        return self.observation_space, reward(self.observation_space),


class Dynamic(gym.Space):
    """
    x where x in available actions {0,1,3,5,...,n-1}
    Example usage:
    self.action_space = spaces.Dynamic(max_space=2)
    """

    def __init__(self, max_space):
        super().__init__()
        self.n = max_space

        # initially all actions are available
        self.available_actions = range(0, max_space)

    def disable_actions(self, actions):
        """ You would call this method inside your environment to remove available actions"""
        self.available_actions = [action for action in self.available_actions if action not in actions]
        return self.available_actions

    def enable_actions(self, actions):
        """ You would call this method inside your environment to enable actions"""
        self.available_actions = self.available_actions.append(actions)
        return self.available_actions

    def sample(self):
        return np.random.choice(self.available_actions)

    def contains(self, x):
        return x in self.available_actions

    @property
    def shape(self):
        return ()

    def __repr__(self):
        return "Dynamic(%d)" % self.n

    def __eq__(self, other):
        return self.n == other.n
