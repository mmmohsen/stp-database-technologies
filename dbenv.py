import numpy as np

import gym
from gym import spaces

from db import drop_indexes, get_execution_time, add_index


class DatabaseIndexesEnv(gym.Env):
    """
    Base env providing the actor the means to abstract actions/state. Abstracts all real manipulations with db
    (e.g. taking actions, getting cost).
    """

    def __init__(self, n, table_name, query_batch, connector, k=3):
        """
        Constructs new env.
        :param n: number of columns
        :param table_name: name of the table
        :param query_batch: collection of queries
        :param connector: database connection
        :param k: max number of indices
        """
        super(DatabaseIndexesEnv, self).__init__()
        self.action_space = Dynamic(n)
        self.observation_space = spaces.Tuple([spaces.Discrete(2) for _ in range(n)])
        self.state = list(False for _ in range(n))
        self.table_name = table_name
        self.query_batch = query_batch
        self.connector = connector
        self.k = k
        self.episode = 0
        drop_indexes(connector, table_name)
        self.old_cost = self._get_execution_time_for_batch()

    def reset(self):
        self.episode = 0
        drop_indexes(self.connector, self.table_name)
        self.old_cost = self._get_execution_time_for_batch()
        self.state = list(False for _ in range(len(self.state)))
        self.action_space = Dynamic(len(self.state))
        return self.state

    def render(self, mode='human'):
        # no fancy stuff for now
        super(DatabaseIndexesEnv, self).render(mode)

    def step(self, action):
        self.episode += 1
        self.state[action] = True
        self.action_space.disable_actions((action,))
        add_index(self.connector, action, self.table_name)
        new_cost = self._get_execution_time_for_batch()
        reward = self.old_cost - new_cost
        self.old_cost = new_cost
        return self.state, reward, self.episode >= self.k, {}

    def _get_execution_time_for_batch(self):
        return sum(
            (get_execution_time(self.connector, query.build_query(self.table_name)) for query in self.query_batch))


class Dynamic(gym.Space):
    """
    x where x in available actions {0,1,3,5,...,n-1}
    Example usage:
    self.action_space = spaces.Dynamic(max_space=2)
    """

    def __init__(self, max_space):
        self.n = max_space
        # initially all actions are available
        self.available_actions = range(0, max_space)
        super(Dynamic, self).__init__((), np.int64)

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

    def __repr__(self):
        return "Dynamic(%d)" % self.n

    def __eq__(self, other):
        return self.n == other.n
