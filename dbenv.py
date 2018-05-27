import numpy as np

import gym
from gym import spaces
from numpy import median
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
        self.step_number = 0
        self.cache = {}
        self.old_cost = self._get_execution_time_for_batch()

    def reset(self):
        self.step_number = 0
        drop_indexes(self.connector, self.table_name)
        self.state = list(False for _ in range(len(self.state)))
        self.action_space = Dynamic(len(self.state))
        return self.state

    def render(self, mode='human'):
        # no fancy stuff for now
        super(DatabaseIndexesEnv, self).render(mode)

    def step(self, action):
        self.step_number += 1
        self.state[action] = True
        self.action_space.disable_actions((action,))
        cached = None
        try:
            cached = self.cache[self._key_for_state_query()]
        except KeyError:
            pass
        cost = cached
        if not cached:
            add_index(self.connector, action, self.table_name)
            costs = list()
            for y in xrange(3):
                costs.append(self._get_execution_time_for_batch_str())
            cost = median(costs)
            self.cache[self._key_for_state_query()] = cost
        reward = 1 / float(cost) * 10000
        return self.state, reward, self.step_number >= self.k, {}

    def set_query_batch(self, query_batch):
        self.query_batch = query_batch

    def clear_cache(self):
        self.cache = {}

    def _get_execution_time_for_batch(self):
        return sum(
            (get_execution_time(self.connector, query.build_query(self.table_name)) for query in self.query_batch))

    def _get_execution_time_for_batch_str(self):
        return sum(
            (get_execution_time(self.connector, query) for query in self.query_batch))

    def _key_for_state_query(self):
        return state_to_int(self.state), str(list([str(x) for x in self.query_batch]))

    def calc_old_cost(self, old_cost=None):
        if old_cost != None:
            self.old_cost = old_cost
        else:
            self.old_cost = self._get_execution_time_for_batch_str()

    def get_old_cost(self):
        return self._get_execution_time_for_batch_str()


def state_to_int(state):
    return reduce(lambda prev, x: prev * 2 + (1 if x else 0), state, 0)


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
