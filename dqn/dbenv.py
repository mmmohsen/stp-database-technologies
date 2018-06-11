import itertools
from functools import reduce

import numpy as np

import gym
from gym import spaces

from db import drop_indexes, get_execution_time, add_index, get_estimated_execution_time


class DatabaseIndexesEnv(gym.Env):
    metadata = {
        'render.modes': ['ansi'],
        # 'video.frames_per_second': 50
    }
    """
    Base env providing the actor the means to abstract actions/state. Abstracts all real manipulations with db
    (e.g. taking actions, getting cost).
    """

    def __init__(self, n, table_name, query_pull, batch_size, connector, max_episodes, k=3):
        """
        Constructs new env.
        :param n: number of columns
        :param table_name: name of the table
        :param query_pull: collection of queries
        :param connector: database connection
        :param k: max number of indices
        """
        super(DatabaseIndexesEnv, self).__init__()
        self.batch_size = batch_size
        self.max_episodes = max_episodes
        self.n = n
        self.action_space = Dynamic(n)
        self.observation_space = spaces.Box(low=0, high=1.0, shape=(batch_size, n))
        self.state = list(False for _ in range(n))
        self.table_name = table_name
        self.query_pull = query_pull
        self.query_batch = np.random.choice(query_pull, batch_size)
        self.connector = connector
        self.k = k
        self.step_number = 0
        self.episode_number = 0
        self.cache = {}

    def reset(self):
        self.step_number = 0
        if self.episode_number >= self.max_episodes:
            self.query_batch = np.random.choice(self.query_pull, self.batch_size)
            self.episode_number = 0
        if self.episode_number == 0:
            indexes = touched_indexes(self.query_batch)
            print("New query batch, touched indexes: " + str(indexes))
        self.episode_number += 1
        drop_indexes(self.connector, self.table_name)
        self.state = list(False for _ in range(len(self.state)))
        self.action_space = Dynamic(len(self.state))
        return np.array([self.state, *[x['sf_array'] for x in self.query_batch]])

    def render(self, mode='human'):
        pass
        # no fancy stuff for now
        # if mode == 'ansi':
        print("\n" + ' '.join(('%*s' % (2, x) for x in list(range(self.n)))) + "\n"
              + " ".join(('%*s' % (2, i) for i in (1 if x else 0 for x in self.state))))
        # else:
        #    return super(DatabaseIndexesEnv, self).render(mode)

    def step(self, action):
        self.step_number += 1
        if self.action_space.contains(action):
            self.state[action] = True
            add_index(self.connector, action, self.table_name)
            self.action_space.disable_actions((action,))
            cost = self._get_execution_time_for_batch()
            reward = 1.0 / cost
        else:
            reward = -1.0
        finished = self.step_number >= self.k
        #self.render()
        #print(reward)
        return np.array([self.state, *[x['sf_array'] for x in self.query_batch]]), reward, finished, {}

    def set_query_batch(self, query_batch):
        self.query_batch = query_batch

    def _get_execution_time_for_batch(self):
        return sum((self._get_execution_time_for_query(self.connector, query['query']) for query in self.query_batch))

    def _get_execution_time_for_query(self, connector, built_query):
        try:
            ex_time = self.cache[state_to_int(self.state), built_query]
        except KeyError:
            ex_time = get_estimated_execution_time(connector, built_query)
            self.cache[state_to_int(self.state), built_query] = ex_time
        return ex_time


def state_to_int(state):
    return reduce(lambda prev, x: prev * 2 + (1 if x else 0), state, 0)


def touched_indexes(query_batch):
        sf_arrays = [x['sf_array'] for x in query_batch]
        indexes_of_touched_per_query = [[i for i, sf in enumerate(sf_array) if sf < 1.0] for sf_array in sf_arrays]
        return set(itertools.chain(*indexes_of_touched_per_query))


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
