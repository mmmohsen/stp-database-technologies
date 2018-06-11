import pickle
from unittest import TestCase

import gym
import numpy as np
from gym.envs import register

import const
from PostgresConnector import PostgresConnector
from dqn.dbenv import state_to_int


class TestDatabaseIndexesEnv(TestCase):
    def test__key_for_state_query(self):
        state = [False, False, False]
        self.assertEqual(state_to_int(state), 0)
        state = [False, False, True]
        self.assertEqual(state_to_int(state), 1)
        state = [False, True, False]
        self.assertEqual(state_to_int(state), 2)
        state = [False, True, True]
        self.assertEqual(state_to_int(state), 3)
        state = [True, False, False]
        self.assertEqual(state_to_int(state), 4)
        state = [True, False, True]
        self.assertEqual(state_to_int(state), 5)
        state = [True, True, False]
        self.assertEqual(state_to_int(state), 6)
        state = [True, True, True]
        self.assertEqual(state_to_int(state), 7)

    def test_cache(self):
        np.random.seed(123)
        with open("..\query_pull_1000v3.pkl", 'rb') as f:
            query_pull = pickle.load(f)
            register(
                id='DatabaseIndexesEnv-v0',
                entry_point='dbenv:DatabaseIndexesEnv',
                kwargs={'n': const.COLUMNS_AMOUNT,
                        'table_name': "test_table",
                        'query_pull': query_pull,
                        'batch_size': 2,
                        'connector': PostgresConnector(),
                        'k': 3,
                        'max_episodes': 1}
            )
            env = gym.make('DatabaseIndexesEnv-v0')
            env.step(0)
            env.step(1)
            env.step(2)
            print(env.cache)
