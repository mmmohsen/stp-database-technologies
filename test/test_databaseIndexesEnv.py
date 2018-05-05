from unittest import TestCase

import gym
import numpy as np
from gym.envs import register

from query import generate_query


class TestDatabaseIndexesEnv(TestCase):
    def test__key_for_state_query(self):
        np.random.seed(123)
        table_column_names = ["col1", "col2", "col3"]
        table_column_types = ["integer", "integer", "integer"]
        query_batch = list(generate_query(table_column_names, table_column_types) for _ in range(5))

        register(
            id='DatabaseIndexesEnv-v0',
            entry_point='dbenv:DatabaseIndexesEnv',
            kwargs={'n': 3, 'table_name': "", 'query_batch': query_batch,
                    'connector': None, 'k': 3}
        )

        env = gym.make('DatabaseIndexesEnv-v0')
        env.state = [False, False, False]
        self.assertEqual(env._key_for_state_query()[0], 0)
        env.state = [False, False, True]
        self.assertEqual(env._key_for_state_query()[0], 1)
        env.state = [False, True, False]
        self.assertEqual(env._key_for_state_query()[0], 2)
        env.state = [False, True, True]
        self.assertEqual(env._key_for_state_query()[0], 3)
        env.state = [True, False, False]
        self.assertEqual(env._key_for_state_query()[0], 4)
        env.state = [True, False, True]
        self.assertEqual(env._key_for_state_query()[0], 5)
        env.state = [True, True, False]
        self.assertEqual(env._key_for_state_query()[0], 6)
        env.state = [True, True, True]
        self.assertEqual(env._key_for_state_query()[0], 7)
