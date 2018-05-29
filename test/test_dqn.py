import pickle
from os import path
from unittest import TestCase

import gym
import numpy as np
from gym.envs import register

import const
from PostgresConnector import PostgresConnector
from dqn import load_agent, ENV_NAME, table_name


class TestDatabaseIndexesEnv(TestCase):
    def test_dqn_against_heuristic(self):
        np.random.seed(103)
        with open(path.join("..", "query_pull_1000v3.pkl"), 'rb') as f:
            query_pull = pickle.load(f)
            workload = np.random.choice(query_pull, const.BATCH_SIZE)
            register(
                id=ENV_NAME,
                entry_point='dbenv:DatabaseIndexesEnv',
                kwargs={'n': const.COLUMNS_AMOUNT,
                        'table_name': table_name,
                        'query_pull': workload,
                        'batch_size': const.BATCH_SIZE,
                        'connector': PostgresConnector(),
                        'k': 3,
                        'max_episodes': 1}
            )
            env = gym.make(ENV_NAME)
            dqn = load_agent(path.join("..", "dqn_{}_weights_better_reward.h5f".format(ENV_NAME)))
            results = dqn.test(env, nb_episodes=1)
            print(results)
            print(env.state)
            print(predict_on_workload(workload))


def predict_on_workload(workload):
    common_column_indexes = list(filter(lambda i: all([query['sf_array'][i] < 1 for query in workload]),
                                        range(const.COLUMNS_AMOUNT)))
    if len(common_column_indexes) > const.N_INDEXES:
        common_column_averages = map(lambda i: (i, sum([query['sf_array'][i] for query in workload]) / 5),
                                     common_column_indexes)
        return [column[0] for column in sorted(common_column_averages, key=lambda x: x[1])[0:const.N_INDEXES]]
    elif len(common_column_indexes) == const.N_INDEXES:
        return common_column_indexes
    else:
        column_averages = map(lambda i: (i, sum([query['sf_array'][i] for query in workload]) / 5),
                              range(const.COLUMNS_AMOUNT))
        additional_columns_number = const.N_INDEXES - len(common_column_indexes)
        return common_column_indexes + [column[0] for column in
                                        sorted(column_averages, key=lambda x: x[1])[0:additional_columns_number]]
