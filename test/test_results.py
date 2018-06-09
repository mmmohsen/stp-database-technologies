import random
from os import path
from unittest import TestCase

import gym
import numpy as np
from scipy.stats import stats
import itertools

import const
from PostgresConnector import PostgresConnector
from const import table_column_types, table_column_names, COLUMNS_AMOUNT
from db import add_index, drop_indexes, get_estimated_execution_time, get_estimated_execution_time_median
from dbenv import DatabaseIndexesEnv
from dqn import ENV_NAME, load_agent
from main import table_name
from qlearn.main import get_indexes_qagent
from queryPull import generate_query_pull


# from qlearn.main import get_indexes_qagent
# from supervised.main import get_indexes_supervised
from supervised.main import get_indexes_supervised


class TestResults(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestResults, self).__init__(*args, **kwargs)
        self.__queries_amount = 5
        self.__index_amount = 3

    def test_results_for_two_columns(self):
        print('Test model for two columns participating')
        self.__test_results(2)

    def test_results_for_four_columns(self):
        print('Test model for four columns participating')
        self.__test_results(4)

    def test_results_with_ten_indexes(self):
        print('Test model for ten columns participating')
        self.__test_results(10)

    def __test_results(self, columns_participating):

        def get_execution_time_for_indexes_configuration(indexes):
            total_time = 0
            for index in indexes:
                add_index(connector, index, table_name)
                total_time = 0
            for query in queries:
                total_time += get_estimated_execution_time_median(connector, query['query'], 3)
            drop_indexes(connector, table_name)
            return total_time

        def add_execution_time_for_method_and_indexes_configuration(method, indexes):
            if method in methods:
                methods[method].append(get_execution_time_for_indexes_configuration(indexes))
            else:
                methods[method] = [get_execution_time_for_indexes_configuration(indexes)]

        def get_indexes_dqn():
            env = DatabaseIndexesEnv(n=const.COLUMNS_AMOUNT,
                                     table_name=table_name,
                                     query_pull=queries,
                                     batch_size=const.BATCH_SIZE,
                                     connector=connector,
                                     k=3,
                                     max_episodes=1)
            dqn = load_agent(path.join("..", "dqn_{}_weights_better_reward.h5f".format(ENV_NAME)))
            dqn.test(env, nb_episodes=1)
            return [i for i, x in enumerate(env.state) if x]

        connector = PostgresConnector()
        drop_indexes(connector, table_name)
        methods = {}
        i = 0
        np.warnings.filterwarnings('ignore')
        while True:
            queries = generate_query_pull('../.test_query_pull_' + str(columns_participating) + '_' + str(i),
                                          self.__queries_amount,
                                          columns_participating, table_column_types,
                                          table_column_names,
                                          table_name, connector)
            i += 1
            sf_array = np.array([query['sf_array'] for query in queries]).sum(axis=0)

            indexes_to_add = [i[0] for i in
                              (sorted(enumerate(sf_array), key=lambda x: x[1]))[:self.__index_amount]]
            add_execution_time_for_method_and_indexes_configuration('heuristic', indexes_to_add)

            indexes_to_add = get_indexes_qagent(self.__index_amount, queries, True)
            add_execution_time_for_method_and_indexes_configuration('qlearning', indexes_to_add)
            #extra clean up to make sure no indices left from the agent
            drop_indexes(connector, table_name)

            # dqn
            indexes_to_add = get_indexes_dqn()
            add_execution_time_for_method_and_indexes_configuration('dqn', indexes_to_add)
            drop_indexes(connector, table_name)

            indexes_to_add = get_indexes_supervised(self.__index_amount, queries)
            add_execution_time_for_method_and_indexes_configuration('supervised', indexes_to_add)

            indexes_to_add = random.sample(range(COLUMNS_AMOUNT), self.__index_amount)
            add_execution_time_for_method_and_indexes_configuration('random', indexes_to_add)

            times_combinations = list(itertools.combinations(methods.values(), 2))
            p_values = [stats.ttest_ind(time[0], time[1])[1] for time in times_combinations]
            print(p_values)
            if all(p_value < 0.01 for p_value in p_values) and i >= 10 or i >= 50:
                break
            print('try #' + str(i))
            for method, times in methods.items():
                print('{}: {}'.format(method, np.mean(times)))
        print('')

        for method, times in methods.items():
            print('{}: {}'.format(method, np.mean(times)))
