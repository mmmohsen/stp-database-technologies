import random
from unittest import TestCase
import numpy as np
from scipy.stats import stats
import itertools
import os, json

from PostgresConnector import PostgresConnector
from const import table_column_types, table_column_names, COLUMNS_AMOUNT
from db import add_index, drop_indexes, get_estimated_execution_time, get_estimated_execution_time_median
from main import table_name
from queryPull import generate_query_pull
from qlearn.main import get_indexes_qagent
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

    def test_against_tpch(self):
        """test our heuristic algorithm, Q-learning, supervised and random approach with TPC-H Queries"""
        def get_execution_time_for_indexes_configuration(indexes):
            total_time = 0
            for index in indexes:
                add_index(connector, index, "lineitem")
                total_time = 0
            for query in queries:
                total_time += get_estimated_execution_time_median(connector, query['query'], 3)
            drop_indexes(connector, "lineitem")
            return total_time

        def add_execution_time_for_method_and_indexes_configuration(method, indexes):
            if method in methods:
                methods[method].append(get_execution_time_for_indexes_configuration(indexes))
            else:
                methods[method] = [get_execution_time_for_indexes_configuration(indexes)]

        connector = PostgresConnector()
        drop_indexes(connector, "lineitem")
        methods = {}
        i = 0
        np.warnings.filterwarnings('ignore')
        total_amount_of_rows = connector.query("select count (*) from lineitem;").fetchone()[0]
        queries = []
        sf_array = []
        with open("../tpc_h_queries/tpch.json") as infile:
            json_obj = json.load(infile)
        for elem in json_obj:
            for subquery in elem["subquery"]:
                sf_array.append(
                    float(connector.query(subquery).fetchone()[0]) / float(total_amount_of_rows))
            queries.append({'query': elem["query"], 'sf_array': sf_array})

        sf_array = np.array([query['sf_array'] for query in queries]).sum(axis=0)

        indexes_to_add = [i[0] for i in
                          (sorted(enumerate(sf_array), key=lambda x: x[1]))[:self.__index_amount]]
        add_execution_time_for_method_and_indexes_configuration('heuristic', indexes_to_add)

        indexes_to_add = get_indexes_qagent(self.__index_amount, queries, True)
        add_execution_time_for_method_and_indexes_configuration('qlearning', indexes_to_add)
        drop_indexes(connector, table_name)

        indexes_to_add = get_indexes_supervised(self.__index_amount, queries)
        add_execution_time_for_method_and_indexes_configuration('supervised', indexes_to_add)

        indexes_to_add = random.sample(range(COLUMNS_AMOUNT), self.__index_amount)
        add_execution_time_for_method_and_indexes_configuration('random', indexes_to_add)

        times_combinations = list(itertools.combinations(methods.values(), 2))
        p_values = [stats.ttest_ind(time[0], time[1])[1] for time in times_combinations]
        print(p_values)
        print('try #' + str(i))
        for method, times in methods.items():
            print('{}: {}'.format(method, np.mean(times)))
        print('')

        for method, times in methods.items():
            print('{}: {}'.format(method, np.mean(times)))