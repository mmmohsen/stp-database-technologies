from unittest import TestCase
import pickle
import numpy as np

from PostgresConnector import PostgresConnector
from const import table_column_types, table_column_names
from db import add_index, drop_indexes, get_estimated_execution_time
from qlearn.main import table_name
from queryPull import generate_query_pull
from qlearn.main import get_indexes_qagent


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
        connector = PostgresConnector()
        drop_indexes(connector, table_name)
        ## though I would strongly recommend ... in the generate query_pull
        ## with checking the file exsistence, if it does exsist .... then load data
        ## from the pickle file
        generate_query_pull('../.test_query_pull_' + str(columns_participating), self.__queries_amount,
                            columns_participating, table_column_types,
                            table_column_names,
                            table_name, connector)
        ## this with open to be inside the generate query pull :D
        with open('../.test_query_pull_' + str(columns_participating), 'rb') as pickleFile:
            queries = pickle.load(pickleFile)

        sf_array = np.array([query['sf_array'] for query in queries]).sum(axis=0)
        indexes_to_add = [i[0] for i in
                          (sorted(enumerate(sf_array), key=lambda x: x[1]))[:self.__index_amount]]
        for index in indexes_to_add:
            add_index(connector, index, table_name)
        heuristically_estimated_execution_time = 0
        for query in queries:
            heuristically_estimated_execution_time += get_estimated_execution_time(connector, query['query'])
        drop_indexes(connector, table_name)
        indexes_to_add = get_indexes_qagent(self.__index_amount, queries, True)
        # extra clean up to make sure no indices left from the agent
        drop_indexes(connector, table_name)
        print(indexes_to_add)
        for index in indexes_to_add:
            add_index(connector, index, table_name)
        qlearning_estimated_execution_time = 0
        for query in queries:
            qlearning_estimated_execution_time += get_estimated_execution_time(connector, query['query'])
        print(heuristically_estimated_execution_time, ' ', qlearning_estimated_execution_time)
        self.assertGreater(heuristically_estimated_execution_time, qlearning_estimated_execution_time)
