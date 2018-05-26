from unittest import TestCase
import pickle

from PostgresConnector import PostgresConnector
from const import table_column_types, table_column_names
from db import add_index, drop_indexes, get_estimated_execution_time
from main import table_name
from queryPull import generate_query_pull


class TestResults(TestCase):

    def __init__(self):
        super().__init__()
        self.__queries_amount = 10
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
        generate_query_pull('../.query_pull', self.__queries_amount, columns_participating, table_column_types,
                            table_column_names,
                            table_name, connector)
        with open('../.query_pull', 'rb') as pickleFile:
            queries = pickle.load(pickleFile)
        for query in queries:
            indexes_to_add = [i[0] for i in (sorted(enumerate(query['sf_array']), key=lambda x: x[1]))[:self.__index_amount]]
            for index in indexes_to_add:
                add_index(connector, index, table_name)
            heuristically_estimated_execution_time = get_estimated_execution_time(connector, query['query'])
            drop_indexes(connector, table_name)
            indexes_to_add = get_indexes(self.__index_amount, query)
            for index in indexes_to_add:
                add_index(connector, index, table_name)
            model_estimated_execution_time = get_estimated_execution_time(connector, query['query'])
            print(heuristically_estimated_execution_time, ' ', model_estimated_execution_time)

