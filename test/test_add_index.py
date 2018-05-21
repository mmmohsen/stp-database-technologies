import pickle
from unittest import TestCase

from PostgresConnector import PostgresConnector
from const import table_column_types, table_column_names
from db import add_index, get_execution_time, drop_indexes
from main import queries_amount, table_name
from queryPull import generate_query_pull


class TestAdd_index(TestCase):
    def test_add_index(self):
        connector = PostgresConnector()
        drop_indexes(connector, table_name)
        generate_query_pull('../.query_pull', queries_amount, [4, 6], table_column_types, table_column_names,
                                         table_name, connector)
        with open('../.query_pull', 'rb') as pickleFile:
            queries = pickle.load(pickleFile)
        for query in queries:
            execution_time_before = 0
            execution_time_after = 0
            indexes_to_add = [i[0] for i in (sorted(enumerate(query['sf_array']), key=lambda x: x[1]))[:3]]
            for t in range(10):
                execution_time_before += get_execution_time(connector, query['query'])
                for index in indexes_to_add:
                    add_index(connector, index, table_name)
                execution_time_after += get_execution_time(connector, query['query'])
                drop_indexes(connector, table_name)
            print(execution_time_before/10, execution_time_after/10)
            # self.assertLess(execution_time_after, execution_time_before)