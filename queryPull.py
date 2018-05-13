import os.path
import pickle
import random
from PostgresConnector import PostgresConnector
from query import generate_query


def generate_query_pull(file_path, queries_amount, selected_columns_amount_range, table_column_types,
                        table_column_names, table_name, connector, first_run=True):
    if first_run:
        if not os.path.exists(file_path):
            participating_columns = [[column_name, column_type, index] for index, (column_name, column_type) in
                                     enumerate(zip(table_column_names, table_column_types)) if
                                     column_type == "integer" or column_type == "date"]
            selected_columns_amount = min(
                random.randint(min(selected_columns_amount_range), max(selected_columns_amount_range)),
                len(participating_columns))
            total_amount_of_rows = PostgresConnector().query("select count (*) from " + table_name + ";").fetchone()[0]
            data = [generate_query(selected_columns_amount, participating_columns, table_name, connector,
                                   len(table_column_names), total_amount_of_rows) for i in
                    range(queries_amount)]
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
    else:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    return data
