import os.path
import pickle
import random

from query import generate_query


def generate_query_pull(file_path, queries_amount, selected_columns_amount_range, table_column_types,
                        table_column_names, table_name, connector):
    if not os.path.exists(file_path):
        participating_columns = [[column_name, column_type, index] for index, (column_name, column_type) in
                                 enumerate(zip(table_column_names, table_column_types)) if
                                 column_type == "integer" or column_type == "date"]
        selected_columns_amount = min(
            random.randint(min(selected_columns_amount_range), max(selected_columns_amount_range)),
            len(participating_columns))
        data = [generate_query(selected_columns_amount, participating_columns, table_name, connector, len(table_column_names)) for i in
                range(queries_amount)]
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
