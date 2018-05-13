import random

import numpy as np

from db import get_min_value, get_max_value

POSTGRES_MAX_INT = 2147483647
POSTGRES_MIN_INT = -2147483648


def generate_query(selected_columns_amount, participating_columns, table_name, connector, total_columns_amount,
                   total_amount_of_rows=None):
    """
    Generates random range query for the column amount given.
    -> selects the column to be used in query randomly
    -> selects the predicated for every column in a query randomly
    -> select the data value of column randomly
    """
    idxs = np.random.choice(len(participating_columns), selected_columns_amount, replace=False)
    participating_columns = np.array(participating_columns)
    columns_to_query = [Column(column) for column in participating_columns[idxs]]
    query = Query(table_name, connector, total_columns_amount)
    for column in columns_to_query:
        min_value = get_min_value(connector, table_name, column.name)
        max_value = get_max_value(connector, table_name, column.name)
        if column.type == 'integer':
            bounds = np.random.randint(min_value, max_value, 2)
        else:
            min_date = np.datetime64(min_value)
            max_date = np.datetime64(max_value)
            bounds = random_date(min_date, max_date), random_date(min_date, max_date)
        column.set_bounds(bounds)
        query.add_column(column)
    return query.generate_query_row(total_amount_of_rows)


def random_date(start, end):
    """Generate a random datetime between `start` and `end`"""
    return start + np.random.choice(np.arange(0, end - start))


class Query(object):
    def __init__(self, table_name, connector, total_columns_amount):
        self.columns = []
        self.table_name = table_name
        self.connector = connector
        self.sf_array = []
        self.total_columns_amount = total_columns_amount

    def add_column(self, column):
        self.columns.append(column)

    def build_query(self):
        where_clause = "WHERE " + " AND ".join(
            ["{0} >= {1} AND {0} <= {2}".format(column.name, column.bounds[0], column.bounds[1])
             for column in self.columns]
        )
        return "SELECT * FROM {} ".format(self.table_name) + where_clause + ";"

    def generate_query_row(self, total_amount_of_rows=None):
        """generates query with SF for each column selected"""

        if total_amount_of_rows == None:
            total_amount_of_rows = self.connector.query("select count (*) from " + self.table_name + ";").fetchone()[0]
        count_query_for_column = "select count (*) from " + self.table_name + " where {0} >= {1} AND {0} <= {2};"
        for i in range(self.total_columns_amount):
            column_participating = list(filter(lambda column: column.index == i, self.columns))
            if column_participating:
                rows_selected = count_query_for_column.format(column_participating[0].name,
                                                              column_participating[0].bounds[0],
                                                              column_participating[0].bounds[1])
                self.sf_array.append(
                    float(self.connector.query(rows_selected).fetchone()[0]) / float(total_amount_of_rows))
            else:
                self.sf_array.append(1)

        return {'query': self.build_query(), 'sf_array': self.sf_array}


class Column(object):
    def __init__(self, received_column):
        self.name = received_column[0]
        self.type = received_column[1]
        self.index = int(received_column[2])
        self.bounds = None

    def set_bounds(self, bounds):
        self.bounds = (min(bounds), max(bounds)) if self.type == 'integer' else (
            "'" + str(min(bounds)) + "'", "'" + str(max(bounds)) + "'")
