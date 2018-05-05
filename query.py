import numpy as np

POSTGRES_MAX_INT = 2147483647
POSTGRES_MIN_INT = -2147483648


def generate_query(table_column_names, table_column_types):
    """
    Creates a workload with 5 queries.
    -> Selects number of columns to be used in query randomly
    -> selects the column to be used in query randomly
    -> selects the predicated for every column in a query randomly
    -> select the data value of column randomly
    > Parameters:
        table : str                 | Table on which query will be applied
        dataset : pandas dataframe       | data to be inputed in table will be gathered from dataset
    > Returns:
        workload_queries: list          | list containing 5 queries for the particular workload
        columns_in_workload: set        | set containing columns used in workload
    """
    integer_columns = tuple(filter(lambda it: it[1][1] in ("integer", "INTEGER"),
                                      enumerate(zip(table_column_names, table_column_types))))
    # loop over 5 times to have 5 queries in workload
    # Selects number of columns to be used in query randomly
    number_of_cols = np.random.randint(1, len(integer_columns) + 1)
    columns_to_query = tuple(integer_columns[i] for i in (np.random.choice(len(integer_columns), number_of_cols, replace=False)))

    query = Query(table_column_names, table_column_types)
    # loop over all the columns to be used in query
    for (index, _) in columns_to_query:
        # selects the predicated for every column in a query randomly
        # if col in ['column8', 'column9', 'column13', 'column14', 'column15', 'column16']:
        #    pred = np.random.choice(['LIKE', 'NOT LIKE'])
        # else:
        #    pred = np.random.choice(["=", ">", ">=", "<", "<="])
        # select the data value of column randomly
        bounds = np.random.randint(POSTGRES_MIN_INT, POSTGRES_MAX_INT, 2)
        query.bounds[index] = (min(bounds), max(bounds))
    return query


class Query(object):
    def __init__(self, table_column_names, table_column_types, bounds=None):
        self.table_column_types = table_column_types
        self.table_column_names = table_column_names
        self.bounds = bounds
        if not bounds:
            self.bounds = list(None for _ in range(len(table_column_names)))

    def build_query(self, table_name):
        where_clause = "WHERE " + " AND ".join(
            ["{0} >= {1} AND {0} <= {2}".format(column_name, left_bound, right_bound)
             for (column_name, (left_bound, right_bound))
             in filter(lambda x: x[1], zip(self.table_column_names, self.bounds))
             ]
        )
        return "SELECT * FROM {} ".format(table_name) + where_clause + ";"
