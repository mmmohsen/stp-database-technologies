import os
import pandas as pd
import random
import numpy


def create_table(dataset, connector):
    query = []
    types = ['integer', 'integer', 'integer', 'integer', 'integer', 'decimal', 'decimal', 'decimal', 'char(1)',
             'char(1)', 'date', 'date', 'date', 'text', 'text', 'text', 'text']
    for (col, type) in zip(dataset.columns, types):
        query.append(col + " " + type)
    connector.query("CREATE TABLE " + os.environ['TABLENAME'] + "(" + ",".join(query) + ");")
    for index, row in dataset.iterrows():
        vals = []
        for item in row:
            vals.append("'" + str(item) + "'")
        vals = ",".join(vals)
        connector.query("INSERT INTO " + os.environ['TABLENAME'] + " VALUES (" + vals + ");")
    connector.commit()


def table_exists(connector):
    return bool(
        connector.query(
            "select exists(select relname from pg_class where relname='" + os.environ['TABLENAME'] + "')").fetchone()[
            0])


def add_index(connector, column_number):
    connector.query("CREATE INDEX column" + str(column_number) + "_index ON " + os.environ[
        'TABLENAME'] + "(column" + str(column_number) + ")")
    connector.commit()


def get_execution_time(connector, query):
    return explain_analyze_query(connector, query)[0][0]['Execution Time']


def explain_analyze_query(connector, query):
    return connector.query("EXPLAIN (ANALYZE true, FORMAT json)  " + query).fetchone()


def drop_indexes(connector):
    connector.query(connector.query("SELECT 'DROP INDEX ' || string_agg(indexrelid::regclass::text, ', ')"
                    " FROM   pg_index  i"
                    " LEFT   JOIN pg_depend d ON d.objid = i.indexrelid"
                    " AND d.deptype = 'i'"
                    " WHERE  i.indrelid = '" + os.environ['TABLENAME'] + "'::regclass"
                    " AND    d.objid IS NULL"
                    ";").fetchone()[0])
    connector.commit()


def query_generator(dataset, table):
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
    workload_queries = []
    columns_in_workload = []
    # loop over 5 times to have 5 queries in workload
    for x in xrange(5):
        # Selects number of columns to be used in query randomly
        number_of_cols = random.choice(xrange(1, len(dataset.columns) + 1))
        # Selects the column to be used in query randomly
        cols_in_query = random.sample(dataset.columns, number_of_cols)
        columns_in_workload.append(cols_in_query)
        query = "select count(*) from " + table + " where "
        cols_per_query = []
        # loop over all the columns to be used in query
        for col in cols_in_query:
            # selects the predicated for every column in a query randomly
            if col in ['column8','column9','column13','column14', 'column15', 'column16']:
                pred = random.choice(['LIKE', 'NOT LIKE'])
            else:
                pred = random.choice(["=", ">", ">=", "<", "<="])
            # select the data value of column randomly
            subquery = col + " " + pred + " " + str(dataset[col].sample(n=1).iloc[0])
            cols_per_query.append(subquery)
        cols_per_query = query + " AND ".join(cols_per_query)
        workload_queries.append(cols_per_query)
    return workload_queries, set(columns_in_workload)
