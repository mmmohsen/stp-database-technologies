import os

import pandas as pd

from DBQueriesGenerator import create_table, table_exists, get_execution_time, add_index, drop_indexes
from PostgresConnector import PostgresConnector


def main():
    connector = PostgresConnector()
    # print (connector.query('SELECT * FROM fortest;').fetchall())
    if not table_exists(connector):
        df = pd.read_csv('lineitemSF1.csv', header=None, names=list(['column' + str(x) for x in range(17)]))
        create_table(df, connector)
    query = "SELECT * from test_data where column3 = '1'"
    print(get_execution_time(connector, query))
    add_index(connector, 3)
    print(get_execution_time(connector, query))
    drop_indexes(connector)
if __name__ == "__main__":
    main()