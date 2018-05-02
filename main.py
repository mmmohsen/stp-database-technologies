import os

import pandas as pd

from DBQueriesGenerator import create_table, table_exists
from PostgresConnector import PostgresConnector


def main():
    connector = PostgresConnector()
    # print (connector.query('SELECT * FROM fortest;').fetchall())
    if not table_exists(connector, os.environ['TABLENAME']):
        df = pd.read_csv("lineitemSF1.csv", header=None, names=list(['column' + str(x + 1) for x in range(17)]))
        create_table(os.environ['TABLENAME'], df, connector)


if __name__ == "__main__":
    main()