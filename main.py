import os

import pandas as pd

from DBQueriesGenerator import create_table, table_exists
from PostgresConnector import PostgresConnector


def main():
    connector = PostgresConnector()
    # print (connector.query('SELECT * FROM fortest;').fetchall())
    if not table_exists(connector):
        chunksize = 100000
        chunks = []
        for chunk in pd.read_csv('lineitemSF1.csv', chunksize=chunksize, engine='c', low_memory=False, header=None, names=list(['column' + str(x) for x in range(17)])):
            chunks.append(chunk)
        df = pd.concat(chunks, axis=0)
        create_table(df, connector)
if __name__ == "__main__":
    main()