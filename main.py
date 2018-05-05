import os

import gym
import numpy as np
import pandas as pd
from gym.envs import register

from PostgresConnector import PostgresConnector
from db import create_table, table_exists
# change this config for different data types
from query import generate_query

table_name = os.environ["TABLENAME"]
table_column_types = ['integer', 'integer', 'integer', 'integer', 'integer', 'decimal', 'decimal', 'decimal', 'char(1)',
                      'char(1)', 'date', 'date', 'date', 'text', 'text', 'text', 'text']
table_column_names = list(['column' + str(x) for x in range(17)])

ALPHA = 1.0
GAMMA = 0.8
def main():
    """
    Entry point. Remember to configure DBNAME, USER, PASSWORD, PORT, TABLENAME and FILENAME environment variables.
    DBNAME = name of the database
    USER = user (default = 'postrgres")
    PASSWORD = user's password
    PORT = port on which postgres runs (usually 5432)
    TABLENAME = name of the table to which test data would be exported and the table that will be used for experiments
    FILENAME = .csv that will be used to import data
    """
    connector = PostgresConnector()
    # print (connector.query('SELECT * FROM fortest;').fetchall())
    if not table_exists(connector, table_name):
        df = pd.read_csv(os.environ["FILENAME"], header=None, names=table_column_names)
        create_table(df, table_name, table_column_types, connector)

    # make results repeatable
    np.random.seed(123)

    # get batch of 5 queries
    query_batch = list(generate_query(table_column_names, table_column_types) for _ in range(5))

    # gym configuration
    register(
        id='DatabaseIndexesEnv-v0',
        entry_point='dbenv:DatabaseIndexesEnv',
        kwargs={'n': len(table_column_names), 'table_name': table_name, 'query_batch': query_batch, 'connector': connector, 'k': 3}
    )
    env = gym.make('DatabaseIndexesEnv-v0')
    print(env.state)

    # take 3 random actions
    for _ in range(3):
        action = env.action_space.sample()
        print(action)
        print(env.step(action))


if __name__ == "__main__":
    main()
