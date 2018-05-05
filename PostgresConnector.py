import os

import psycopg2


class PostgresConnector(object):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
            db_config = {'dbname': os.environ['DBNAME'], 'host': 'localhost',
                         'password': os.environ['PASSWORD'], 'port': os.environ['PORT'], 'user': os.environ['USER']}
            try:
                print('connecting to PostgreSQL database...')
                connection = PostgresConnector._instance.connection = psycopg2.connect(**db_config)
                cursor = PostgresConnector._instance.cursor = connection.cursor()
                cursor.execute('SELECT VERSION()')
                db_version = cursor.fetchone()

            except Exception as error:
                raise Exception(error, 'Error: connection not established')

            else:
                print('connection established\n{}'.format(db_version[0]))

        return cls._instance

    def __init__(self):
        self.connection = self._instance.connection
        self.cursor = self._instance.cursor

    def query(self, query):
        try:
            self.cursor.execute(query)
        except Exception as error:
            raise Exception(error, "Error while executing query")
        else:
            return self.cursor

    def commit(self):
        self.connection.commit()

    def __del__(self):
        if hasattr(self, "connection"):
            self.connection.close()
        if hasattr(self, "cursor"):
            self.cursor.close()
