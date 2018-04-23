from PostgresConnector import PostgresConnector


def main():
    connector = PostgresConnector()
    print (connector.query('SELECT * FROM fortest;').fetchall())

if __name__ == "__main__":
    main()