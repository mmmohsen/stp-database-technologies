def create_table(dataset, table_name, table_column_types, connector):
    """
    Creates table for the given representation of csv as pandas table.
    :param dataset:
    :param table_column_types:
    :param connector:
    """
    query = []
    for (col, column_type) in zip(dataset.columns, table_column_types):
        query.append(col + " " + column_type)
    connector.query("CREATE TABLE " + table_name + "(" + ",".join(query) + ");")
    for index, row in dataset.iterrows():
        vals = []
        for item in row:
            vals.append("'" + str(item) + "'")
        vals = ",".join(vals)
        connector.query("INSERT INTO " + table_name + " VALUES (" + vals + ");")
    connector.commit()


def create_table_2(connector):
    query = """
    CREATE TABLE LINEITEM ( column0    INTEGER NOT NULL,
                             column1     INTEGER NOT NULL,
                             column2     INTEGER NOT NULL,
                             column3  INTEGER NOT NULL,
                             column4    DECIMAL(15,2) NOT NULL,
                             column5  DECIMAL(15,2) NOT NULL,
                             column6    DECIMAL(15,2) NOT NULL,
                             column7        DECIMAL(15,2) NOT NULL,
                             column8  CHAR(1) NOT NULL,
                             column9  CHAR(1) NOT NULL,
                             column10    DATE NOT NULL,
                             column11  DATE NOT NULL,
                             column12 DATE NOT NULL,
                             column13 CHAR(25) NOT NULL,
                             column14     CHAR(10) NOT NULL,
                             column15      VARCHAR(44) NOT NULL);

    """
    connector.query(query)
    connector.commit()


def load_table(connector):
    query = """
    
        COPY lineitem FROM '/Users/pegasus/tpch-dbgen/1/lineitem.csv' WITH DELIMITER AS '|';
    """
    connector.query(query)
    connector.commit()


def table_exists(connector, table_name):
    return bool(
        connector.query(
            "select exists(select relname from pg_class where relname='" + table_name + "')").fetchone()[
            0])


def add_index(connector, column_number, table_name):
    connector.query("CREATE INDEX column" + str(column_number) + "_index ON " + table_name + "(column" +
                    str(column_number) + ")")
    connector.commit()


def get_execution_time(connector, query):
    return explain_analyze_query(connector, query)[0][0]['Execution Time']


def explain_analyze_query(connector, query):
    return connector.query("EXPLAIN (ANALYZE true, FORMAT json) " + query).fetchone()


def drop_indexes(connector, table_name):
    fetchone_ = connector.query("SELECT 'DROP INDEX ' || string_agg(indexrelid::regclass::text, ', ')" \
                                " FROM   pg_index  i" \
                                " LEFT   JOIN pg_depend d ON d.objid = i.indexrelid" \
                                " AND d.deptype = 'i'" \
                                " WHERE  i.indrelid = '" + table_name + "'::regclass" \
                                                                        " AND    d.objid IS NULL" \
                                                                        ";").fetchone()[0]
    if fetchone_:
        connector.query(fetchone_)
        connector.commit()


def get_min_value(connector, table_name, column):
    return connector.query("SELECT MIN({0}) FROM {1};".format(column, table_name)).fetchone()[0]


def get_max_value(connector, table_name, column):
    return connector.query("SELECT MAX({0}) FROM {1};".format(column, table_name)).fetchone()[0]
