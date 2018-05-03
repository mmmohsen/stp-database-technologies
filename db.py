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
    connector.query(connector.query("SELECT 'DROP INDEX ' || string_agg(indexrelid::regclass::text, ', ')"
                                    " FROM   pg_index  i"
                                    " LEFT   JOIN pg_depend d ON d.objid = i.indexrelid"
                                    " AND d.deptype = 'i'"
                                    " WHERE  i.indrelid = '" + table_name + "'::regclass"
                                                                            " AND    d.objid IS NULL"
                                                                            ";").fetchone()[0])
    connector.commit()
