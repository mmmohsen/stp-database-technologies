import os


def create_table(dataset, connector):
    query = []
    for col in dataset.columns:
        query.append(col + " text")
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
                    " WHERE  i.indrelid = 'test_data'::regclass"
                    " AND    d.objid IS NULL"
                    ";").fetchone()[0])
    connector.commit()