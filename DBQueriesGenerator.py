def create_table(table_name, dataset, connector):
    query = []
    for col in dataset.columns:
        query.append(col + " text")
    connector.query("CREATE TABLE " + table_name + "(" + ",".join(query) + ");")

    for index, row in dataset.iterrows():
        vals = []
        for item in row:
            vals.append("'" + str(item) + "'")
        vals = ",".join(vals)
        connector.query("INSERT INTO " + table_name + " VALUES (" + vals + ");")
    connector.commit()


def table_exists(connector, table_str):
    return bool(
        connector.query("select exists(select relname from pg_class where relname='" + table_str + "')").fetchone()[0])
