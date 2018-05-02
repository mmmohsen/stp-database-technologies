import os


def create_table(dataset, connector):
    query = []
    for col in dataset.columns:
        query.append(col + " text")
    connector.query("CREATE TABLE " + os.environ['TABLENAME'], + "(" + ",".join(query) + ");")
    i=0
    for index, row in dataset.iterrows():
        if i>50:
            break
        vals = []
        for item in row:
            vals.append("'" + str(item) + "'")
        vals = ",".join(vals)
        connector.query("INSERT INTO " + os.environ['TABLENAME'], + " VALUES (" + vals + ");")
    connector.commit()


def table_exists(connector):
    return bool(
        connector.query("select exists(select relname from pg_class where relname='" + os.environ['TABLENAME'] + "')").fetchone()[0])

def add_index(connector, column_number):
    connector.query("CREATE INDEX column"+column_number+"_index ON "+os.environ['TABLENAME']+"(column"+column_number+")")
    connector.commit()

# def drop_indexes(connector):
#