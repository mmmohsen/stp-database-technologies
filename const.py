COLUMNS_AMOUNT = 17
table_column_types = ['integer', 'integer', 'integer', 'integer', 'integer', 'decimal', 'decimal', 'decimal', 'char(1)',
                      'char(1)', 'date', 'date', 'date', 'text', 'text', 'text', 'text']
table_column_names = list(['column' + str(x) for x in range(COLUMNS_AMOUNT)])
BATCH_SIZE = 5

table_columns_names_1 = ["ORDERKEY", "PARTKEY", "SUPPKEY", "LINENUMBER", "QUANTITY", "EXTENDEDPRICE", "DISCOUNT", "TAX",
                        "RETURNFLAG", "LINESTATUS", "SHIPDATE", "COMMITDATE", "RECEIPTDATE", "SHIPINSTRUCT",
                       "SHIPMODE", "COMMENT", "COMMENT_1"]