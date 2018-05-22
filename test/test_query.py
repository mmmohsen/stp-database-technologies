import numpy as np
from unittest import TestCase

from query import Query, generate_query


class TestQuery(TestCase):
    pass
    # def test_build_query(self):
    #     query = Query(("integer", "integer"), ("column1", "column2"), bounds=((0, 1), (1, 2)))
    #     self.assertEqual("SELECT * FROM table WHERE column1>=0 AND column1<=1 AND column2>=1 AND column2<=2",
    #                      query.build_query("table"))
    #
    # def test_generate_query(self):
    #     np.random.seed(123)
    #     for _ in range(100):
    #         query = generate_query(("column1", "column2", "column3"), ("integer", "integer", "char"))
    #         if query.bounds[0]:
    #             self.assertGreater(query.bounds[0][1], query.bounds[0][0])
    #         if query.bounds[1]:
    #             self.assertGreater(query.bounds[1][1], query.bounds[1][0])
    #         self.assertIsNone(query.bounds[2])
