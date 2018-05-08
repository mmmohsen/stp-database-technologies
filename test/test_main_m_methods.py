import numpy as np
from unittest import TestCase

from main import state_to_string
from main import is_new_state
from main import get_action_maximum_reward

class TestMain_m(TestCase):

    def test_state_to_string(self):
        n = 10
        state = list(False for _ in range(n))
        self.assertEqual(state_to_string(state), "0000000000")
        state[1] = True
        self.assertEqual(state_to_string(state), "0100000000")

    # def test_read_from_Q_tables(self):
    #
    #     n = 10
    #     state = list(False for _ in range(n))
    #     action = 1
    #     read_from_Q_table(state, action)
    #     self.assertEqual(read_from_Q_table(state, action), 0)
    #
    # def test_write_to_Q_tables(self):
    #     n = 10
    #     state = list(False for _ in range(n))
    #     action = 1
    #     read_from_Q_table(state, action)
    #     self.assertEqual(read_from_Q_table(state, action), 0)
    #     write_to_Q_table(state, action, 10)
    #     self.assertEqual(read_from_Q_table(state, action), 10)
    #     action2 = 2
    #     self.assertEqual(read_from_Q_table(state, action2), 0)
    #     write_to_Q_table(state, action2, 10)
    #     self.assertEqual(read_from_Q_table(state, action2), 10)

    def test_is_new_state(self):
        n = 10
        state = list(False for _ in range(n))
        action = 40
        self.assertEqual(is_new_state(state), True)
        write_to_Q_table(state, action, 10)
        self.assertEqual(is_new_state(state), False)

    def test_get_max_action_reward(self):
        n = 10
        state = list(False for _ in range(n))
        state[1] = True
        action1 = 1
        write_to_Q_table(state, action1, 10)
        action2 = 2
        write_to_Q_table(state, action2, 5)
        action3 = 3
        write_to_Q_table(state, action3, 0)
        self.assertEqual(get_action_maximum_reward(state)[0], 1)
        self.assertEqual(get_action_maximum_reward(state)[1], 10)
