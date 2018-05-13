import os

import gym
import numpy as np
import pandas as pd
import os_params_values
from gym.envs import register
# from typing import Dict, Any

from PostgresConnector import PostgresConnector
from db import create_table, table_exists, create_table_2, load_table
# change this config for different data types
from dbenv import state_to_int
from queryPull import generate_query_pull

table_name = os.environ["TABLENAME"]
COLUMNS_AMOUNT = 17
table_column_types = ['integer', 'integer', 'integer', 'integer', 'integer', 'decimal', 'decimal', 'decimal', 'char(1)',
                      'char(1)', 'date', 'date', 'date', 'text', 'text', 'text', 'text']
table_column_names = list(['column' + str(x) for x in range(COLUMNS_AMOUNT)])

queries_amount = 1

Q_table = {}  # type: Dict[Any, Any]
NUM_EPISODES = 100
GAMMA = 0.9  # the cummulative reward, how much the future reward matters
ALPHA = 0.01  # the learning rate
exploration_rate = 1.0  # represents the exploration rate to be decayed by the time
num_actions = 3
num_queries_batch = 5
min_exp_rate = 0.01

""" 
we want to convert the state from the list representation to be 
a string of indices, to serve as a key for the q table.
"""


def state_to_string(state):
    return ''.join(str(int(e)) for e in state)


"""
The Q table shall be in the form of Q[state][action] where the state is our representation
Q["0000000000"][1] = reward ... index conforts to an action.

if the state is there 

because the Q table is large, I will prefer to build the Q table on the fly.
I mean check first if you had the state already, if not then add it to the Q table
"""

"""
WAAAAAAAAAAARNING Buggggggs a heads

"""


def is_new_state(state):
    int_state = state_to_int(state)
    if int_state not in Q_table:
        return True
    actions_rewards_dict = Q_table[int_state]
    return not bool(actions_rewards_dict)


"""
returns the action causes the maximum reward and the reward corresponding to that action
"""


def get_action_maximum_reward(state):
    # assert Q_table[state_to_int(state)], "This state has no corresponding action: %r" % state_to_int(state)
    max_reward = float('-inf')
    int_state = state_to_int(state)
    actions_rewards_dict = Q_table[int_state]
    for key, val in actions_rewards_dict.items():
        if val > max_reward:
            max_reward = val
            max_key = key
    return max_key, max_reward


def run_qlearning():
    connector = PostgresConnector()
    query_pull = generate_query_pull('.query_pull', queries_amount, [4, 6], table_column_types, table_column_names,
                                     table_name, connector, first_run= True)
    print query_pull
    # if not table_exists(connector, table_name):
    #     create_table_2(connector)
    #     load_table(connector)
    #
    #   # make results repeatable
    # np.random.seed(123)
    # query_batch = list(generate_query(table_column_names, table_column_types) for _ in range(num_queries_batch))
    # gym configuration
    # register(
    #     id='DatabaseIndexesEnv-v0',
    #     entry_point='dbenv:DatabaseIndexesEnv',
    #     kwargs={'n': len(table_column_names), 'table_name': table_name, 'query_batch': query_batch,
    #             'connector': connector, 'k': 3}
    # )
    #
    # env = gym.make('DatabaseIndexesEnv-v0')
    # for episode in range(NUM_EPISODES):
    #     state = env.reset()
    #     actions_taken = list()
    #     # decay the exploration as the number of episodes grows, the Q table becomes more mature
    #     eps = exploration_rate / np.sqrt(episode + 1)
    #     eps = max(eps, min_exp_rate)
    #     # get batch of 5 queries, update the corresponding query batch of the environment
    #     # query_batch = list(generate_query(table_column_names, table_column_types) for _ in range(num_queries_batch))
    #     # env.set_query_batch(query_batch)
    #     episode_total_reward = 0
    #     episode_strategy = []
    #     ## now the learning comes
    #
    #     for _ in range(3):
    #         # do exploration, i.e., choose a random actions
    #
    #         if is_new_state(state) or np.random.uniform(0, 1) < eps:
    #             episode_strategy.append("explore")
    #             action = env.action_space.sample()
    #             if is_new_state(state):
    #                 Q_table[state_to_int(state)] = {}
    #             Q_table[state_to_int(state)][action] = 0
    #         else:
    #             # else exploit choose the maximum value from the Q table
    #             episode_strategy.append("exploit")
    #             action = get_action_maximum_reward(state)[0]
    #         actions_taken.append(action)
    #         state_old_int = state_to_int(state)
    #         state_new, reward, done, _ = env.step(action)
    #         episode_total_reward += reward
    #         next_action = 0
    #         next_action_q_value = 0
    #         if is_new_state(state_new):
    #             next_action = env.action_space.sample()
    #             while (action == next_action):
    #                 next_action = env.action_space.sample()
    #             next_action_q_value = 0
    #
    #         else:
    #             next_action, next_action_q_value = get_action_maximum_reward(state_new)
    #         Q_table[state_old_int][action] += ALPHA * (reward + GAMMA * next_action_q_value -
    #                                                    Q_table[state_old_int][action])
    #         state, action = state_new, next_action
    #     actions_taken_s = ','.join(str(e) for e in actions_taken)
    #     print(
    #         "episode num = '{0}', episode_total_reward = '{1}', current_state = '{2}', actions_taken = '{3}', strategy = {4}"
    #             .format(float(episode), float(episode_total_reward), state_to_string(state), actions_taken_s,
    #                     episode_strategy))


def main():
    run_qlearning()


if __name__ == "__main__":
    main()
