import pickle
from queryPull import generate_query_pull

import heapq
import os
from sklearn.metrics import hamming_loss
import gym
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC, SVC
from sklearn import neighbors
import os_params_values
import matplotlib.pyplot as plt
from sklearn import preprocessing
from gym.envs import register
from sklearn.metrics import f1_score
# from typing import Dict, Any
from sklearn.model_selection import cross_val_score
import csv
from sklearn.naive_bayes import GaussianNB
from PostgresConnector import PostgresConnector
from db import create_table, table_exists, create_table_2, load_table, get_execution_time, add_index, drop_indexes, \
    get_estimated_execution_time
# change this config for different data types
from dbenvm import state_to_int
from timeit import default_timer as timer
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from numpy import median

table_name = os.environ["TABLENAME"]
COLUMNS_AMOUNT = 17
table_column_types = ['integer', 'integer', 'integer', 'integer', 'integer', 'decimal', 'decimal', 'decimal', 'char(1)',
                      'char(1)', 'date', 'date', 'date', 'text', 'text', 'text', 'text']
table_column_names = list(['column' + str(x) for x in range(COLUMNS_AMOUNT)])


class Q_table_c:
    Q_table = {}  # type: Dict[Any, Any]


NUM_EPISODES = 30
GAMMA = 0.9  # the cummulative reward, how much the future reward matters
exploration_rate = 1.0  # represents the exploration rate to be decayed by the time
num_actions = 3
num_queries_batch = 5  # the number of queries per workload.
min_exp_rate = 0.01
initial_lr = 1.0  # Learning rate
min_lr = 0.003
""" 
No two workloads can share the same Q_table_c.Q_table.
"""

""" 
we want to convert the state from the list representation to be 
a string of indices, to serve as a key for the q table.
"""


def state_to_string(state):
    return ''.join(str(int(e)) for e in state)


def is_new_state(state):
    int_state = state_to_int(state)
    if int_state not in Q_table_c.Q_table:
        return True
    actions_rewards_dict = Q_table_c.Q_table[int_state]
    return not bool(actions_rewards_dict)


"""
returns the action causes the maximum reward and the reward corresponding to that action
"""


def get_action_maximum_reward(state):
    # assert Q_table_c.Q_table[state_to_int(state)], "This state has no corresponding action: %r" % state_to_int(state)
    max_reward = float('-inf')
    int_state = state_to_int(state)
    actions_rewards_dict = Q_table_c.Q_table[int_state]
    for key, val in actions_rewards_dict.items():
        if val > max_reward:
            max_reward = val
            max_key = key
    return max_key, max_reward


register(
    id='DatabaseIndexesEnv-v0',
    entry_point='dbenvm:DatabaseIndexesEnv',
    kwargs={'n': len(table_column_names), 'table_name': table_name, 'query_batch': list(),
            'connector': PostgresConnector(), 'k': 3}
)


def get_indexes_qagent(index_amount, queries, Log=False):
    connector = PostgresConnector()
    if not table_exists(connector, table_name):
        create_table_2(connector)
        load_table(connector)

    #   make results repeatable
    np.random.seed(123)
    # gym configuration
    query_batch = list()

    env = gym.make('DatabaseIndexesEnv-v0')
    env.set_indices_num(index_amount)

    current_query_idx = 0
    query_batch = list()
    for workload in xrange(1):

        exploration_rate = 1.0  # represents the exploration rate to be decayed by the time
        initial_lr = 1.0  # Learning rate
        query_batch = list()
        Q_table_c.Q_table = {}
        query_batch = list()
        workload_selectivity_l = list()
        # 1. generate the queries per workload
        # 2. generate the cummlative selectivity per workload
        start = timer()
        for i in range(current_query_idx, current_query_idx + num_queries_batch):
            query_batch.append(queries[i]['query'])
            workload_selectivity_l.append(map(lambda x: x, queries[i]['sf_array']))
        current_query_idx += num_queries_batch
        workload_selectivity = np.prod(workload_selectivity_l, axis=0).tolist()
        max_workload_selectivity = max(workload_selectivity)
        env.set_query_batch(query_batch)
        actions_taken = list()
        # as a heuristic: the indices with the lowest selectivity
        selectivity_indices = heapq.nsmallest(3, xrange(len(workload_selectivity)), workload_selectivity.__getitem__)
        if Log:
            print "Entering the q learning ..... the process can take time."
            print workload_selectivity
        env.clear_cache()
        for episode in range(NUM_EPISODES):
            state = env.reset()
            actions_taken = list()
            # decay the exploration as the number of episodes grows, the Q table becomes more mature
            eps = exploration_rate / np.sqrt(episode + 1)
            eps = max(eps, min_exp_rate)
            episode_total_reward = 0
            episode_total_qreward = 0
            episode_strategy = []
            eta = max(min_lr, initial_lr * (0.85 ** (episode // 100)))
            ## now the learning comes
            for kk in range(3):
                # do exploration, i.e., choose a random actions
                # make sure the last step is exploitation unless the state is new
                if episode == 0:
                    episode_strategy.append("explore")
                    action = selectivity_indices[kk]
                    Q_table_c.Q_table[state_to_int(state)] = {}
                    Q_table_c.Q_table[state_to_int(state)][action] = 0
                elif (is_new_state(state) or (np.random.uniform(0, 1) < eps)) and episode != NUM_EPISODES - 1:
                    episode_strategy.append("explore")
                    # generate only actions that matches something with selectivity.
                    action = env.action_space.sample()
                    # high selectivity, not a good option for an index
                    while workload_selectivity[action] >= max_workload_selectivity:
                        action = env.action_space.sample()
                    if is_new_state(state):
                        Q_table_c.Q_table[state_to_int(state)] = {}
                    if action not in Q_table_c.Q_table[state_to_int(state)]:
                        Q_table_c.Q_table[state_to_int(state)][action] = 0
                else:
                    # else exploit choose the maximum value from the Q table
                    episode_strategy.append("exploit")
                    action = get_action_maximum_reward(state)[0]

                actions_taken.append(action)
                state_old_int = state_to_int(state)
                state_new, reward, done, _ = env.step(action)
                episode_total_reward += reward
                next_action = 0
                next_action_q_value = 0

                if is_new_state(state_new):
                    next_action = env.action_space.sample()
                    while (action == next_action or workload_selectivity[next_action] >= max_workload_selectivity):
                        next_action = env.action_space.sample()
                    next_action_q_value = 0

                else:
                    next_action, next_action_q_value = get_action_maximum_reward(state_new)

                Q_table_c.Q_table[state_old_int][action] += eta * (reward + GAMMA * next_action_q_value -
                                                                   Q_table_c.Q_table[state_old_int][action])
                episode_total_qreward += Q_table_c.Q_table[state_old_int][action]
                state, action = state_new, next_action
            actions_taken_s = ','.join(str(e) for e in actions_taken)
            if Log:
                print(
                    "episode num = '{0}', episode_total_immediate_rewards = '{1}', episode_total_reward = '{2}', current_state = '{3}', actions_taken = '{4}', strategy = {5}"
                        .format(episode, float(episode_total_reward), float(episode_total_qreward),
                                state_to_string(state), actions_taken_s,
                                episode_strategy))

        return actions_taken
