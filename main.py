import os

import gym
import numpy as np
import pandas as pd
import os_params_values
import matplotlib.pyplot as plt
from gym.envs import register
# from typing import Dict, Any
import csv
from PostgresConnector import PostgresConnector
from db import create_table, table_exists, create_table_2, load_table
# change this config for different data types
from dbenv import state_to_int
from queryPull import generate_query_pull
from timeit import default_timer as timer
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

table_name = os.environ["TABLENAME"]
COLUMNS_AMOUNT = 17
table_column_types = ['integer', 'integer', 'integer', 'integer', 'integer', 'decimal', 'decimal', 'decimal', 'char(1)',
                      'char(1)', 'date', 'date', 'date', 'text', 'text', 'text', 'text']
table_column_names = list(['column' + str(x) for x in range(COLUMNS_AMOUNT)])

num_workloads = 2


class Q_table_c:
    Q_table = {}  # type: Dict[Any, Any]


NUM_EPISODES = 2
GAMMA = 0.9  # the cummulative reward, how much the future reward matters
ALPHA = 0.01  # the learning rate, worth tunning.
exploration_rate = 1.0  # represents the exploration rate to be decayed by the time
num_actions = 3
num_queries_batch = 5  # the number of queries per workload.
min_exp_rate = 0.01
queries_amount = num_queries_batch * num_workloads  # the queries to be generated

""" 
No two workloads can share the same Q_table_c.Q_table.
"""

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


def run_qlearning(query_pull, connector):
    if not table_exists(connector, table_name):
        create_table_2(connector)
        load_table(connector)

    #   make results repeatable
    np.random.seed(123)
    query_batch = list()
    # gym configuration
    register(
        id='DatabaseIndexesEnv-v0',
        entry_point='dbenv:DatabaseIndexesEnv',
        kwargs={'n': len(table_column_names), 'table_name': table_name, 'query_batch': query_batch,
                'connector': connector, 'k': 3}
    )

    env = gym.make('DatabaseIndexesEnv-v0')

    current_query_idx = 0

    for workload in range(num_workloads):

        query_batch = list()
        Q_table_c.Q_table = {}
        query_batch = list()
        workload_selectivity_l = list()
        # 1. generate the queries per workload
        # 2. generate the cummlative selectivity per workload
        start = timer()
        for i in range(current_query_idx, current_query_idx + num_queries_batch):
            query_batch.append(query_pull[i]['query'])
            workload_selectivity_l.append(map(lambda x: x if x != 1 else 0, query_pull[i]['sf_array']))

        current_query_idx += num_queries_batch
        workload_selectivity = [sum(x) for x in zip(*workload_selectivity_l)]

        # so what is -5 .... it represent columns with high selectivity
        # in the normal case it will adding 1 + 1 + 1 + 1 + 1 = 5
        # this can conflict with the addition of workload selectivity
        # remember that I'm considering the features are selectivity per workload not per column
        # to not also break the concept of normalization I have choosen -5, can be -1 but not 0
        workload_selectivity = map(lambda x: x if x != 0 else -5, workload_selectivity)
        # set the corrsponding batch, here please notice it is an array of strings not an array of query objects.
        env.set_query_batch(query_batch)

        actions_taken = list()
        # the good old q learning we discussed before, Epsilon greedy policy
        # one should clear the cache.
        env.clear_cache()
        for episode in range(NUM_EPISODES):
            state = env.reset()
            actions_taken = list()
            # decay the exploration as the number of episodes grows, the Q table becomes more mature
            eps = exploration_rate / np.sqrt(episode + 1)
            eps = max(eps, min_exp_rate)
            episode_total_reward = 0
            episode_strategy = []
            ## now the learning comes
            for _ in range(3):
                # do exploration, i.e., choose a random actions
                # make sure the last step is exploitation unless the state is new
                if is_new_state(state) or (np.random.uniform(0, 1) < eps and episode != NUM_EPISODES - 1):
                    episode_strategy.append("explore")
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
                    while (action == next_action):
                        next_action = env.action_space.sample()
                    next_action_q_value = 0

                else:
                    next_action, next_action_q_value = get_action_maximum_reward(state_new)

                Q_table_c.Q_table[state_old_int][action] += ALPHA * (reward + GAMMA * next_action_q_value -
                                                                     Q_table_c.Q_table[state_old_int][action])
                state, action = state_new, next_action
            actions_taken_s = ','.join(str(e) for e in actions_taken)
            print(
                "episode num = '{0}', episode_total_reward = '{1}', current_state = '{2}', actions_taken = '{3}', strategy = {4}"
                    .format(episode, float(episode_total_reward), state_to_string(state), actions_taken_s,
                            episode_strategy))
            with open(os.environ["QLEARNINGLOG"], 'a') as myfile:
                myfile.write(
                    "episode num = '{0}', episode_total_reward = '{1}', current_state = '{2}', actions_taken = '{3}', strategy = {4} \n"
                    .format(episode, float(episode_total_reward), state_to_string(state), actions_taken_s,
                            episode_strategy))

        # with the assumption q learning is doing fine, the end of the episode shall give you the highest reward
        # thus the best action, choose them as the best action for this workload
        indices_arr = [0] * COLUMNS_AMOUNT
        # please remember I'm c/c++ programmer, if you can do it in fancy python way. go ahead
        for action in actions_taken:
            indices_arr[action] = 1
        end = timer()
        print(end - start)
        # save features/labels to a csv file
        with open(os.environ["GENERATED_DATA"], 'a') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(workload_selectivity + indices_arr)


# why xgboost, it wins on kaggle
# tabular data no fancy neural network needed
# NOTE: xgboost doesn't support multi-labeled data
# the solution is to use OneVsRest classifier
def build_xgboost_model(test_size=0.33):
    dataset = loadtxt(os.environ["GENERATED_DATA"], delimiter=",")
    X = dataset[:, 0:COLUMNS_AMOUNT]
    Y = dataset[:, COLUMNS_AMOUNT:]
    seed = 7
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # fit model no training data
    model = XGBClassifier()
    classif = OneVsRestClassifier(model)
    classif.fit(X_train, y_train)
    y_pred = classif.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    return {'classifier': classif, 'accuracy': accuracy}


def evaluate_model():
    print "coming soon :D"


def main():
    print "#########################################################################################################"
    print "1. Generating the queries and the corresponding selectivities"
    print "#########################################################################################################"
    connector = PostgresConnector()
    query_pull = generate_query_pull('.query_pull', queries_amount, [4, 6], table_column_types, table_column_names,
                                     table_name, connector, first_run=False)
    start = timer()
    print "#########################################################################################################"
    print "2. Begin the Q-learning & generating the data."
    print "#########################################################################################################"
    run_qlearning(query_pull, connector)
    end = timer()
    print(end - start)

    with open(os.environ["OVERALLLOG"], 'a') as myfile:
        myfile.write(
            "num of workloads = '{0}', num of episodes per workload = '{1}', qlearning total time in (ms) = '{2}' \n"
                .format(num_workloads, NUM_EPISODES, end - start))

    start = timer()

    print "#########################################################################################################"
    print "3. build the supervised learning model"
    print "#########################################################################################################"

    accuracy = build_xgboost_model()['accuracy']
    end = timer()

    with open(os.environ["OVERALLLOG"], 'a') as myfile:
        myfile.write(
            "accuarcy = '{0}', training time in (ms) = '{1}'\n"
                .format(accuracy, end - start ))

    print "#########################################################################################################"
    print "4. Evaluate the model"
    print "#########################################################################################################"
    evaluate_model()


if __name__ == "__main__":
    main()
