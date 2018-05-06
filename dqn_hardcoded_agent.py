"""
based on
Deep Q-Learning example using OpenAI gym CartPole environment
Source: https://github.com/keon/deep-q-learning
"""

import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import copy
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
from query import generate_query
num_actions = 3
num_queries_batch = 5
NUM_EPISODES = 35
load_weights_from_file = False
# the batch to be loaded from the memory .... deep q learning replay trick
batch_size = 1

table_name = os.environ["TABLENAME"]
table_column_types = ['integer', 'integer', 'integer', 'integer', 'integer', 'decimal', 'decimal', 'decimal', 'char(1)',
                      'char(1)', 'date', 'date', 'date', 'text', 'text', 'text', 'text']
table_column_names = list(['column' + str(x) for x in range(17)])



def state_to_string(state):
    return ''.join(str(int(e)) for e in state)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        # the input layer
        model.add(Dense(8, input_dim=self.state_size, activation='relu'))
        # four  hidden layers with 8 neurons
        model.add(Dense(8, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(8, activation='relu'))
        # the output layer
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    # need to fine tune what states to be kept inside of the table
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, strategy):
        if np.random.rand() <= self.epsilon:
            strategy.append("exploration")
            return env.action_space.sample()
        copied_state = copy.copy(state)
        copied_state  = np.reshape(state, [1, self.state_size])
        strategy.append("exploitation")
        act_values = self.model.predict(copied_state)
        return np.argmax(act_values [0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                copied_state = copy.copy(next_state)
                copied_state = np.reshape(next_state, [1, self.state_size])
                target = (reward + self.gamma *
                          np.amax(self.model.predict(copied_state)[0]))
            copied_state = copy.copy(state)
            copied_state = np.reshape(state, [1, self.state_size])
            target_f = self.model.predict(copied_state)
            target_f[0][action] = target
            self.model.fit(copied_state, target_f, epochs=1, verbose=2)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":

    connector = PostgresConnector()
    if not table_exists(connector, table_name):
        create_table_2(connector)
        load_table(connector)

    #    make results repeatable
    np.random.seed(123)
    query_batch = list(generate_query(table_column_names, table_column_types) for _ in range(num_queries_batch))

    # gym configuration
    register(
        id='DatabaseIndexesEnv-v0',
        entry_point='dbenv:DatabaseIndexesEnv',
        kwargs={'n': len(table_column_names), 'table_name': table_name, 'query_batch': query_batch,
                'connector': connector, 'k': 3}
    )

    env = gym.make('DatabaseIndexesEnv-v0')

    state_size = env.action_space.n
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    if load_weights_from_file:
        agent.load("mydqn.h5")
    done = False

    for episode in range(NUM_EPISODES):
        state = env.reset()
        actions_taken = list()
        # decay the exploration as the number of episodes grows, the Q table becomes more mature
        # get batch of 5 queries, update the corresponding query batch of the environment
        # query_batch = list(generate_query(table_column_names, table_column_types) for _ in range(num_queries_batch))
        # env.set_query_batch(query_batch)
        episode_total_reward = 0
        episode_strategy = []
        ## now the learning comes
        episode_strategy = []
        for _ in range(3):
            action = agent.act(state, episode_strategy)
            actions_taken.append(action)
            old_state = copy.copy(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            episode_total_reward += reward
            state = next_state
            if done:
                break

        if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        if episode % 10 == 0:
            agent.save("./mydqn.h5")

        actions_taken_s = ','.join(str(e) for e in actions_taken)
        print(
            "episode num = '{0}', episode_total_reward = '{1}', current_state = '{2}', actions_taken = '{3}', strategy = {4}"
                .format(float(episode), float(episode_total_reward), state_to_string(state), actions_taken_s,
                        episode_strategy))

