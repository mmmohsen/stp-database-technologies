import os

import numpy as np
import gym
from gym.envs import register

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import os_params_values
from PostgresConnector import PostgresConnector
from query import generate_query
from rl.processors import Processor


class CustomProcessor(Processor):
    def __init__(self, action_space):
        self.action_space = action_space

    def process_action(self, action):
        # Here we should filter out from the batch the action that are not valid...
        if self.action_space.contains(action):
            return action
        else:
            return -1


table_name = os.environ["TABLENAME"]
table_column_types = ['integer', 'integer', 'integer', 'integer', 'integer', 'decimal', 'decimal', 'decimal', 'char(1)',
                      'char(1)', 'date', 'date', 'date', 'text', 'text', 'text', 'text']
num_queries_batch = 1
table_column_names = list(['column' + str(x) for x in range(17)])
connector = PostgresConnector()
np.random.seed(123)
query_batch = list(generate_query(table_column_names, table_column_types) for _ in range(num_queries_batch))

ENV_NAME = 'DatabaseIndexesEnv-v0'
register(
        id=ENV_NAME,
        entry_point='dbenv:DatabaseIndexesEnv',
        kwargs={'n': len(table_column_names), 'table_name': table_name, 'query_batch': query_batch,
                'connector': connector, 'k': 3}
    )

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
env.seed(123)

nb_actions = len(table_column_names)

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(env.action_space.n))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=1000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy, processor=CustomProcessor(env.action_space))
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=50, visualize=False, verbose=1)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=False)