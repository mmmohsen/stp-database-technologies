import os
import pickle

import gym
import numpy as np
from gym.envs import register
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy

from PostgresConnector import PostgresConnector
from const import COLUMNS_AMOUNT, BATCH_SIZE

table_name = os.environ["TABLENAME"]

num_queries_batch = 1
connector = PostgresConnector()
np.random.seed(123)
with open(".query_pull_1000", 'rb') as f:
    query_pull = pickle.load(f)

    ENV_NAME = 'DatabaseIndexesEnv-v0'
    register(
        id=ENV_NAME,
        entry_point='dbenv:DatabaseIndexesEnv',
        kwargs={'n': COLUMNS_AMOUNT,
                'table_name': table_name,
                'query_pull': query_pull,
                'batch_size': BATCH_SIZE,
                'connector': PostgresConnector(),
                'k': 3,
                'max_episodes': 50}
    )

    # Get the environment and extract the number of actions.
    env = gym.make(ENV_NAME)
    env.seed(123)

    # Next, we build a very simple model.
    model = Sequential()
    model.add(Flatten(input_shape=(1, BATCH_SIZE + 1, COLUMNS_AMOUNT)))
    model.add(Dense((BATCH_SIZE + 1) * COLUMNS_AMOUNT))
    model.add(Activation('relu'))
    model.add(Dense((BATCH_SIZE + 1) * COLUMNS_AMOUNT))
    model.add(Activation('relu'))
    model.add(Dense(COLUMNS_AMOUNT))
    model.add(Activation('linear'))
    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=1000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=COLUMNS_AMOUNT, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    dqn.fit(env, nb_steps=1000, visualize=True, verbose=1)

    # After training is done, we save the final weights.
    dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=5, visualize=False)
