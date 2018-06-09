import os
import pickle

import gym
import numpy as np
from gym.envs import register
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy

from PostgresConnector import PostgresConnector
from const import COLUMNS_AMOUNT, BATCH_SIZE

table_name = os.environ["TABLENAME"]
ENV_NAME = 'DatabaseIndexesEnv-v0'

num_queries_batch = 1
episodes = 1000


def train_model():
    np.random.seed(123)
    with open("query_pull_1000v2.pkl", 'rb') as f:
        query_pull = pickle.load(f)[0:5]

        register(
            id=ENV_NAME,
            entry_point='dbenv:DatabaseIndexesEnv',
            kwargs={'n': COLUMNS_AMOUNT,
                    'table_name': table_name,
                    'query_pull': query_pull,
                    'batch_size': BATCH_SIZE,
                    'connector': PostgresConnector(),
                    'k': 3,
                    'max_episodes': episodes}
        )

        # Get the environment and extract the number of actions.
        env = gym.make(ENV_NAME)
        env.seed(123)

        # Next, we build a very simple model.
        model = build_model()
        print(model.summary())

        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        dqn = initialize_agent(model)

        # Okay, now it's time to learn something! We visualize the training here for show, but this
        # slows down training quite a lot. You can always safely abort the training prematurely using
        # Ctrl + C.
        dqn.fit(env, nb_steps=episodes, visualize=False, verbose=2)

        # After training is done, we save the final weights.
        dqn.save_weights('dqn_specific_{}.h5f'.format(ENV_NAME), overwrite=True)

        # Finally, evaluate our algorithm for 5 episodes.
        dqn.test(env, nb_episodes=5, visualize=False)


def build_model():
    model = Sequential()
    model.add(Flatten(input_shape=(1, COLUMNS_AMOUNT)))
    model.add(Dense(COLUMNS_AMOUNT))
    model.add(Activation('relu'))
    model.add(Dense(COLUMNS_AMOUNT))
    model.add(Activation('relu'))
    model.add(Dense(COLUMNS_AMOUNT))
    model.add(Activation('linear'))
    return model


def initialize_agent(model):
    memory = SequentialMemory(limit=100, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model,
                   nb_actions=COLUMNS_AMOUNT,
                   memory=memory,
                   nb_steps_warmup=100,
                   target_model_update=1e-2,
                   policy=policy,
                   processor=ReducingProcessor(),
                   )
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    return dqn


class ReducingProcessor(Processor):
    def process_observation(self, observation):
        return observation[0]


def load_agent(file_path):
    model = build_model()
    dqn = initialize_agent(model)
    dqn.load_weights(file_path)
    return dqn


if __name__ == "__main__":
    train_model()
