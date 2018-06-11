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

num_queries_batch = 5

COLUMNS_AMOUNT = 17


def get_indexes_supervised(index_amount, queries):
    current_query_idx = 0
    query_batch = list()
    query_batch = list()
    workload_selectivity_l = list()
    for i in range(current_query_idx, current_query_idx + num_queries_batch):
        query_batch.append(queries[i]['query'])
        workload_selectivity_l.append(list(map(lambda x: x, queries[i]['sf_array'])))
    workload_selectivity = np.prod(workload_selectivity_l, axis=0).tolist()
    x = np.array(workload_selectivity)
    x = x.reshape(1, 17)
    classifier = None
    with open("../supervised/classifier", 'rb') as f:
        classifier = pickle.load(f)
    predicted_probabilities = classifier.predict_proba(x)[0]
    #    print predicted_probabilities
    return heapq.nlargest(index_amount, range(len(predicted_probabilities)), predicted_probabilities.__getitem__)


def build_xgboost_model(test_size=0.33):
    dataset = loadtxt(os.environ["GENERATED_DATA"], delimiter=",")
    X = dataset[:, 0:COLUMNS_AMOUNT]
    Y = dataset[:, COLUMNS_AMOUNT:]
    seed = 7

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # fit model no training data
    model = XGBClassifier()
    classif = OneVsRestClassifier(model)
    fited_model = classif.fit(X_train, y_train)
    y_pred = classif.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_sr_micro = f1_score(y_test, y_pred, average="micro")
    ham_loss = hamming_loss(y_test, y_pred)
    print("f1 score with micro average: %.2f" % f1_sr_micro)
    print("hamming loss: %.2f" % ham_loss)
    print("Accuracy: %.2f" % (accuracy))
    with open("../supervised/classifier2", 'wb') as f:
        pickle.dump(classif, f)

    return {'classifier': classif, 'accuracy': accuracy}
