# stp-database-technologies

A team project from year 2018. The goal of the project is to use reinforcement learning to select candidate columns to assign indexes to a database table.

Initially the goal was to implement a deep q learning agent "model-based".
My roles was to implement a Q-learning agent "model-free agent". Furthermore, due to the long running time of the deep q learning agent I provided a competitor to the deep q learning agent by
storing Q-learning experiment as a training data and then using xgboost to build a model from this data.

As a result the training time for the Qlearning + Xgboost was siginficanlty shorter the deep q learning agent.
