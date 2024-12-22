import torch
import sys
import math


class CollaborativeFiltering:

    def __init__(self, strategy, data_matrix, time_matrix, user_index):
        self.strategy = strategy
        self.results = self.strategy.run(data_matrix, time_matrix, user_index)

