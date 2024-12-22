import torch
import sys
import math


class CollaborativeFiltering:

    def __init__(self, strategy, data_matrix, time_matrix, user_index):
        self.strategy = strategy
        print("Collaborative Filtering Technique Chosen: ", strategy)
        print()
        self.results = self.strategy.run(data_matrix, time_matrix, user_index)

