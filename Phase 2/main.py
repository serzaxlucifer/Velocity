import pandas as pd
import numpy as np
import torch
import sys
import gc
from VelocityRecommender import VelocityRecommender

# # # # # # # # # # # # # # # # # # # # # # # # #
# Module:  MAIN PHASE 2 MODEL DRIVER            #
# Authors: Mukul Malik, Mukul Verma             #
# # # # # # # # # # # # # # # # # # # # # # # # #

# NOTE TO VIEWER: Some matrices are big in size (for example, when we prepare training input for the neural network) 
# and can consume a lot of memory. We have thus included a function called RAMStats() which can be called anytime to
# monitor how much total RAM current variables are consuming to identify bottlenecks. It surely helped us a lot during
# our work. This code can be optimized space-wise so your suggestions on the same would be highly appreciated!
# Calling each function also prints the additional memory consumed by the function after its execution ends.

# To simplify invocation, we have included the function showExecutionContext() which prints information about the 
# functions available and suggests the order in which functions should be called. It also blocks out-of-order function calls
# by printing Execution Context Error.

recommender = VelocityRecommender()
recommender.load_dataset("ml-100k/u2.base", "ml-100k/u2.test", "MOVIELENS-100K")
recommender.initialize_engine()
recommender.showExecutionContext()
recommender.RAMStats()

recommender.generate_CM_Current_Embeddings_Users(100, 0.05, 5000)
recommender.generate_CM_Current_Embeddings_Items(100, 0.05, 5000)
recommender.generate_RIM_Current_Embeddings_Users(100, 0.05, 5000)
recommender.generate_RIM_Current_Embeddings_Items(100, 0.05, 5000)
recommender.generate_CM_History_Embeddings_Users()
recommender.generate_CM_History_Embeddings_Items()
recommender.generate_RIM_History_Embeddings_Users()
recommender.generate_RIM_History_Embeddings_Items()

recommender.process_embeddings()
recommender.initialize_multiview_neural(200, 100, 100, 100, 100, 1000, 80000, 0.1, 15)
recommender.train_MVNN()
