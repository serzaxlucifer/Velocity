import pandas as pd
import numpy as np
import torch
import sys
import gc
import VelocityRecommender from "./VelocityRecommender.py"

recommender = VelocityRecommender()
recommender.load_dataset("ml-100k/u2.base", "ml-100k/u2.test", "MOVIELENS-100K")
recommender.initialize_engine()

recommender.generate_CM_Current_Embeddings_Users(400, 0.05, 1000)
recommender.generate_CM_Current_Embeddings_Items(400, 0.05, 1000)
recommender.generate_RIM_Current_Embeddings_Users(400, 0.05, 1000)
recommender.generate_RIM_Current_Embeddings_Items(400, 0.05, 1000)
recommender.generate_CM_History_Embeddings_Users()
recommender.generate_CM_History_Embeddings_Items()
recommender.generate_RIM_History_Embeddings_Users()
recommender.generate_RIM_History_Embeddings_Items()

recommender.process_embeddings()
recommender.initialize_multiview_neural(800, 200, 200, 200, 200, 4000, 80000, 0.01, 15)
recommender.train_MVNN()
