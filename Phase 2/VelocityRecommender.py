import pandas as pd
import numpy as np
import torch
import sys
import gc
from CMEmbeddingCurrentGenerator import CMEmbeddingCurrentGenerator
from RIMEmbeddingCurrentGenerator import RIMEmbeddingCurrentGenerator
from CMEmbeddingHistoryGenerator import CMEmbeddingHistoryGenerator
from RIMEmbeddingHistoryGenerator import RIMEmbeddingHistoryGenerator
from MultiviewNetwork import MultiviewNetwork

# # # # # # # # # # # # # # # # # # # # # # # # #
# Module:  VELOCITY RECOMMENDER                 #
# Authors: Mukul Malik, Mukul Verma             #
# # # # # # # # # # # # # # # # # # # # # # # # #

# NOTE TO VIEWER: Some matrices are big in size (for example, when we prepare training input for the neural network) 
# and can consume a lot of memory. We have thus included a function called RAMStats() which can be called anytime to
# monitor how much total RAM current variables are consuming to identify bottlenecks. It surely helped us a lot during
# our work. This code can be optimized space-wise so your suggestions on the same would be highly appreciated!

class VelocityRecommender:
    def __init__(self):
        self.RAM = 0
        self.datasetLoaded = False
        self.engineInitialize = False
        
        self.UserCMCurrent = False
        self.ItemCMCurrent = False
        self.UserRIMCurrent = False
        self.ItemRIMCurrent = False
        
        self.UserCMHistory = False
        self.ItemCMHistory = False
        self.UserRIMHistory = False
        self.ItemRIMHistory = False
        
        self.EmbeddingsProcessed = False
        self.NeuralInitialize = False
        self.Trained = False
        
        print("Engine Started.")
        
    def RAMStats(self):
        print("RAM Consumed in MB: ", self.RAM/(1024**2))
        
    def load_dataset(self, dataset_train_path, dataset_test_path, dataset_name="NO NAME"):
        print("Loading dataset: ", dataset_name)
        
        r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
        ratings = pd.read_csv(dataset_train_path, sep = '\t', names = r_cols, encoding = 'latin-1')
        ratings_test = pd.read_csv(dataset_test_path, sep = '\t', names = r_cols, encoding = 'latin-1')
        
        n_users_train = 943
        n_items_train = 1682
        n_users_test = 943
        n_items_test = 1682
        
        self.data_matrix_train = np.zeros((n_users_train, n_items_train))
        self.data_matrix_test = np.zeros((n_users_test, n_items_test))

        for line in ratings.itertuples():
            self.data_matrix_train[line[1]-1, line[2]-1] = line[3]

        for line in ratings_test.itertuples():
            self.data_matrix_test[line[1]-1, line[2]-1] = line[3]
            
        del(ratings)
        del(ratings_test)
        gc.collect()
        
        self.datasetLoaded = True
            
        print("RAM Consumed by Dataset (MB): ", (self.data_matrix_train.nbytes + self.data_matrix_test.nbytes) / (1024 ** 2))
        self.RAM += (self.data_matrix_train.nbytes + self.data_matrix_test.nbytes)
        print("Dataset loaded: ", dataset_name)
        
    def findExecutionContext(self):
        if(self.datasetLoaded != True):
            return 1
        elif(self.engineInitialize != True):
            return 2
        elif(self.UserCMCurrent != True or self.ItemCMCurrent != True or self.UserRIMCurrent != True or self.ItemRIMCurrent != True):
            return 3
        elif(self.UserCMHistory != True or self.ItemCMHistory != True or self.UserRIMHistory != True or self.ItemRIMHistory != True):
            return 4
        elif(self.EmbeddingsProcessed != True):
            return 5
        elif(self.NeuralInitialize != True):
            return 6
        elif(self.Trained != True):
            return 7
            
        return 8

        
    def showExecutionContext(self):
        print("Invoking Sequence of Functions:                                                 and execution status", )
        print("1. load_dataset(dataset_train_path, dataset_test_path, dataset_name):           ", self.datasetLoaded)
        print("2. initialize_engine():                                                         ", self.engineInitialize)
        print("3. generate_CM_Current_Embeddings_Users(embedding_dim, lr, epochs):             ", self.UserCMCurrent)
        print("3. generate_CM_Current_Embeddings_Items(embedding_dim, lr, epochs):             ", self.ItemCMCurrent)
        print("3. generate_RIM_Current_Embeddings_Users(embedding_dim, lr, epochs):            ", self.UserRIMCurrent)
        print("3. generate_RIM_Current_Embeddings_Items(embedding_dim, lr, epochs):            ", self.ItemRIMCurrent)
        print("4. generate_CM_History_Embeddings_Users():                                      ", self.UserCMHistory)
        print("4. generate_CM_History_Embeddings_Items():                                      ", self.ItemCMHistory)
        print("4. generate_RIM_History_Embeddings_Users():                                     ", self.UserRIMHistory)
        print("4. generate_RIM_History_Embeddings_Items():                                     ", self.ItemRIMHistory)
        print("5. process_embeddings():                                                        ", self.EmbeddingsProcessed)
        print("6. initialize_multiview_neural():                                               ", self.NeuralInitialize)
        print("7. train_MVNN():                                                                ", self.Trained)
        print("8. compute_RMSE(): ")
        
        print("---------------------------------------------")
        print("Functions with same index number can be called in any sequence but before invoking functions of a particular index, all functions of previous indexes must have been invoked otherwise the function call will fail. For example for invoking a function with index (3), all functions with index (1) and (2) must have been invoked! Additionally, following helper functions can be called anytime and aren't restricted in the invoking pipeline:")
        print("showExecutionContext()")
        print("RAMStats()")
        print("collectGarbage()")
        print("---------------------------------------------")
        
    def initialize_engine(self):
        if(self.findExecutionContext() < 2):
            print("[Execution Context Failure] Please invoke prior functions displayed by showExecutionContext(). Your current function index is ", self.findExecutionContext())
            return 
        
        print("Engine Initialized.")
        self.engineInitialize = True
        
    
    def generate_CM_Current_Embeddings_Users(self, embedding_dim = 400, lr = 0.05, epochs = 25000):
        if(self.findExecutionContext() < 3):
            print("[Execution Context Failure] Please invoke prior functions displayed by showExecutionContext(). Your current function index is ", self.findExecutionContext())
            return 
        generator = CMEmbeddingCurrentGenerator(embedding_dim, lr, epochs)
        self.CM_Current_Embeddings_Users = generator.generate_user_embeddings(self.data_matrix_train)
        
        del(generator)
        gc.collect()
        self.UserCMCurrent = True
        
        print()
        print("Additional RAM Consumed by Embeddings (MB): ", (self.CM_Current_Embeddings_Users.numel() * self.CM_Current_Embeddings_Users.element_size()) / (1024 ** 2))
        self.RAM += (self.CM_Current_Embeddings_Users.numel() * self.CM_Current_Embeddings_Users.element_size())
        
    def generate_CM_Current_Embeddings_Items(self, embedding_dim = 400, lr = 0.05, epochs = 25000):
        if(self.findExecutionContext() < 3):
            print("[Execution Context Failure] Please invoke prior functions displayed by showExecutionContext(). Your current function index is ", self.findExecutionContext())
            return
        generator = CMEmbeddingCurrentGenerator(embedding_dim, lr, epochs)
        self.CM_Current_Embeddings_Items = generator.generate_item_embeddings(self.data_matrix_train)
        
        del(generator)
        gc.collect()
        self.ItemCMCurrent = True
        
        print()
        print("Additional RAM Consumed by Embeddings (MB): ", (self.CM_Current_Embeddings_Items.numel() * self.CM_Current_Embeddings_Items.element_size()) / (1024 ** 2))
        self.RAM += (self.CM_Current_Embeddings_Items.numel() * self.CM_Current_Embeddings_Items.element_size())
        
    def generate_RIM_Current_Embeddings_Users(self, embedding_dim = 400, lr = 0.05, epochs = 25000):
        if(self.findExecutionContext() < 3):
            print("[Execution Context Failure] Please invoke prior functions displayed by showExecutionContext(). Your current function index is ", self.findExecutionContext())
            return
        generator = RIMEmbeddingCurrentGenerator(embedding_dim, lr, epochs)
        self.RIM_Current_Embeddings_Users = generator.generate_embeddings_users(self.data_matrix_train)
        
        del(generator)
        gc.collect()
        self.UserRIMCurrent = True
        
        print()
        print("Additional RAM Consumed by Embeddings (MB): ", (self.RIM_Current_Embeddings_Users.numel() * self.RIM_Current_Embeddings_Users.element_size()) / (1024 ** 2))
        self.RAM += (self.RIM_Current_Embeddings_Users.numel() * self.RIM_Current_Embeddings_Users.element_size())
        
    def generate_RIM_Current_Embeddings_Items(self, embedding_dim = 400, lr = 0.05, epochs = 25000):
        if(self.findExecutionContext() < 3):
            print("[Execution Context Failure] Please invoke prior functions displayed by showExecutionContext(). Your current function index is ", self.findExecutionContext())
            return
        
        generator = RIMEmbeddingCurrentGenerator(embedding_dim, lr, epochs)
        self.RIM_Current_Embeddings_Items = generator.generate_embeddings_items(self.data_matrix_train)
        
        del(generator)
        gc.collect()
        self.ItemRIMCurrent = True
        
        print()
        print("Additional RAM Consumed by Embeddings (MB): ", (self.RIM_Current_Embeddings_Items.numel() * self.RIM_Current_Embeddings_Items.element_size()) / (1024 ** 2))
        self.RAM += (self.RIM_Current_Embeddings_Items.numel() * self.RIM_Current_Embeddings_Items.element_size())
        
    def generate_CM_History_Embeddings_Users(self):
        if(self.findExecutionContext() < 4):
            print("[Execution Context Failure] Please invoke prior functions displayed by showExecutionContext(). Your current function index is ", self.findExecutionContext())
            return
        
        generator = CMEmbeddingHistoryGenerator()
        self.CM_History_Embeddings_Users = generator.compute_historical_user_embeddings(self.data_matrix_train, self.CM_Current_Embeddings_Items)
        
        del(generator)
        gc.collect()
        self.UserCMHistory = True
        
        print()
        print("Additional RAM Consumed by Embeddings (MB): ", (self.CM_Current_Embeddings_Users.numel() * self.CM_Current_Embeddings_Users.element_size()) / (1024 ** 2))
        self.RAM += (self.CM_Current_Embeddings_Users.numel() * self.CM_Current_Embeddings_Users.element_size())
        
    def generate_CM_History_Embeddings_Items(self):
        if(self.findExecutionContext() < 4):
            print("[Execution Context Failure] Please invoke prior functions displayed by showExecutionContext(). Your current function index is ", self.findExecutionContext())
            return
        
        generator = CMEmbeddingHistoryGenerator()
        self.CM_History_Embeddings_Items = generator.compute_historical_item_embeddings(self.data_matrix_train, self.CM_Current_Embeddings_Users)
        
        del(generator)
        gc.collect()
        self.ItemCMHistory = True
        
        print()
        print("Additional RAM Consumed by Embeddings (MB): ", (self.CM_History_Embeddings_Items.numel() * self.CM_History_Embeddings_Items.element_size()) / (1024 ** 2))
        self.RAM += (self.CM_History_Embeddings_Items.numel() * self.CM_History_Embeddings_Items.element_size())
        
    def generate_RIM_History_Embeddings_Users(self):
        if(self.findExecutionContext() < 4):
            print("[Execution Context Failure] Please invoke prior functions displayed by showExecutionContext(). Your current function index is ", self.findExecutionContext())
            return
        
        generator = RIMEmbeddingHistoryGenerator(self.data_matrix_train, self.RIM_Current_Embeddings_Items, self.RIM_Current_Embeddings_Users)
        self.RIM_History_Embeddings_Users = generator.compute_historical_user_embeddings()
        
        del(generator)
        gc.collect()
        self.UserRIMHistory = True
        
        print()
        print("Additional RAM Consumed by Embeddings (MB): ", (self.RIM_History_Embeddings_Users.numel() * self.RIM_History_Embeddings_Users.element_size()) / (1024 ** 2))
        self.RAM += (self.RIM_History_Embeddings_Users.numel() * self.RIM_History_Embeddings_Users.element_size())
        
    def generate_RIM_History_Embeddings_Items(self):
        if(self.findExecutionContext() < 4):
            print("[Execution Context Failure] Please invoke prior functions displayed by showExecutionContext(). Your current function index is ", self.findExecutionContext())
            return
        
        generator = RIMEmbeddingHistoryGenerator(self.data_matrix_train, self.RIM_Current_Embeddings_Items, self.RIM_Current_Embeddings_Users)
        self.RIM_History_Embeddings_Items = generator.compute_historical_item_embeddings()
        
        del(generator)
        gc.collect()
        self.ItemRIMHistory = True
        
        print()
        print("Additional RAM Consumed by Embeddings (MB): ", (self.RIM_History_Embeddings_Items.numel() * self.RIM_History_Embeddings_Items.element_size()) / (1024 ** 2))
        self.RAM += (self.RIM_History_Embeddings_Items.numel() * self.RIM_History_Embeddings_Items.element_size())
                  
    # ------------------------------------------------------------------------------------------
        
    def process_embeddings(self):
        if(self.findExecutionContext() < 5):
            print("[Execution Context Failure] Please invoke prior functions displayed by showExecutionContext(). Your current function index is ", self.findExecutionContext())
            return
        
        input_rows = []  # To store concatenated input rows
        target_values = []  # To store target ratings
        rating_levels = 5

        for user_idx in range(self.data_matrix_train.shape[0]):
            for item_idx in range(self.data_matrix_train.shape[1]):

                rating = self.data_matrix_train[user_idx, item_idx]

                if rating > 0:

                    # 1. User and item embeddings
                    user_embed = self.CM_Current_Embeddings_Users[user_idx].unsqueeze(0)  # (1x800)
                    item_embed = self.CM_Current_Embeddings_Items[item_idx].unsqueeze(0)  # (1x800)

                    # 2. Item embeddings for each rating level (I_E_RIM)
                    item_rim_embeds = torch.stack([
                        self.RIM_Current_Embeddings_Items[item_idx * rating_levels + (k - 1)]
                        for k in range(1, rating_levels + 1)
                    ], dim=0)  # (5x800)

                    # 3. User embeddings for each rating level (U_E_RIM)
                    user_rim_embeds = torch.stack([
                        self.RIM_Current_Embeddings_Users[user_idx * rating_levels + (k - 1)]
                        for k in range(1, rating_levels + 1)
                    ], dim=0)
                    # 5. Historical items for each rating level (history_items)
                    history_item_rim_embeds = torch.stack([
                        self.RIM_History_Embeddings_Items[item_idx * rating_levels + (k - 1)]
                        for k in range(1, rating_levels + 1)
                    ], dim=0)  # (5x800)

                    # 6. Historical users for each rating level (history_users)
                    history_user_rim_embeds = torch.stack([
                        self.RIM_History_Embeddings_Users[user_idx * rating_levels + (k - 1)]
                        for k in range(1, rating_levels + 1)
                    ], dim=0)  # (5x800)
                    
                    # 4. Historical embeddings
                    history_item_embed = self.CM_History_Embeddings_Items[item_idx].unsqueeze(0)  # (1x800)
                    history_user_embed = self.CM_History_Embeddings_Users[user_idx].unsqueeze(0)  # (1x800)

                    # Stack all components into a 3D tensor of shape (24x800)
                    input_row = torch.cat([
                        user_embed,               # (1x800)
                        item_embed,               # (1x800)
                        item_rim_embeds,          # (5x800)
                        user_rim_embeds,          # (5x800)
                        history_item_embed,       # (1x800)
                        history_user_embed,       # (1x800)
                        history_item_rim_embeds,  # (5x800)
                        history_user_rim_embeds   # (5x800)
                    ], dim=0)  # Shape (24x800)=0)  # (5x800)


                    # Append to input rows
                    input_rows.append(input_row.unsqueeze(0))  # Add batch dimension, making it (1x24x800)

                    # Append to target values
                    target_values.append(torch.tensor([rating], dtype=torch.float32))  # (1x1)

        # Stack all rows into tensors
        self.input_tensor = torch.cat(input_rows, dim=0)  # Shape: (No_Of_Training_Examples)x19200
        self.target_tensor = torch.cat(target_values, dim=0)  # Shape: (No_Of_Training_Examples)x1

        print(  self.target_tensor.shape )
        print(  self.target_tensor )
        
        print()
        print("Additional RAM Allocated (MB): ", ((self.input_tensor.numel() * self.input_tensor.element_size()) + (self.target_tensor.numel() * self.target_tensor.element_size()) / (1024 ** 2)))
        self.RAM += (self.input_tensor.numel() * self.input_tensor.element_size()) + (self.target_tensor.numel() * self.target_tensor.element_size())
        
        self.EmbeddingsProcessed = True
        
    def get_embeddings(self, user, movie):
        pass
        
        
    def initialize_multiview_neural(self, embedding_dim, FCNeurons_current, CONVNeurons_current, FCNeurons_history, CONVNeurons_history, num_final_hidden, num_ratings, lr, eps):
        if(self.findExecutionContext() < 6):
            print("[Execution Context Failure] Please invoke prior functions displayed by showExecutionContext(). Your current function index is ", self.findExecutionContext())
            return
        
        self.network = MultiviewNetwork(embedding_dim, FCNeurons_current, CONVNeurons_current, FCNeurons_history, CONVNeurons_history, num_final_hidden, num_ratings, lr, eps)
        self.NeuralInitialize = True
        
    def train_MVNN(self, ):
        if(self.findExecutionContext() < 7):
            print("[Execution Context Failure] Please invoke prior functions displayed by showExecutionContext(). Your current function index is ", self.findExecutionContext())
            return
        
        self.network.train_model(self.input_tensor, self.target_tensor, 15)
        self.Trained = True
        
    def compute_RMSE(self, ):
        pass
    
    def setVariables(self, data_matrix_train, data_matrix_test, CM_Current_Embeddings_Users, CM_Current_Embeddings_Items, RIM_Current_Embeddings_Users, RIM_Current_Embeddings_Items , CM_History_Embeddings_Users, CM_History_Embeddings_Items, RIM_History_Embeddings_Users, RIM_History_Embeddings_Items):
        self.data_matrix_train = data_matrix_train
        self.data_matrix_test = data_matrix_test
        self.CM_Current_Embeddings_Users = CM_Current_Embeddings_Users
        self.CM_Current_Embeddings_Items = CM_Current_Embeddings_Items
        self.RIM_Current_Embeddings_Users = RIM_Current_Embeddings_Users
        self.RIM_Current_Embeddings_Items = RIM_Current_Embeddings_Items
        self.CM_History_Embeddings_Users = CM_History_Embeddings_Users
        self.CM_History_Embeddings_Items = CM_History_Embeddings_Items
        self.RIM_History_Embeddings_Users = RIM_History_Embeddings_Users
        self.RIM_History_Embeddings_Items = RIM_History_Embeddings_Items
        
        self.datasetLoaded = True
        self.engineInitialize = True
        
        self.UserCMCurrent = True
        self.ItemCMCurrent = True
        self.UserRIMCurrent = True
        self.ItemRIMCurrent = True
        
        self.UserCMHistory = True
        self.ItemCMHistory = True
        self.UserRIMHistory = True
        self.ItemRIMHistory = True
        
        self.EmbeddingsProcessed = True
    
    def getVariables(self, ):
        return self.data_matrix_train, self.data_matrix_test, self.CM_Current_Embeddings_Users, self.CM_Current_Embeddings_Items, self.RIM_Current_Embeddings_Users, self.RIM_Current_Embeddings_Items , self.CM_History_Embeddings_Users, self.CM_History_Embeddings_Items, self.RIM_History_Embeddings_Users, self.RIM_History_Embeddings_Items
        
    # data_matrix_train, data_matrix_test, CM_Current_Embeddings_Users, CM_Current_Embeddings_Items, RIM_Current_Embeddings_Users, RIM_Current_Embeddings_Items , CM_History_Embeddings_Users, CM_History_Embeddings_Items, RIM_History_Embeddings_Users, RIM_History_Embeddings_Items
    def collectGarbage(self):
        gc.collect()