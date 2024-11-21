import torch
import numpy as np
import pandas as pd
import sys
import gc

class RIMEmbeddingCurrentGenerator:
    
    def __init__(self, embedding_dim=3, learning_rate=0.01, epochs=100):
        """
        Initialize the RIM Embedding Generator.
        
        :param embedding_dim: Dimensionality of the embeddings (default 3).
        :param learning_rate: Learning rate for gradient descent (default 0.01).
        :param epochs: Number of training epochs (default 100).
        """
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def generate_user_sets_by_rating(self, ratings_matrix, rating_levels):
        """
        Generate a dictionary of user sets for each item-rating pair.
        Each key is a tuple (item_idx, rating), and the value is a set of users who rated the item with that rating.
        """
        user_sets = {}
        n_items = ratings_matrix.shape[1]
        
        for item_idx in range(n_items):
            for rating in rating_levels:
                user_set_key = (item_idx, rating)
                user_sets[user_set_key] = set(np.where(ratings_matrix[:, item_idx] == rating)[0])
        return user_sets
    
    def generate_embeddings_items(self, ratings_matrix):
        """
        Generate RIM embeddings for both users and items.
        
        :param ratings_matrix: User-item ratings matrix.
        :return: Final embeddings for users and items.
        """
        # Generate co-occurrence matrices for users and items
        user_sets = self.generate_user_sets_by_rating(ratings_matrix, rating_levels=[1, 2, 3, 4, 5])  # Rating levels 1 to 5
        
        cooccurrence_matrix_users = self.calculate_occurrence_matrix(user_sets)
        
        # Initialize embeddings and biases
        left_embeddings_items, right_embeddings_items, item_biases = self.initialize_embeddings(ratings_matrix.shape[1], 5)
        
        # Perform gradient descent to learn embeddings
        left_embeddings_items, right_embeddings_items = self._optimize(
            cooccurrence_matrix_users, left_embeddings_items, right_embeddings_items, self.learning_rate, self.epochs, "Generate RIM Current Item Embeddings", item_biases
        )
        
        del(cooccurrence_matrix_users)
        del(item_biases)
        gc.collect()
        
        return self._concatenate_embeddings(left_embeddings_items, right_embeddings_items)
    
    def generate_embeddings_users(self, ratings_matrix):
        """
        Generate RIM embeddings for both users and items.
        
        :param ratings_matrix: User-item ratings matrix.
        :return: Final embeddings for users and items.
        """
        # Generate co-occurrence matrices for users and items
        item_sets = self.generate_user_sets_by_rating(ratings_matrix.T, rating_levels=[1, 2, 3, 4, 5])
        
        cooccurrence_matrix_items = self.calculate_occurrence_matrix(item_sets)
        
        # Initialize embeddings and biases
        left_embeddings_users, right_embeddings_users, user_biases = self.initialize_embeddings(ratings_matrix.shape[0], 5)
        
        # Perform gradient descent to learn embeddings
        
        left_embeddings_users, right_embeddings_users = self._optimize(
            cooccurrence_matrix_items, left_embeddings_users, right_embeddings_users, self.learning_rate, self.epochs, "Generate RIM Current User Embeddings", user_biases
        )
        
        del(cooccurrence_matrix_items)
        del(user_biases)
        gc.collect()
        
        return self._concatenate_embeddings(left_embeddings_users, right_embeddings_users)
    
    def calculate_occurrence_matrix(self, user_sets):
        """
        Calculate the co-occurrence matrix for item-rating pairs based on user overlap.
        The matrix is symmetric with dimensions (n_item_rating_pairs x n_item_rating_pairs).
        """
        n_item_rating_pairs = len(user_sets)
        cooccurrence_matrix = np.zeros((n_item_rating_pairs, n_item_rating_pairs))
        print(cooccurrence_matrix.shape)
        
        item_rating_pairs = list(user_sets.keys())
        
        for idx_i, pair_i in enumerate(item_rating_pairs):
            for idx_j, pair_j in enumerate(item_rating_pairs):
                if idx_i != idx_j:
                    users_i = user_sets[pair_i]
                    users_j = user_sets[pair_j]
                    cooccurrence_matrix[idx_i, idx_j] = len(users_i.intersection(users_j))
        
        return cooccurrence_matrix

    def initialize_embeddings(self, num, num_ratings):
        """Initialize embeddings and biases for users and items."""
        left_embeddings = np.random.normal(scale=0.01, size=(num * num_ratings, self.embedding_dim)) 
        right_embeddings = np.random.normal(scale=0.01, size=(num * num_ratings, self.embedding_dim)) 
        item_biases = np.random.normal(scale=0.01, size=num * num_ratings)  # Biases for items
        return left_embeddings, right_embeddings, item_biases
    
    def _optimize(self, cooccurrence_matrix, left_embeddings, right_embeddings, lr, epochs, task, biases={}):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        CM_tensor = torch.tensor(cooccurrence_matrix, dtype=torch.float32, device=device)
        P_tensor = torch.tensor(left_embeddings, dtype=torch.float32, device=device, requires_grad=True)
        Q_tensor = torch.tensor(right_embeddings.T, dtype=torch.float32, device=device, requires_grad=True)  # Transpose Q for proper matrix multiplication
        bias_tensor = torch.tensor(biases, dtype=torch.float32, device=device, requires_grad=True)
        
        n = CM_tensor.shape[0]
        optimizer = torch.optim.Adagrad([P_tensor, Q_tensor, bias_tensor], lr=lr)
        

        for epoch in range(epochs):
            # Calculate predicted ratings
              pred_ratings = torch.mm(P_tensor, Q_tensor) + bias_tensor
              # Calculate error
              error = CM_tensor - pred_ratings
              # Compute loss with regularization
              loss = torch.sum(torch.square(error * (CM_tensor > 0))) #+ self.beta/2 * (torch.sum(torch.square(P_tensor)) + torch.sum(torch.square(Q_tensor)))

              optimizer.zero_grad()
              loss.backward()
              optimizer.step()

              sys.stdout.write("\r({})Step {}, Loss: {:.4f}".format(task, epoch, loss.item()))

        return P_tensor.detach().cpu(), Q_tensor.T.detach().cpu()


    def _concatenate_embeddings(self, left_embeddings, right_embeddings):
        """
        Concatenate left and right embeddings for the final representation.
        :param left_embeddings: Left embeddings.
        :param right_embeddings: Right embeddings.
        :return: Final concatenated embeddings.
        """

        return torch.cat((left_embeddings, right_embeddings), dim=1)



