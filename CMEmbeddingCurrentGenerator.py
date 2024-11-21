import torch
import numpy as np
import pandas as pd
import sys
import gc

class CMEmbeddingCurrentGenerator:
    def __init__(self, embedding_dim=10, learning_rate=0.01, epochs=100):
        """
        Initialize parameters for CMEmbeddingGenerator.
        :param embedding_dim: Dimension of the embeddings.
        :param learning_rate: Learning rate for gradient descent.
        :param epochs: Number of training epochs.
        """
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.epochs = epochs


    def generate_user_embeddings(self, ratings_matrix):
        """
        Generate CM embeddings for users.
        :param ratings_matrix: User-item ratings matrix.
        :return: Final user embeddings.
        """
        cooccurrence_matrix = self._calculate_occurrence_values_users(ratings_matrix)
        left_embeddings, right_embeddings, biases = self._initialize_embeddings(
            ratings_matrix.shape[0], self.embedding_dim)
        left_embeddings, right_embeddings = self._optimize(
            cooccurrence_matrix, left_embeddings, right_embeddings, self.learning_rate, self.epochs, "Generate CM Current User Embeddings", biases)
        
        del(cooccurrence_matrix)
        del(biases)
        gc.collect()
        
        return self._concatenate_embeddings(left_embeddings, right_embeddings)

    def generate_item_embeddings(self, ratings_matrix):
        """
        Generate CM embeddings for items.
        :param ratings_matrix: User-item ratings matrix.
        :return: Final item embeddings.
        """
        cooccurrence_matrix = self._calculate_occurrence_values_items(ratings_matrix)
        left_embeddings, right_embeddings, biases = self._initialize_embeddings(
            ratings_matrix.shape[1], self.embedding_dim)
        
        left_embeddings, right_embeddings = self._optimize(
            cooccurrence_matrix, left_embeddings, right_embeddings, self.learning_rate, self.epochs, "Generate CM Current Item Embeddings", biases)
        
        del(cooccurrence_matrix)
        del(biases)
        gc.collect()
        
        return self._concatenate_embeddings(left_embeddings, right_embeddings)

    def _calculate_occurrence_values_items(self, ratings_matrix):
        """
        Calculate item-item co-occurrence matrix based on sets of users who gave the same rating.
        :param ratings_matrix: User-item ratings matrix.
        :return: Item-item co-occurrence matrix.
        """
        n = ratings_matrix.shape[1]  # Number of items
        cooccurrence_matrix = np.zeros((n, n), dtype=int)

        for i in range(n):
            for j in range(i + 1, n):
                common_users = np.where((ratings_matrix[:, i] > 0) & (ratings_matrix[:, j] > 0))[0]

                if len(common_users) > 0:
                    # Count items where both users gave the same rating
                    same_rating_count = np.sum(ratings_matrix[common_users, i] == ratings_matrix[common_users, j])

                    # Store the count in the co-occurrence matrix
                    cooccurrence_matrix[i, j] = same_rating_count
                    cooccurrence_matrix[j, i] = same_rating_count  # Symmetric matrix

        return cooccurrence_matrix

    def _calculate_occurrence_values_users(self, ratings_matrix):
        """
        Calculate user-user co-occurrence matrix based on items rated with the same rating.
        :param ratings_matrix: User-item ratings matrix (rows: users, columns: items).
        :return: User-user co-occurrence matrix.
        """
        num_users = ratings_matrix.shape[0]
        cooccurrence_matrix = np.zeros((num_users, num_users), dtype=int)

        for i in range(num_users):
            for j in range(i + 1, num_users):
                # Find items rated by both users
                common_items = np.where((ratings_matrix[i, :] > 0) & (ratings_matrix[j, :] > 0))[0]

                if len(common_items) > 0:
                    # Count items where both users gave the same rating
                    same_rating_count = np.sum(ratings_matrix[i, common_items] == ratings_matrix[j, common_items])

                    # Store the count in the co-occurrence matrix
                    cooccurrence_matrix[i, j] = same_rating_count
                    cooccurrence_matrix[j, i] = same_rating_count  # Symmetric matrix

        return cooccurrence_matrix

    def _initialize_embeddings(self, n, embedding_dim):
        """
        Initialize embeddings and biases randomly.
        :param n: Number of entities (users or items).
        :param embedding_dim: Dimension of the embeddings.
        :return: Initialized left embeddings, right embeddings, and biases.
        """
        left_embeddings = np.random.normal(scale = 0.1, size=(n, embedding_dim))
        right_embeddings = np.random.normal(scale = 0.1, size=(n, embedding_dim))
        biases = np.random.normal(scale = 0.1, size=n)  # Biases for items
        return left_embeddings, right_embeddings, biases


    def _optimize(self, cooccurrence_matrix, left_embeddings, right_embeddings, lr, epochs, task, biases={}):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        CM_tensor = torch.tensor(cooccurrence_matrix, dtype=torch.float32, device=device)
        P_tensor = torch.tensor(left_embeddings, dtype=torch.float32, device=device, requires_grad=True)
        Q_tensor = torch.tensor(right_embeddings.T, dtype=torch.float32, device=device, requires_grad=True)  # Transpose Q for proper matrix multiplication
        bias_tensor = torch.tensor(biases, dtype=torch.float32, device=device, requires_grad=True)
        
        n = CM_tensor.shape[0]
        optimizer = torch.optim.Adagrad([P_tensor, Q_tensor, bias_tensor], lr=lr)
        
        # Is this the right place for the log.
        CM_tensor = torch.log(torch.clamp(CM_tensor, min=1e-10))

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

              sys.stdout.write("\r({}) Step {}, Loss: {:.4f}".format(task, epoch, loss.item()))

        return P_tensor.detach().cpu(), Q_tensor.T.detach().cpu()


    def _concatenate_embeddings(self, left_embeddings, right_embeddings):
        """
        Concatenate left and right embeddings for the final representation.
        :param left_embeddings: Left embeddings.
        :param right_embeddings: Right embeddings.
        :return: Final concatenated embeddings.
        """
        return torch.cat((left_embeddings, right_embeddings), dim=1)

