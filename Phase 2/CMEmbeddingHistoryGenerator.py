import torch

# # # # # # # # # # # # # # # # # # # # # # # #
# Module: CM EMBEDDING HISTORY VIEW GENERATOR #
# Authors: Mukul Malik, Mukul Verma           #
# # # # # # # # # # # # # # # # # # # # # # # #

class CMEmbeddingHistoryGenerator:
    def __init__(self):
        """
        Initialize the Historical CM Embedding Generator.
        """
        pass

    def compute_historical_item_embeddings(self, ratings_matrix, user_embeddings):
        """
        Compute historical CM embeddings for each item based on the users who rated it.
        The formula for each item j:
            h(t_j) = sum(user_embeddings) / |users|
        """
        ratings_matrix = torch.tensor(ratings_matrix, dtype=torch.float32)
#         user_embeddings = torch.tensor(user_embeddings, dtype=torch.float32)

        n_items = ratings_matrix.shape[1]
        n_users = ratings_matrix.shape[0]
        embedding_dim = user_embeddings.shape[1]
        historical_item_embeddings = torch.zeros((n_items, embedding_dim), dtype=torch.float32)

        for item_idx in range(n_items):
            rated_users = []
            for user_idx in range(n_users):
                if ratings_matrix[user_idx, item_idx] > 0:
                    rated_users.append(user_idx)
            
            if rated_users:
                sum_embeddings = torch.zeros(embedding_dim, dtype=torch.float32)
                for user_idx in rated_users:
                    sum_embeddings += user_embeddings[user_idx]
                historical_item_embeddings[item_idx] = sum_embeddings / len(rated_users)
            else:
                historical_item_embeddings[item_idx] = torch.zeros(embedding_dim)

        return historical_item_embeddings

    def compute_historical_user_embeddings(self, ratings_matrix, item_embeddings):
        """
        Compute historical CM embeddings for each user based on other users
        who interacted with the same items.
        The formula for each user i:
            h(u_i) = sum(user_embeddings[j]) / |users|
        where user j has interacted with at least one item user i has interacted with.
        """
        ratings_matrix = torch.tensor(ratings_matrix, dtype=torch.float32)
#         item_embeddings = torch.tensor(item_embeddings, dtype=torch.float32)

        n_items = ratings_matrix.shape[1]
        n_users = ratings_matrix.shape[0]
        embedding_dim = item_embeddings.shape[1]
        historical_user_embeddings = torch.zeros((n_users, embedding_dim), dtype=torch.float32)
                
        for user_idx in range(n_users):
            rated_items = []
            for item_idx in range(n_items):
                if ratings_matrix[user_idx, item_idx] > 0:
                    rated_items.append(item_idx)
            
            if rated_items:
                sum_embeddings = torch.zeros(embedding_dim, dtype=torch.float32)
                for item_idx in rated_items:
                    sum_embeddings += item_embeddings[item_idx]
                historical_user_embeddings[user_idx] = sum_embeddings / len(rated_items)
            else:
                historical_user_embeddings[user_idx] = torch.zeros(embedding_dim)

        return historical_user_embeddings
