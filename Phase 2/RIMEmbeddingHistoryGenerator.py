import torch

# # # # # # # # # # # # # # # # # # # # # # # # #
# Module: RIM EMBEDDING HISTORY VIEW GENERATOR  #
# Authors: Mukul Malik, Mukul Verma             #
# # # # # # # # # # # # # # # # # # # # # # # # #

class RIMEmbeddingHistoryGenerator:
    def __init__(self, ratings_matrix, item_embeddings, user_embeddings, rating_levels=5):
        """
        Initialize the RIMEmbeddingGenerator with a ratings matrix and embeddings.

        Args:
            ratings_matrix (torch.Tensor): User-item ratings matrix (n_users x n_items).
            embeddings (torch.Tensor): Initial embeddings matrix (n_entities x embedding_dim), 
                                       can be for users or items.
            rating_levels (int): Number of rating levels, e.g., 5 for a 1-5 rating system.
        """
        self.ratings_matrix = torch.tensor(ratings_matrix, dtype=torch.float32)
        self.item_embeddings = item_embeddings #torch.tensor(item_embeddings, dtype=torch.float32)
        self.user_embeddings = user_embeddings #torch.tensor(user_embeddings, dtype=torch.float32)
        self.rating_levels = rating_levels
        self.users = self.ratings_matrix.shape[0]
        self.items = self.ratings_matrix.shape[1]
        self.embedding_dim = self.item_embeddings.shape[1]
    
    def compute_historical_item_embeddings(self):
        """
        Compute historical RIM embeddings for each item based on the rating levels.
        
        Returns:
            torch.Tensor: Historical RIM item embeddings for each rating level 
                          (rating_levels x n_items x embedding_dim).
        """
        n_items = self.ratings_matrix.shape[1]
        historical_item_embeddings = torch.zeros((self.rating_levels * n_items, self.embedding_dim), dtype=torch.float32)
        
        for item_idx in range(n_items):
            for k in range(1, self.rating_levels + 1):

                rated_users = [user_idx for user_idx in range(self.ratings_matrix.shape[0]) 
                               if self.ratings_matrix[user_idx, item_idx] == k]
                
                if rated_users:
                    sum_embeddings = torch.zeros(self.embedding_dim, dtype=torch.float32)
                    for user_idx in rated_users:
                        sum_embeddings += self.user_embeddings[user_idx*5 + (k-1)]
                    historical_item_embeddings[item_idx*5 + (k-1)] = sum_embeddings / len(rated_users)
                else:
                    historical_item_embeddings[item_idx*5 + (k-1)] = torch.zeros(self.embedding_dim)

        return historical_item_embeddings
    
    def compute_historical_user_embeddings(self):
        """
        Compute historical RIM embeddings for each item based on the rating levels.
        
        Returns:
            torch.Tensor: Historical RIM item embeddings for each rating level 
                          (rating_levels x n_items x embedding_dim).
        """
        n_users = self.ratings_matrix.shape[0]
        historical_user_embeddings = torch.zeros((self.rating_levels * n_users, self.embedding_dim), dtype=torch.float32)
        
        for user_idx in range(n_users):
            for k in range(1, self.rating_levels + 1):

                rated_items = [item_idx for item_idx in range(self.ratings_matrix.shape[1]) 
                               if self.ratings_matrix[user_idx, item_idx] == k]
                
                if rated_items:
                    sum_embeddings = torch.zeros(self.embedding_dim, dtype=torch.float32)
                    for item_idx in rated_items:
                        sum_embeddings += self.item_embeddings[item_idx*5 + (k-1)]
                    historical_user_embeddings[user_idx*5 + (k-1)] = sum_embeddings / len(rated_items)
                else:
                    historical_user_embeddings[user_idx*5 + (k-1)] = torch.zeros(self.embedding_dim)

        return historical_user_embeddings
