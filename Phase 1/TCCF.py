import torch
import numpy as np
import sys
import math
from SimilarityMetrics import Pearson
from SimilarityMetrics import Cosine


class TCCF:       # Memory based Collaborative Filtering

    def __init__(self, similarity_metric=Pearson(1)):
        self.sim_metric = similarity_metric

    def run(self, data_matrix, time_matrix, user_index):
        self.user_index = user_index

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_matrix_tensor = torch.tensor(data_matrix, dtype=torch.float32, device=device)
        self.time_matrix_tensor = torch.tensor(time_matrix, dtype=torch.float32, device=device)
        self.max_time_per_user, _ = torch.max(self.time_matrix_tensor, dim=1, keepdim=True)

        prediction_of_user = torch.zeros(self.data_matrix_tensor.shape[1])
        print("Collaborative Filtering Engine Initialized.")

        self.user_sim_matrix = self.sim_metric.run(self.data_matrix_tensor, self.time_matrix_tensor, self.max_time_per_user, self.user_index)
        print("Similarity Computed!")

        for movie_index in range(self.data_matrix_tensor.shape[1]):
            if self.data_matrix_tensor[user_index][movie_index] == 0:

                predicted_rating = self.predict_missing_rating(movie_index)
                prediction_of_user[movie_index] = predicted_rating

        print("Collaborative Filtering Complete!")
        self.results = prediction_of_user.numpy()

        return prediction_of_user.numpy()

    def predict_missing_rating(self, item_index):

        # Extract the ratings and mean rating for the given user
        user_ratings = self.data_matrix_tensor[self.user_index]
        non_zero_ratings = user_ratings[user_ratings != 0]

        # Calculate the mean of non-zero ratings
        mean_rating_user = torch.mean(non_zero_ratings)

        # Initialize variables for the weighted sum and sum of similarities
        weighted_sum = 0.0
        sum_of_similarities = 0.0

        # Iterate over all users
        for i in range(self.data_matrix_tensor.shape[0]):

            if i != self.user_index and self.data_matrix_tensor[i][item_index] != 0:
                # Extract the rating and mean rating fo the current user
                original = self.data_matrix_tensor[i][item_index].clone()
                self.data_matrix_tensor[i][item_index] = 0

                ratings_other_user = self.data_matrix_tensor[i]
                otherNonZero = ratings_other_user[ratings_other_user != 0]
                mean_rating_other_user = torch.mean(otherNonZero)

                self.data_matrix_tensor[i][item_index] = original
                ratings_other_user = self.data_matrix_tensor[i]

                # Compute the difference between the rating and mean rating for the current user
                rating_diff = ratings_other_user - mean_rating_other_user
                rating_diff = rating_diff[item_index]

                # Compute the weighted contribution using similarity and rating difference
                weighted_contribution = self.user_sim_matrix[i] * rating_diff

                # Accumulate the weighted contribution and sum of similarities
                weighted_sum += weighted_contribution
                sum_of_similarities += self.user_sim_matrix[i]

        # Calculate the predicted rating
        predicted_rating = mean_rating_user + (weighted_sum / sum_of_similarities) if sum_of_similarities != 0 else mean_rating_user

        return predicted_rating.item()
    
class WithoutTCCF:       # Memory based Collaborative Filtering

    def __init__(self, similarity_metric=Cosine(1)):
        self.sim_metric = similarity_metric

    def run(self, data_matrix, time_matrix, user_index, data_matrix_test):
        self.user_index = user_index

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_matrix_tensor = torch.tensor(data_matrix, dtype=torch.float32, device=device)
        self.time_matrix_tensor = torch.tensor(time_matrix, dtype=torch.float32, device=device)
        self.max_time_per_user, _ = torch.max(self.time_matrix_tensor, dim=1, keepdim=True)

        prediction_of_user = torch.zeros(self.data_matrix_tensor.shape[1])
        print("Collaborative Filtering Engine Initialized.")

        # Now, look we will compute this somewhere else! (Moving Similarity Computation)
        self.user_sim_matrix = self.sim_metric.run(self.data_matrix_tensor, self.time_matrix_tensor, self.max_time_per_user, self.user_index)
        print("Similarity Computed!")

        for movie_index in range(self.data_matrix_tensor.shape[1]):
            if self.data_matrix_tensor[user_index][movie_index] == 0 and data_matrix_test[user_index][movie_index] != 0:

                # Compute prediction for missing rating
                predicted_rating = self.predict_missing_rating(movie_index)
                prediction_of_user[movie_index] = predicted_rating

        print("Collaborative Filtering Complete!")
        self.results = prediction_of_user.numpy()

        return prediction_of_user.numpy()

    def predict_missing_rating(self, item_index):

        # Extract the ratings and mean rating for the given user
        user_ratings = self.data_matrix_tensor[self.user_index]
        non_zero_ratings = user_ratings[user_ratings != 0]

        # Calculate the mean of non-zero ratings
        mean_rating_user = torch.mean(non_zero_ratings)

        # Initialize variables for the weighted sum and sum of similarities
        weighted_sum = 0.0
        sum_of_similarities = 0.0

        # Iterate over all users
        for i in range(self.data_matrix_tensor.shape[0]):

            if i != self.user_index and self.data_matrix_tensor[i][item_index] != 0:
                # Extract the rating and mean rating fo the current user
                original = self.data_matrix_tensor[i][item_index].clone()
                self.data_matrix_tensor[i][item_index] = 0

                ratings_other_user = self.data_matrix_tensor[i]
                otherNonZero = ratings_other_user[ratings_other_user != 0]
                mean_rating_other_user = torch.mean(otherNonZero)

                self.data_matrix_tensor[i][item_index] = original
                ratings_other_user = self.data_matrix_tensor[i]

                # Compute the difference between the rating and mean rating for the current user
                rating_diff = ratings_other_user - mean_rating_other_user
                rating_diff = rating_diff[item_index]

                # Compute the weighted contribution using similarity and rating difference
                weighted_contribution = self.user_sim_matrix[i] * rating_diff

                # Accumulate the weighted contribution and sum of similarities
                weighted_sum += weighted_contribution
                sum_of_similarities += self.user_sim_matrix[i]

        # Calculate the predicted rating
        predicted_rating = mean_rating_user + (weighted_sum / sum_of_similarities) if sum_of_similarities != 0 else mean_rating_user

        return predicted_rating.item()
