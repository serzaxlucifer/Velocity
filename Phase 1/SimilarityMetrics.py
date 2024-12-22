import torch
import numpy
import math
import sys
from datetime import datetime

class Pearson:

    def __init__(self, sigma=1):
        self.sigma = sigma

    def time_correlation_coefficient(self, delta_T, sigma):

        term1 = 1 / (sigma * torch.sqrt(2 * torch.tensor(math.pi)))
        term2 = 1 - torch.exp(- (delta_T**2) / (2 * sigma**2))
        tcc = 1 - (term1 * term2)

        return tcc

    def get_month_difference(self, timestamp1, timestamp2):

        date2 = datetime.utcfromtimestamp(timestamp2.item())

        dates1 = [datetime.utcfromtimestamp(ts.item()) for ts in timestamp1]
        month_differences = [abs((date2.year - date1.year) * 12 + (date2.month - date1.month)) for date1 in dates1]

        return torch.tensor(month_differences, dtype=torch.float32)

    def run(self, data_matrix, time_matrix, timestamp_matrix, user_index):
        # VERMA, SUJAL: You'll find the code in BTP.ipynb (but its for numpy input, convert into torch and fill this.)

        num_users = data_matrix.shape[0]
        self.user_sim_matrix = torch.zeros(num_users)

        # UPDATED CODE FOR PEARSON: Map user_index's items to TCC(i) (used globally)

        time_stamps = time_matrix[user_index]
        max_timestamp = timestamp_matrix[user_index]

        self.time_correlation_coefficients = self.time_correlation_coefficient(self.get_month_difference(time_stamps, max_timestamp), self.sigma).to('cuda:0')

        for i in range(num_users):
            if i != user_index:  # Exclude the user itself

                # Find indices where both users have non-zero ratings
                non_zero_indices = (data_matrix[user_index] != 0) & (data_matrix[i] != 0)

                if len(non_zero_indices) == 0:
                    self.user_sim_matrix[i] = 0
                    continue

                # Filter ratings and timestamps to keep only non-zero elements
                ratings_user = data_matrix[user_index].clone()
                ratings_other_user = data_matrix[i].clone()

                # Adjust ratings with time correlation coefficients
                ratings_user = ratings_user.to('cuda:0')
                ratings_other_user = ratings_other_user.to('cuda:0')

                ratings_user = ratings_user * self.time_correlation_coefficients
                ratings_other_user = ratings_other_user * self.time_correlation_coefficients

                adjusted_ratings_user = ratings_user[non_zero_indices]
                adjusted_ratings_other_user = ratings_other_user[non_zero_indices]

                if adjusted_ratings_user.shape[0] < 15:
                    self.user_sim_matrix[i] = 0
                    continue

                # Compute pearson similarity
                self.user_sim_matrix[i] = self.pearson_correlation(adjusted_ratings_user, adjusted_ratings_other_user)

        similarity_numpy = self.user_sim_matrix.numpy()  # Convert PyTorch tensor to NumPy array

        return self.user_sim_matrix

    def run_test(self, data_matrix, time_matrix, timestamp_matrix, user_index, movie_index, time_matrix_test):
        # VERMA, SUJAL: You'll find the code in BTP.ipynb (but its for numpy input, convert into torch and fill this.)
        num_users = data_matrix.shape[0]
        self.user_sim_matrix = torch.zeros(num_users)

        # UPDATED CODE FOR PEARSON: Map user_index's items to TCC(i) (used globally)

        time_stamps = time_matrix[user_index]
        max_timestamp = 0
        if time_matrix_test[user_index][movie_index] != 0:
            max_timestamp = time_matrix_test[user_index][movie_index]
        else:
            max_timestamp = timestamp_matrix[user_index]

        self.time_correlation_coefficients = self.time_correlation_coefficient(self.get_month_difference(time_stamps, max_timestamp), self.sigma).to('cuda:0')

        for i in range(num_users):
            if i != user_index:  # Exclude the user itself

                # Find indices where both users have non-zero ratings
                non_zero_indices = numpy.nonzero((data_matrix[user_index] != 0) & (data_matrix[i] != 0))

                if len(non_zero_indices) < 0:
                    self.user_sim_matrix[i] = 0
                    continue

                # Filter ratings and timestamps to keep only non-zero elements
                ratings_user = data_matrix[user_index].clone()
                ratings_other_user = data_matrix[i].clone()

                # Adjust ratings with time correlation coefficients
                ratings_user = ratings_user.to('cuda:0')
                ratings_other_user = ratings_other_user.to('cuda:0')

                ratings_user = ratings_user * self.time_correlation_coefficients
                ratings_other_user = ratings_other_user * self.time_correlation_coefficients

                adjusted_ratings_user = ratings_user[non_zero_indices]
                adjusted_ratings_other_user = ratings_other_user[non_zero_indices]


                # Compute pearson similarity
                self.user_sim_matrix[i] = self.pearson_correlation(adjusted_ratings_user, adjusted_ratings_other_user)

        similarity_numpy = self.user_sim_matrix.numpy()  # Convert PyTorch tensor to NumPy array

        # Set NumPy to display three decimal places
        numpy.set_printoptions(formatter={'float': lambda x: "{:.3f}".format(x)})
        return self.user_sim_matrix

    def pearson_correlation(self, x, y):
        """
        Compute Pearson correlation coefficient between two vectors x and y.

        Parameters:
            x, y (torch.Tensor): Input vectors.

        Returns:
            float: Pearson correlation coefficient between x and y.
        """
        # Find indices where both x and y are non-zero
        non_zero_indices = torch.logical_and(x != 0, y != 0)

        # Filter x and y to keep only non-zero elements
        x_non_zero = x[non_zero_indices]
        y_non_zero = y[non_zero_indices]

        if len(x_non_zero) == 0 or len(y_non_zero) == 0:
            return 0  # If no non-zero elements, return 0

        mean_x = torch.mean(x_non_zero)
        mean_y = torch.mean(y_non_zero)
        numerator = torch.sum((x_non_zero - mean_x) * (y_non_zero - mean_y))
        denominator = torch.sqrt(torch.sum((x_non_zero - mean_x)**2) * torch.sum((y_non_zero - mean_y)**2))
        if denominator == 0:
            return 0  # To handle division by zero
        else:
            return numerator / denominator
        
class Cosine:

    def __init__(self, sigma=1):
        self.sigma = sigma

    def run(self, data_matrix, time_matrix, timestamp_matrix, user_index):
        # VERMA, SUJAL: You'll find the code in BTP.ipynb (but its for numpy input, convert into torch and fill this.)
        num_users = data_matrix.shape[0]
        self.user_sim_matrix = torch.zeros(num_users)

        for i in range(num_users):
            if i != user_index:  # Exclude the user itself
                non_zero_indices = numpy.nonzero((data_matrix[user_index] != 0) & (data_matrix[i] != 0))
                # Find indices where both users have non-zero ratings

                if len(non_zero_indices) == 0:
                    self.user_sim_matrix[i] = 0
                    continue

                # Filter ratings and timestamps to keep only non-zero elements
                ratings_user = data_matrix[user_index][non_zero_indices].clone()
                ratings_other_user = data_matrix[i][non_zero_indices].clone()

                # Adjust ratings with time correlation coefficients
                ratings_user = ratings_user.to('cuda:0')
                ratings_other_user = ratings_other_user.to('cuda:0')

                adjusted_ratings_user = ratings_user #* time_correlation_coefficients

                adjusted_ratings_other_user = ratings_other_user #* time_correlation_coefficients


                # Compute pearson similarity
                self.user_sim_matrix[i] = self.pearson_correlation(adjusted_ratings_user, adjusted_ratings_other_user)

        similarity_numpy = self.user_sim_matrix.numpy()  # Convert PyTorch tensor to NumPy array

        # Set NumPy to display three decimal places
        numpy.set_printoptions(formatter={'float': lambda x: "{:.3f}".format(x)})
        return self.user_sim_matrix

    def pearson_correlation(self, x, y):
        """
        Compute Pearson correlation coefficient between two vectors x and y.

        Parameters:
            x, y (torch.Tensor): Input vectors.

        Returns:
            float: Pearson correlation coefficient between x and y.
        """
        # Find indices where both x and y are non-zero
        non_zero_indices = torch.nonzero(torch.logical_and(x != 0, y != 0))

        # Filter x and y to keep only non-zero elements
        x_non_zero = x[non_zero_indices]
        y_non_zero = y[non_zero_indices]

        if len(x_non_zero) == 0 or len(y_non_zero) == 0:
            return 0  # If no non-zero elements, return 0

        mean_x = torch.mean(x_non_zero)
        mean_y = torch.mean(y_non_zero)
        numerator = torch.sum((x_non_zero - mean_x) * (y_non_zero - mean_y))
        denominator = torch.sqrt(torch.sum((x_non_zero - mean_x)**2) * torch.sum((y_non_zero - mean_y)**2))
        if denominator == 0:
            return 0  # To handle division by zero
        else:
            return numerator / denominator