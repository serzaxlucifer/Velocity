import numpy as np
import pandas as pd
import sys
from datetime import datetime
from VelocityRecommender import VelocityRecommender
from TCCF import TCCF, WithoutTCCF
from MatrixFactorization import MatrixFactorization
from TestDriver import TestDriver
from SimilarityMetrics import Pearson, Cosine

# The primary goal while writing this code has been to make it easily extensible and maintainable. The code is 
# well-structured and modular, with each function performing a specific task. Every thing has been defined in its own
# class utilizing Strategy Design Pattern, which makes it easy to switch between different algorithms. 

####                         LOAD DATASET                         ####

# reading users file:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,encoding='latin-1')

# reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,encoding='latin-1')

# reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,
encoding='latin-1')

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_train = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')
ratings_train.shape, ratings_test.shape

n_users = ratings.user_id.unique().shape[0]
n_items = ratings.movie_id.unique().shape[0]

data_matrix = np.zeros((n_users, n_items))
for line in ratings.itertuples():
    data_matrix[line[1]-1, line[2]-1] = line[3]

time_matrix = np.zeros((n_users, n_items))

for line in ratings.itertuples():
    time_matrix[line[1]-1, line[2]-1] = line[4]

average_ratings = np.mean(data_matrix, axis=0)

# Create a columnar numpy array with movie index and corresponding average rating
movie_indices = np.arange(0, 1682)  # Assuming movie indices start from 1
movie_ratings_array = np.column_stack((movie_indices, average_ratings))

# Initialize empty lists to store months and years
months = []
years = []

# Iterate over each row in the 2D array
for row in time_matrix:
    # Convert UNIX timestamps to datetime objects for each row
    time_matrix_datetime = [datetime.utcfromtimestamp(ts) for ts in row]

    # Extract month and year from datetime objects for each row
    row_months = [dt.month for dt in time_matrix_datetime]
    row_years = [dt.year for dt in time_matrix_datetime]

    # Append row results to the overall lists
    months.append(row_months)
    years.append(row_years)

# Convert lists to numpy arrays
months = np.array(months)
years = np.array(years)

####                 DATASET LOADED                ####

recommender = VelocityRecommender(data_matrix, time_matrix, months, years, items, 18, TCCF(), 1)   # In second last param, can also use MatrixFactorization()

recommender.cluster(2, P=25, pa=0.25, beta=1.5, bound=[0,5], plot=False, min=True, verbose=True, Tmax=15, max_iters=100, lr=0.02, tolerance=100, optimizer="CuckooSearch")   
# We tried implementing other optimizers like Gradient Descent to minimize the fitness function discussed in our BTP Report but didn't work out well.

recs = recommender.recommendItems(0, 15)

print(recs)

for index in recs:
    print(items.iloc[index][1])



#####      FOR ACCURACY TESTING PURPOSES:

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,encoding='latin-1')

# reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u1.base', sep='\t', names=r_cols,encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/u1.test', sep='\t', names=r_cols,encoding='latin-1')

# reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,
encoding='latin-1')

n_users_train = 943
n_items_train = 1682
n_users_test = 943
n_items_test = 1682

data_matrix_train = np.zeros((n_users_train, n_items_train))
data_matrix_test = np.zeros((n_users_test, n_items_test))

for line in ratings.itertuples():
    data_matrix_train[line[1]-1, line[2]-1] = line[3]

for line in ratings_test.itertuples():
    data_matrix_test[line[1]-1, line[2]-1] = line[3]

time_matrix_train = np.zeros((n_users_train, n_items_train))
time_matrix_test = np.zeros((n_users_test, n_items_test))

for line in ratings.itertuples():
    time_matrix_train[line[1]-1, line[2]-1] = line[4]

for line in ratings_test.itertuples():
    time_matrix_test[line[1]-1, line[2]-1] = line[4]

TestDriver(data_matrix_train, data_matrix_test, time_matrix_train, 0, time_matrix_test)