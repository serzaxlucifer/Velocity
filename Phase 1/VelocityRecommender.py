import numpy as np
import torch
from CollaborativeFiltering import CollaborativeFiltering
from CSKMeans import CSKMeans
from datetime import datetime

# # # # # # # # # # # # # # # # # # # # # # # # #
# Module:  VELOCITY RECOMMENDER (Phase 1)       #
# Authors: Mukul Malik, Mukul Verma             #
# # # # # # # # # # # # # # # # # # # # # # # # #

class VelocityRecommender:

  def __init__(self, data_matrix, time_matrix, months, years, items, numG, strategy, clusters=1, optimizer="CuckooSearch"):
    self.data_matrix = data_matrix.copy()
    self.time_matrix = time_matrix.copy()
    self.strategy = strategy              # Collaborative Filtering Method MF vs TCCF, upcoming: TripleMF
    self.months = months.copy()
    self.optimizer = optimizer
    self.years = years.copy()
    self.items = items.copy()
    self.numGenres = numG
    self.clusters = clusters
    self.user_index = 0
    self.coldStartThreshold = 10
    self.H = np.zeros(self.numGenres)

    self.tagMovieMapping = np.empty(self.numGenres, dtype=object)
    masked_data_matrix = np.ma.masked_equal(data_matrix, 0)
    self.average_ratings = np.ma.mean(masked_data_matrix, axis=0)

    # This helps in powering the Diversity Generation ALgorithm.
    for i in range(self.numGenres):
        self.tagMovieMapping[i] = []  # Initialize each element as an empty list


    for i in range(items.shape[0]):
        for j in range(self.numGenres):
              if( items.iloc[i][6+j] == 1 ):
                  self.tagMovieMapping[j].append(i)

    # Sort the average ratings in descending order
    indexArr_desc = np.argsort(-self.average_ratings)
    self.topMovies = indexArr_desc

    self.tagMovieMax = np.empty(self.numGenres, dtype=object)

    # Fill the NumPy array with Python lists
    for i in range(self.numGenres):
        self.tagMovieMax[i] = []  # Initialize each element as an empty list

    for i in range(indexArr_desc.shape[0]):
        for j in range(self.numGenres):
              if( items.iloc[indexArr_desc[i]][6+j] == 1 ):
                  self.tagMovieMax[j].append(indexArr_desc[i])


    self.clusterMatrices = []
    self.clusterLabels = []
    self.userID = []

    print("Engine Initialized!")
    print()

  def setStartThreshold(self, num):
    inp = 'y'
    if num == 0:
      print("Threshold can't be 0.")
      inp = 'n'

    elif num < 5:
      inp = input("We don't recommend using the collaborative filtering engine for a threshold below 5! This may lead to unstable recommendations. Are you sure you want to do this? Y/N:   ")

    if inp == 'y':
      self.coldStartThreshold = num

  # Invoke this periodically.
  def cluster(self, num_c, P=25, pa=0.25, beta=1.5, bound=None, plot=False, min=True, verbose=False, Tmax=300, max_iters=100, lr=0.02, tolerance=100, optimizer="CuckooSearch"):

    print("OPERATION: Cluster         (Beginning) K-Means Clustering")
    print()
    self.clusters = num_c
    self.optimizer = optimizer
    cluster_data = CSKMeans(self.data_matrix, self.time_matrix, P, self.clusters, pa, beta, bound, plot, min, verbose, Tmax, max_iters, optimizer, lr, tolerance)

    self.clusterMatrices = cluster_data.cluster_matrices
    self.clusterLabels = cluster_data.labels
    self.userID = cluster_data.user_ids
    self.clusterTimes = cluster_data.timeClusters
    print("Clustering Operation Finished.")
    print()

  def collaborativeFiltering(self, nonCluster=True, items=10):
    # Strategy Design Pattern used here to dynamically switch between MF (Matrix Factorization) and TCCF.

    if nonCluster == True:
      self.predictions = CollaborativeFiltering(self.strategy, self.data_matrix, self.time_matrix, self.user_index).results

    else:
      self.predictions = CollaborativeFiltering(self.strategy, self.clusterMatrices[self.clusterLabels[self.user_index]], self.clusterTimes[self.clusterLabels[self.user_index]], self.userID[self.user_index]).results

    self.top_indices = np.argsort(self.predictions)[(-1*int(items)):]      # Indices of top-N items
    print("TOP RATINGS: ", self.predictions[self.top_indices])
    print()
    self.predicted_ratings = self.predictions[self.top_indices]   # Predicted ratings of top-N items


  def editRating(self, user, item, rating):
    import time

    self.data_matrix[user, item] = rating
    self.time_matrix[user, item] = int(time.time())
    self.months[user, item] = datetime.now().month
    self.years[user, item] = datetime.now().year

  def deleteRating(self, user, item):
    self.data_matrix[user, item] = 0
    self.time_matrix[user, item] = 0
    self.months[user, item] = 0
    self.years[user, item] = 0

  def computeTimeEntropy(self):
    # Compute time entropy for each genre for self.user_index!
    from collections import defaultdict
    from math import log

    self.H = np.zeros(self.numGenres)
    for tag in range(self.numGenres):
        monthly_ratings = defaultdict(lambda: {'total_rating': 0, 'count': 0})
        count = 0

        for movie in self.tagMovieMapping[tag]:
            if self.data_matrix[self.user_index, movie] != 0:
                key = (self.months[self.user_index, movie], self.years[self.user_index, movie])
                monthly_ratings[key]['total_rating'] += self.data_matrix[self.user_index, movie]
                monthly_ratings[key]['count'] += 1
                count += 1

        if count == 0:
           self.H[tag] = -500

        # Now average them.
        for key, value in monthly_ratings.items():
            month, year = key
            average_ratings = value['total_rating'] / value['count']
            self.H[tag] += (-1*((average_ratings*value['count'])/count)*log(value['count']/count))

  def recommendItems(self, user_index, items=15):
    import math
    self.user_index = user_index
    ratings = np.count_nonzero(self.data_matrix[user_index])
    self.final_recommendations = []

    if ratings > self.coldStartThreshold:   # Pattern 1
      print("Starting Collaborative Filtering!")
      print()
        
      if self.clusters < 2:
        self.collaborativeFiltering(True, items/2)
      else:
        self.collaborativeFiltering(False, items/2)

      print("Collaborative Filtering Ended")
        
      print(self.top_indices)
      for i in range(len(self.top_indices)):
          self.final_recommendations.append(self.top_indices[i])

      # Now self.top_indices, self.predicted_ratings contain the CF outputs.
      print("Computing Entropy")
      print()
      self.computeTimeEntropy()
      print("Entropy Computed")
      print()
      print(self.H)

      # Formulate Tag/Genre Table (Diversified Recommendations!)
      top_entropies = np.argsort(self.H)[::-1]    # take the maximum entropy still
      print(top_entropies)

      limit = math.ceil(items/4)
      itemsPerTag = max(1, math.ceil(limit/len(top_entropies)))
      for i in range(limit):
          if i == len(top_entropies):
              break

          count = 0
          for movie in self.tagMovieMax[top_entropies[i]]:
                if self.data_matrix[self.user_index, movie] == 0:
                    self.final_recommendations.append(movie)
                    count += 1
                if count == itemsPerTag:
                    break

      # Now Add Diversity Stuff.
      indices = np.where(self.H == -500)[0]
      limit = math.ceil(items/4)
      if(len(indices) != 0):
        itemsPerTag = max(1, math.ceil(limit/len(indices)))

      for i in range(limit):
          if i == len(indices):
              break

          for j in limit(itemsPerTag):
              self.final_recommendations.append(self.tagMovieMax[indices[i]][j])      # add boundary condition here!


    else:                                   # Pattern 2
      for i in range(items):
          self.final_recommendations.append(self.topMovies[i])

    return self.final_recommendations

  #def addMovie():
  #def addUser():
  #def deleteUser():
  #def deleteMovie():
