import numpy as np
import torch
from TCCF import TCCF
from MF import MatrixFactorization
from WithoutTCCF import WithoutTCCF
from SimilarityMetrics import Pearson, Cosine
import math
import sys

class TestDriver:
  def __init__(self, data_matrix_train, data_matrix_test, time_matrix_train, user_index, time_matrix_test):
      collaborativeFilter = TCCF(Pearson(1))
      collaborativeFilterWTCC = WithoutTCCF(Cosine(1))
      predictedSet = collaborativeFilter.run(data_matrix_train.copy(), time_matrix_train.copy(), user_index, time_matrix_test.copy(), data_matrix_test)
      p2set = collaborativeFilterWTCC.run(data_matrix_train.copy(), time_matrix_train.copy(), user_index, data_matrix_test)
      predictedSet = collaborativeFilter.results

      # Define Intersection

      originalSet = data_matrix_test[user_index]

      count = 0
      MAE = 0
      MAEWT = 0

      for i in range(originalSet.shape[0]):
          if originalSet[i] != 0 and predictedSet[i] != 0:
              print("Predicted rating: ", predictedSet[i], "  | Original Rating: ", originalSet[i], "  | Without TCCF: ", p2set[i])
              MAE += abs(originalSet[i] - predictedSet[i])
              MAEWT += abs(originalSet[i] - p2set[i])
              count += 1

      MAE = MAE/count
      MAEWT = MAEWT/count

      print("MAE for ", user_index, "  : ", MAE, "  | MAE without TCCF = ", MAEWT)
      print()