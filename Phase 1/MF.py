import torch
import numpy
import sys
import math
from datetime import datetime

class MatrixFactorization:

  def __init__(self, K=100, steps=25000, alpha=0.002, beta=0.02):
    self.K = 100
    self.steps = steps
    self.alpha = alpha
    self.beta = beta

  def run(self, R, user_index, K=100, steps=25000, alpha=0.002, beta=0.02):
    R = numpy.array(R)
    self.original_ratings = R[user_index].copy()

    # N: num of User
    self.N = R.shape[0]
    # M: num of Movie
    self.M = R.shape[1]
    # Num of Features
    self.K = K
    self.steps = steps
    self.alpha = alpha
    self.beta = beta

    P = numpy.random.rand(self.N,self.K)
    Q = numpy.random.rand(self.M,self.K)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.R_tensor = torch.tensor(R, dtype=torch.float32, device=device)
    self.P_tensor = torch.tensor(P, dtype=torch.float32, device=device, requires_grad=True)
    self.Q_tensor = torch.tensor(Q.T, dtype=torch.float32, device=device, requires_grad=True)  # Transpose Q for proper matrix multiplication

    self.prediction_matrix = self.matrix_factorization_GPU()[user_index]      # NVIDIA Drivers and Hardware required. PyTorch must run on it.

    for i in range(self.original_ratings.shape[0]):
      if self.original_ratings[i] != 0:
        self.prediction_matrix[i] = 0

    self.results = self.prediction_matrix

    return self.prediction_matrix


  def matrix_factorization_GPU(self):
      '''
      R: rating matrix
      P: |U| * K (User features matrix)
      Q: |D| * K (Item features matrix)
      K: latent features
      steps: iterations
      alpha: learning rate
      beta: regularization parameter'''

      # Convert input matrices to PyTorch tensors and move them to GPU if available

      optimizer = torch.optim.Adam([self.P_tensor, self.Q_tensor], lr=self.alpha)

      for step in range(self.steps):
          # Calculate predicted ratings
          pred_ratings = torch.mm(self.P_tensor, self.Q_tensor)
          # Calculate error
          error = self.R_tensor - pred_ratings
          # Compute loss with regularization
          loss = torch.sum(torch.square(error * (self.R_tensor > 0))) #+ self.beta/2 * (torch.sum(torch.square(self.P_tensor)) + torch.sum(torch.square(self.Q_tensor)))

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          if step % 100 == 0:
              sys.stdout.write("\rStep {}, Loss: {:.4f}".format(step, loss.item()))
              sys.stdout.flush()

      print()

      return numpy.dot(self.P_tensor.detach().cpu().numpy(), self.Q_tensor.detach().cpu().numpy())