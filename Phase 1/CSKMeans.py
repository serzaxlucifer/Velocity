import matplotlib.pyplot as plt
from math import gamma
import numpy as np
import torch


class CSKMeans:

    def GradientOptimizer(self, learning_rate=0.02, tolerance=100):
        # self.X contains centroids.

        device = torch.device('cuda:0')

        # Convert the NumPy array to a PyTorch tensor and move it to the CUDA device
        X = torch.tensor(self.X, device=device)

        prev_loss = float('inf')
        X.requires_grad = True
        optimizer = torch.optim.SGD([X], lr=learning_rate)

        for i in range(self.Tmax):
            optimizer.zero_grad()
            loss = self.fitnessTORCH(X)
            print("Loss at epoch ", i , " = ", loss)
            loss.backward()
            optimizer.step()

            if abs(prev_loss - loss.item()) < tolerance:
                break
            prev_loss = loss.item()

        X.detach()
        X = X.numpy()
        print(X)
        return X

    def __init__(self, data_matrix, time_matrix, P=25, clusters=5, pa=0.25, beta=1.5, bound=None, plot=False, min=True, verbose=False, Tmax=300, max_iters=100, optimizer="CuckooSearch", lr=0.02, tolerance=100):

        '''
        PARAMETERS:

        fitness: A FUNCTION WHICH EVALUATES COST (OR THE FITNESS) VALUE
        min: True if fitness function has to be minimized and false if maximized.
        verbose: True for printing fitness score at each iteration
        plot: True to plot fitness score vs iterations.
        P: POPULATION SIZE
        clusters: TOTAL DIMENSIONS
        pa: ASSIGNED PROBABILITY
        beta: LEVY PARAMETER
        bound: AXIS BOUND FOR EACH DIMENSION
        X: PARTICLE POSITION OF SHAPE (P,n)
        Tmax: MAXIMUM ITERATION
        best: GLOBAL BEST POSITION OF SHAPE (n,1)

        '''
        self.data_matrix = data_matrix.copy()      # add for our purpose
        self.time_matrix = time_matrix
        self.P = P
        self.max_iters = max_iters
        self.n = clusters
        self.Tmax = Tmax
        self.pa = pa
        self.beta = beta
        self.bound = bound      # How to process bounds here? Assume bound is of composed of two parameters[0 , maximum_rating]
        self.plot = plot
        self.min = min
        self.verbose = verbose  # too many words?!

        self.X = self.initialize_centroids()

        if optimizer == "CuckooSearch":
          self.best = self.CuckooSearch()
        elif optimizer == "Gradient":
          self.best = self.GradientOptimizer(lr, tolerance)
        else:
          print("OPTIMIZER ", optimizer, " not found! Skipping Optimization Process.")
          self.best = self.X[0]

        # Now, run the K-means functions

        self.fit()

        # Now we have labels! MAKE CHANGES HERE

        # Initialize dictionaries to store user vectors for each cluster
        self.cluster_matrices = {cluster_label: [] for cluster_label in range(self.n)}
        self.user_ids = [0 for _ in range(self.data_matrix.shape[0])]
        clusterCounts = [0 for _ in range(self.n)]
        self.timeClusters = {cluster_label: [] for cluster_label in range(self.n)}

        # Group user vectors based on cluster labels
        for user_index, cluster_label in enumerate(self.labels):
            self.cluster_matrices[int(cluster_label)].append(self.data_matrix[user_index])
            self.timeClusters[int(cluster_label)].append(self.time_matrix[user_index])
            self.user_ids[user_index] = clusterCounts[int(cluster_label)]
            clusterCounts[int(cluster_label)] = clusterCounts[int(cluster_label)] + 1

        # Convert lists to numpy arrays
        for cluster_label, user_vectors in self.cluster_matrices.items():
            self.cluster_matrices[cluster_label] = np.array(user_vectors)

        for cluster_label, time_vectors in self.timeClusters.items():
            self.timeClusters[cluster_label] = np.array(time_vectors)

    def initialize_centroids(self):    # add for our purpose
      all_centroids = np.empty((self.P, self.n, self.data_matrix.shape[1]))

      for i in range(self.P):
          # Randomly select num_centroids user vectors from the data matrix
          centroid_indices = np.random.choice(self.data_matrix.shape[0], size=self.n, replace=False)
          centroids = self.data_matrix[centroid_indices]
          all_centroids[i] = centroids

      return all_centroids

    def fitness(self, centroids):    # centroid shape = [k_clusters, num_of_movies], data_matrix shape = [users, num_of_movies]
      '''
      Function to evaluate fitness of a solution.
      '''
      # Initialize total fitness value
      total_fitness = 0

      # Iterate over each data point in the matrix
      for data_point in self.data_matrix:

        # Calculate Euclidean distance to each centroid
        distances = np.linalg.norm(centroids - data_point, axis=1)    # broadcasting along axis = 1  | data_point shape = [num_of_movies, ] (1-D)
        nearest_centroid_distance = np.min(distances)
        total_fitness += nearest_centroid_distance

      return total_fitness

    def fitnessTORCH(self, centroids):
        '''
        Function to evaluate fitness of a solution.
        '''
        # Initialize total fitness value
        total_fitness = 0

        # Convert centroids to PyTorch tensor and ensure it's on the same device as data_matrix
        #centroids_tensor = torch.tensor(centroids, dtype=torch.float32, device=self.data_matrix.device)
        total_fitness = torch.zeros(1, device=centroids.device)

        # Iterate over each data point in the matrix
        for data_point in self.data_matrix:

            # Convert data_point to PyTorch tensor and ensure it's on the same device as centroids
            data_point_tensor = torch.tensor(data_point, dtype=torch.float32, device=centroids.device)

            # Calculate Euclidean distance to each centroid
            distances = torch.norm(centroids - data_point_tensor, dim=1)  # broadcasting along dim = 1
            nearest_centroid_distance = torch.min(distances)
            total_fitness += nearest_centroid_distance.item()

        return total_fitness


    def update_position_1(self):

        num = gamma(1+self.beta)*np.sin(np.pi*self.beta/2)
        den = gamma((1+self.beta)/2)*self.beta*(2**((self.beta-1)/2))
        σu = (num/den)**(1/self.beta)
        σv = 1
        u = np.random.normal(0, σu, (self.n, self.data_matrix.shape[1]))
        v = np.random.normal(0, σv, (self.n, self.data_matrix.shape[1]))
        S = u/(np.abs(v)**(1/self.beta))

        # DEFINING GLOBAL BEST SOLUTION BASED ON FITNESS VALUE

        for i in range(self.P):
            if i==0:
                self.best = self.X[i,:,:].reshape(self.n, self.data_matrix.shape[1]).copy()
            else:
                self.best = self.optimum(self.best, self.X[i,:,:].reshape(self.n, self.data_matrix.shape[1]))

        Xnew = self.X.copy()

        for i in range(self.P):
            Xnew[i,:,:] += np.random.randn(self.n, self.data_matrix.shape[1])*0.01*S*(Xnew[i,:,:]-self.best)
            self.X[i,:,:] = self.optimum(Xnew[i,:,:], self.X[i,:,:])

    def update_position_2(self):

        Xnew = self.X.copy()
        Xold = self.X.copy()

        for i in range(self.P):

            d1,d2 = np.random.randint(0,self.P,2)

            for j in range(self.n):
                r = np.random.rand()
                if r < self.pa:
                    Xnew[i,j] += np.random.rand()*(Xold[d1,j]-Xold[d2,j])

            self.X[i,:,:] = self.optimum(Xnew[i,:,:], self.X[i,:,:])

    def optimum(self, best, particle_x):

        if self.min:
            if self.fitness(best) > self.fitness(particle_x):
                best = particle_x.copy()
        else:
            if self.fitness(best) < self.fitness(particle_x):
                best = particle_x.copy()
        return best

    def clip_X(self):

        min_rating = self.bound[0]
        max_rating = self.bound[1]

        self.X = np.clip(self.X, min_rating, max_rating)

    def CuckooSearch(self):
        # 3D CUCKOO SEARCH (vectorized)

        self.fitness_time, self.time = [], []

        for t in range(self.Tmax):

            self.update_position_1()
            self.clip_X()
            self.update_position_2()
            self.clip_X()
            self.fitness_time.append(self.fitness(self.best))
            self.time.append(t)
            if self.verbose:
                print('Iteration:  ',t,'| best global fitness (cost):',round(self.fitness(self.best),7))

        print('\nOPTIMUM SOLUTION\n  >', np.round(self.best.reshape(-1),7))
        print('\nOPTIMUM FITNESS\n  >', np.round(self.fitness(self.best),7))

        print()

        if self.plot:
            self.Fplot()

        return self.best

    def Fplot(self):

        # PLOTS GLOBAL FITNESS (OR COST) VALUE VS ITERATION GRAPH

        plt.plot(self.time, self.fitness_time)
        plt.title('Fitness value vs Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness value')
        plt.show()

    def cosine_distance(self, x1, x2):
      dot_product = np.dot(x1, x2)
      norm_x1 = np.linalg.norm(x1)
      norm_x2 = np.linalg.norm(x2)
      return 1.0 - dot_product / (norm_x1 * norm_x2)

    def fit(self):
        self.centroids = self.best

        # Assign data points to clusters
        for _ in range(self.max_iters):
            clusters = [[] for _ in range(self.n)]
            for point in self.data_matrix:
                distances = [self.cosine_distance(point, centroid) for centroid in self.centroids]
                closest_centroid_idx = np.argmin(distances)
                clusters[closest_centroid_idx].append(point)

            # Update centroids
            new_centroids = []
            for cluster in clusters:
                new_centroid = np.mean(cluster, axis=0)
                new_centroids.append(new_centroid)
            new_centroids = np.array(new_centroids)

            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        # Assign labels based on final centroids
        self.labels = np.zeros(self.data_matrix.shape[0])
        for i, cluster in enumerate(clusters):
            for point in cluster:
                self.labels[np.where((self.data_matrix == point).all(axis=1))] = i

