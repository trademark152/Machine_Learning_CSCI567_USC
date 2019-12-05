import numpy as np
import math
import random

'''
Class to implement k means ++ algorithm
High level pseudo code:
1) initiate a random point as the first centroid
2) calculate for each remaining point, the distance to all existing centroids and find each remaining point's closest centroid and corresponding distances
3) get the max (largest probability) of these minimum distances (between remaining points to their closest centroid) to be the next centroid
4) repeat until all centroids are found
'''
class K_Means_Plus_Plus:
    """Input is a 2D list of n d-dimensional points points_list, number of clusters k and some randomized generator"""
    def __init__(self, points_list, k, generator):
        self.generator = generator
        self.centroid_count = 0  # number of centroid
        self.point_count = len(points_list)  # number of point
        self.cluster_count = k  # number of cluster
        self.points_list = list(points_list)  # import data
        self.initialize_random_centroid()
        self.initialize_other_centroids()


    """Picks a random point to serve as the first centroid"""
    def initialize_random_centroid(self):
        self.centroid_list = []  # initialize centroid list
        self.centroid_list_index = []  # initialize centroid list index

        # pick a random point to start
        # index = random.randint(0, len(self.points_list)-1)
        index = self.generator.randint(0, len(self.points_list) - 1)

        # append this point to the centroid list
        new_centroid = self.remove_point(index)
        self.centroid_list.append(new_centroid)

        # update centroid count to 1 for the initial cluster
        self.centroid_count = 1

        # update centroid final index
        self.centroid_list_index.append(index)

    """Removes point associated with given index so it cannot be picked as a future centroid.
    Returns list containing coordinates of newly removed centroid"""
    def remove_point(self, index):
        new_centroid = self.points_list[index]

        # remove this point out of the remaining points_list
        del self.points_list[index]

        # return the newly acquired centroid
        return new_centroid

    """Finds the other k-1 centroids from the remaining lists of points"""
    def initialize_other_centroids(self):
        while not self.is_finished():
            # find all distances of each remaining points to their closest centroid
            distances = self.find_smallest_distances()

            """Chooses an index based on weighted probability """
            chosen_index = self.choose_weighted(distances)

            self.centroid_list.append(self.remove_point(chosen_index))
            self.centroid_count += 1

            # update centroid final index
            self.centroid_list_index.append(chosen_index)

    """Calculates distance from each remaining point to its nearest cluster center."""
    def find_smallest_distances(self):
        distance_list = []
        # loop through each point in the remaining list
        for point in self.points_list:
            distance_list.append(self.find_nearest_centroid(point))
        return distance_list

    """Finds centroid nearest to the given point, and returns its distance"""
    def find_nearest_centroid(self, point):
        # initiate distance
        min_distance = math.inf

        # loop through each existing centroid
        for centroid in self.centroid_list:
            # find distance between this centroid and the point
            distance = self.euclidean_distance(centroid, point)
            if distance < min_distance:
                # update the min_distance
                min_distance = distance
        return min_distance

    """ Then chooses new center based on the weighted probability of these distances"""
    def choose_weighted(self, distance_list):
        # square up the distance like the pseud-code given
        distance_list = [x**2 for x in distance_list]

        # weighted up each distance by normalizing by its sum
        weighted_list = self.weight_values(distance_list)

        # get the corresponding indices
        # indices = [i for i in range(len(distance_list))]

        # return the index with largest probability by providing probabilities associated with each entry in indices
        # return np.random.choice(indices, p = weighted_list)
        return np.argmax(weighted_list)

    """Weights values from [0,1]"""
    def weight_values(self, list):
        sum = np.sum(list)
        return [x/sum for x in list]

    """computes N-d euclidean distance between two points represented as lists:
     (x1, x2, ..., xd) and (y1, y2, ..., yd), default is 2-norm"""
    def euclidean_distance(self, point1, point2):
        point1 = np.asarray(point1)
        point2 = np.asarray(point2)

        return np.linalg.norm(point2-point1)

    """Checks to see if final condition has been satisfied (when K centroids have been created)"""
    def is_finished(self):
        outcome = False
        if self.centroid_count == self.cluster_count:
            outcome = True
        return outcome

    """Returns final centroid values"""
    def final_centroids(self):
        return self.centroid_list

    """Returns final centroid values"""
    def final_centroids_index(self):
        return self.centroid_list_index

def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''
    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centers according to the lecture and notebook
    # Choose 1st center randomly and use Euclidean distance to calculate other centers.
    # raise Exception(
    #          'Implement get_k_means_plus_plus_center_indices function in Kmeans.py')

    # # initialize the output
    # centers = [0] * n_cluster

    # perform the K_means_Plus_Plus algorithm
    kmpp = K_Means_Plus_Plus(x, n_cluster, generator)

    centers = kmpp.final_centroids_index()
    # print(centers)

    # DO NOT CHANGE CODE BELOW THIS LINE

    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers


def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)

""" this class is used for toy dataset"""
class KMeans():
    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates a Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"

        # get the data dimension
        N, D = x.shape
        # print("N: ", N)
        # print("D: ", D)

        # initiate centers using centroid function
        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #      'Implement fit function in KMeans class')

        """ APPROACH 1"""
        # get the data row corresponding to these initial centroids:
        # centroids = x[self.centers, :]  # same with μk
        # print("centroids: ", centroids)
        #
        # # initialize a big number and num_iter
        # J = 10 ** 10
        # num_iter = 0
        #
        # # loop through each iteration
        # for i in range(self.max_iter):
        #     # calculate distance between x and centroids (expand the objective function??
        #     """ F({μk},{rnk})=∑i=1-N ∑k=1-K rnk*∥μk−xn∥2 """
        #     # distances = np.sum(np.square(centroids), axis=1) - 2 * np.dot(x, centroids.T) + np.transpose([np.sum(np.square(x), axis=1)])
        #     distances = np.transpose(np.sum(((x - np.expand_dims(centroids, axis=1)) ** 2), axis=2))
        #     # print("distances: ", distances)
        #     # print("shape distances: ", len(distances))
        #
        #     # update membership based on shortest distance
        #     y = np.argmin(distances, axis=1)
        #     # print("membership: ", y)
        #
        #     # update number of iterations
        #     num_iter = num_iter + 1
        #
        #     # update objective function why divided by N
        #     """ F({μk},{rnk})=∑i=1-N ∑k=1-K rnk*∥μk−xn∥2 """
        #     print(centroids[y, :])
        #     Jnew = np.sum(np.square(x - centroids[y, :])) / N
        #
        #     # if converged then break the loop
        #     if abs(J - Jnew) <= self.e:
        #         break
        #
        #     # if not, update objective
        #     J = Jnew
        #     for j in range(self.n_cluster):
        #         belong_vec = (y == j)
        #         if np.sum(belong_vec) == 0:
        #             continue
        #
        #         # update centroids
        #         centroids[j, :] = np.sum(x[belong_vec, :], axis=0) / np.sum(belong_vec)


        """ APPROACH 2 """
        """
        to calculate distortion objective: 
        need inputs: mu (centroid location n_cluster*D), x (data n*D), r(membership n*1 with each value is the cluster group from index of mu)
        retunr the value of objective function
        """
        def distortion(centroids, x, y):
            # number of data
            N = x.shape[0]

            """ F({μk},{rnk})=∑i=1N∑k=1K  rnk*∥μk−xn∥^2 """
            # perform index [y==k] to represent rnk
            return np.sum([np.sum((x[y == k] - centroids[k]) ** 2) for k in range(self.n_cluster)]) / N

        # Initialize centroids, membership, distortion
        centroids = x[self.centers, :] # use already-initialized initial centers
        y = np.zeros(N, dtype=int)
        J = distortion(centroids, x, y)

        # Loop until convergence/max_iter
        num_iter = 0
        while num_iter < self.max_iter:
            # Compute all distances between current centroids and points??
            # np.expand_dims: Expand the shape of an array
            l2 = np.sum(((x - np.expand_dims(centroids, axis=1)) ** 2), axis=2)

            # locate index with lowest distance to assign membership
            y = np.argmin(l2, axis=0)

            # Compute new distortion
            J_new = distortion(centroids, x, y)
            if np.absolute(J - J_new) <= self.e:
                break

            # update distortion
            J = J_new

            # Compute means
            mu_new = np.array([np.mean(x[y == k], axis=0) for k in range(self.n_cluster)])

            # find new index that fits ??
            index = np.where(np.isnan(mu_new))

            # update means
            mu_new[index] = centroids[index]

            # update centroids
            centroids = mu_new

            # update iteration
            num_iter += 1


        # DO NOT CHANGE CODE BELOW THIS LINE
        return centroids, y, num_iter


""" this class is used for digit classification """
class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #      'Implement fit function in KMeansClassifier class')

        """ APPPROACH 1"""
        # initrow = np.random.choice(N, self.n_cluster)
        # means = x[initrow, :]
        # J = 10 ** 10
        # number_of_updates = 0
        # for i in range(self.max_iter):
        #     dists = -2 * np.dot(x, means.T) + np.sum(np.square(means), axis=1) + np.transpose(
        #         [np.sum(np.square(x), axis=1)])
        #     membership = np.argmin(dists, axis=1)
        #     number_of_updates = number_of_updates + 1
        #     Jnew = np.sum(np.square(x - means[membership, :])) / N
        #     if abs(J - Jnew) <= self.e:
        #         break
        #     J = Jnew
        #     for j in range(self.n_cluster):
        #         belong_vec = (membership == j)
        #         if np.sum(belong_vec) == 0:
        #             continue
        #         means[j, :] = np.sum(x[belong_vec, :], axis=0) / np.sum(belong_vec)
        # centroids = means
        # centroid_labels = np.zeros(self.n_cluster)
        # for i in range(self.n_cluster):
        #     belong_vec = (membership == i)
        #     if np.sum(belong_vec) == 0:
        #         centroid_labels[i] = 0
        #         continue
        #     labels = y[belong_vec]
        #     tu = sorted([(np.sum(labels == j), j) for j in set(labels)], key=lambda x: (x[0], -x[1]))
        #     centroid_labels[i] = tu[-1][1]

        """ APPROACH 2"""
        # create a KMeans object
        k_means = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)

        # find the centroid and membership of the given data
        centroids, membership, numIter = k_means.fit(x)

        # initialize votes list of dict: [cluster1, cluster2]  {label1: numVote1, label2: numVote2 ...}]
        votes = [{} for k in range(self.n_cluster)]
        for label_i, cluster_i in zip(y, membership):
            if label_i not in votes[cluster_i].keys():
                votes[cluster_i][label_i] = 1
            else:
                votes[cluster_i][label_i] += 1
        # print("votes: ", votes)

        # label each centroid with majority voting from its members: argmax sum rik*I(yi=c)
        centroid_labels = []
        for votes_k in votes:
            # if some centroid doesn’t contain any point, set the label of this centroid as 0.
            if not votes_k:
                centroid_labels.append(0)
            centroid_labels.append(max(votes_k, key=votes_k.get))

        # assign labels to centroids
        centroid_labels = np.array(centroid_labels)
        # print("labels of centroids: ", centroid_labels)

        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    """ predict the same label as the nearest centroid (1-NN on centroids) faster than kNN algorithm"""
    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #      'Implement predict function in KMeansClassifier class')

        # labels = np.zeros(N)
        # dists = -2 * np.dot(x, self.centroids.T) + np.sum(np.square(self.centroids), axis=1) + np.transpose(
        #     [np.sum(np.square(x), axis=1)])
        # membership = np.argmin(dists, axis=1)
        # labels = self.centroid_labels[membership]

        # calculate distance between each point to the nearest centroid
        l2 = np.sum(((x - np.expand_dims(self.centroids, axis=1)) ** 2), axis=2)

        # break ties by choosing the smaller index (similar to using argmin)
        membership = np.argmin(l2, axis=0)

        # assign labels to the points
        labels = self.centroid_labels[membership]

        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)
        
""" idea is simply to treat each pixel of an image as a point  xixi , then perform K-means algorithm to cluster these points, and finally replace each pixel with its centroid"""
def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    # DONOT CHANGE CODE ABOVE THIS LINE
    # raise Exception(
    #          'Implement transform_image function')

    # returns a tuple of number of rows, columns and channels
    D1, D2, _ = image.shape
    K, _ = code_vectors.shape

    # initiate new image at the same size like the original
    new_im = np.zeros(image.shape)

    # loop through each pixel
    for d1 in range(D1):
        for d2 in range(D2):
            dist = np.zeros(K)
            for k in range(K):
                # calculate square distance between each pixel and the code vector
                dist[k] = np.inner(image[d1, d2] - code_vectors[k], image[d1, d2] - code_vectors[k])

            # get the index of minimum distance
            kMin = np.argmin(dist)

            # update the pixel
            new_im[d1, d2] = code_vectors[kMin]

    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im

