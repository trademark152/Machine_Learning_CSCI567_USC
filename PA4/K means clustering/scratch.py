import numpy as np
centroids = np.array([[1, 2],[2, 3],[1,1]])
x = np.array([[1, 1],[2, 2],[3, 4],[1, 2],[1, 2],[2, 3]])

distances = np.sum(np.square(centroids), axis=1) - 2 * np.dot(x, centroids.T) + np.transpose([np.sum(np.square(x), axis=1)])
l2 = np.sum(((x - np.expand_dims(centroids, axis=1)) ** 2), axis=2)

print(centroids)
print(np.sum(np.square(centroids), axis=1))
print(np.dot(x, centroids.T))
print(np.transpose([np.sum(np.square(x), axis=1)]))
print(distances)


print(np.expand_dims(centroids, axis=1))
print(l2)