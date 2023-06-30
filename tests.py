import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from hw6 import get_random_centroids, kmeans, kmeans_pp, calc_inertia




image = io.imread('data/small_duck.jpg')
rows = image.shape[0]
cols = image.shape[1]
image = image.reshape(image.shape[0]*image.shape[1],3)
k_means = []
k_plus_means = []
iterations_num = 10
for i in range(10):
    regular_centroids, regular_classes = kmeans(image, k=i + 1, p=1, max_iter=100)
    regular_inertia = calc_inertia(image, regular_centroids, regular_classes)
    k_means.append(regular_inertia)
    regular_classes = regular_classes.reshape(rows,cols)

    plus_centroids, plus_classes = kmeans_pp(image, k=i + 1, p=1, max_iter=100)
    plus_inertia = calc_inertia(image, plus_centroids, plus_classes)
    k_plus_means.append(plus_inertia)
    plus_classes = plus_classes.reshape(rows,cols)


# Plot the results
plt.plot(range(1, len(k_means) + 1),k_means,  marker='o',label='k-means' )
plt.plot(range(1, len(k_plus_means) + 1),  k_plus_means, marker='o', label='k-means++')
plt.xlabel('K value')
plt.ylabel('Inertia value')
plt.legend()
plt.show()