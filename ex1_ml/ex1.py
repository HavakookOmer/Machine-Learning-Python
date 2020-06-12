from copy import deepcopy
from load import data_prep
from init_centroids import init_centroids
import matplotlib.pyplot as plt
import numpy as np

NUM_OF_ITER =11

def k_mean(K):
    X = data_prep()
    centroids = init_centroids(X, K)
    clusters = np.zeros(len(X))
    centroids_new = deepcopy(centroids)  # create np.array new centroids

    print("k=" + str(K) + ":")
    counter = []
    for j in range(NUM_OF_ITER):
        count = 0
        for i in range(len(X)):
            distances = np.linalg.norm(X[i] - centroids, axis=-1)
            cluster = np.argmin(distances)
            clusters[i] = cluster
            count = count + distances[cluster]
        for i in range(K):
            centroids_new[i] = np.mean(X[clusters == i], axis=0)
        print ("iter{0}:{1}".format(str(j),print_cent(centroids)))
        centroids = deepcopy(centroids_new)
        counter.append(count)
    final_image = np.zeros((clusters.shape[0], 3))
    for i in range(final_image.shape[0]):
        final_image[i] = centroids[int(clusters[i])]

    # Create Graphs and Image And save them .
    plt.imshow(final_image.reshape(128, 128, 3))
    r=1
    plt.savefig('figure_%d_%d.png' % (K, r))
    plt.figure()
    plt.plot(range(NUM_OF_ITER), counter)
    r+=1
    plt.savefig('figure_%d_%d.png' % (K, r))


def print_cent(cent):
    if type(cent) == list:
        cent = np.asarray(cent)
    if len(cent.shape) == 1:
        return ' '.join(str(np.floor(100*cent)/100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',']').replace(' ', ', ')
    else:
        return ' '.join(str(np.floor(100*cent)/100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',']').replace(' ', ', ')[1:-1]

def main():
    arges = [2, 4, 8, 16]
    for i in arges:
        k_mean(i)


if  __name__ =='__main__':
    main()

