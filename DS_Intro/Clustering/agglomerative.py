import numpy as np
import random
import datetime
import time
import heapq


def find_k_min_distances(clusters, cluster_idx, k):
    # Initialize an empty min-heap
    min_heap = []

    # Iterate through all clusters
    for clstr_idx, clstr in clusters.items():
        # Calculate the distance
        distance = clstr.centroid_distance(clusters[cluster_idx])

        # If we have less than 2*k elements in the heap, push the new element
        if len(min_heap) < k * 2:
            heapq.heappush(min_heap, (-distance, clstr_idx))
        # Otherwise, push the new element and pop the largest one (to maintain the k*2 smallest)
        else:
            heapq.heappushpop(min_heap, (-distance, clstr_idx))

    # Extract the k*2 smallest clusters and return as a dictionary
    closest_clusters = {clstr_idx: clusters[clstr_idx] for _, clstr_idx in min_heap}

    return closest_clusters

class Cluster:
    def __init__(self, indices: np.ndarray, pixels: np.ndarray):
        """
        :param indices: numpy array of indices of the pixels in image (flattened)
        :param pixels: numpy array of pixels in image
        """
        self.indices = indices
        self.pixels = pixels

    def merge(self, other):
        self.indices = np.concatenate([self.indices, other.indices])
        self.pixels = np.vstack([self.pixels, other.pixels])

    def distance(self, other, linkage: str = 'single'):
        if linkage == 'single':
            return np.min(np.sum((self.pixels[:, None] - other.pixels) ** 2, axis=2, dtype=float))
        if linkage == 'average':
            return np.mean(np.linalg.norm(self.pixels[:, None] - other.pixels, axis=2))

    def centroid_distance(self, other):
        return np.sum((np.mean(self.pixels, axis=0) - np.mean(other.pixels, axis=0)) ** 2)


class ImageAgglomerative:
    def __init__(self, num_clusters: int, linkage: str = 'single', sample_size: int = 1, use_distance: bool = False,
                 distance_scale: float = 1.0, random_state: int = None):
        """
        :param num_clusters: Number of clusters
        :param linkage: Linkage criterion to use. Can be 'single' or 'average'
        :param sample_size: Number of samples to use to calculate the pairwise distances.
        :param use_distance: Add the pixel location as a feature to the image
        :param distance_scale: Weight of the pixel location feature relative to the pixel color features
        """
        self.k = num_clusters
        if linkage not in ['single', 'average']:
            raise ValueError("Invalid linkage criterion. Must be 'single', 'complete' or 'average'")
        self.linkage = linkage
        self.sample_size = sample_size
        self.use_distance = use_distance
        self.scale = distance_scale
        self.random_state = random_state
        self.labels = None

    def fit(self, img: np.ndarray):
        """
        Fits the Agglomerative clustering algorithm to the image
        :param img: Numpy array of shape (H, W, C) where H is the height, W is the width and C is the number of channels
        of the image to cluster.
        """
        num_features = img.shape[-1]
        if self.use_distance:
            indices = np.indices(img.shape[:2]).transpose(1, 2, 0)
            img = np.concatenate([img, indices], axis=-1)
            num_features += 2

        x = img.reshape(-1, num_features).copy()
        n = x.shape[0]
        clusters = {i: Cluster(np.array([i]), x[i].reshape(1, -1)) for i in range(n)}
        not_in_cluster = set(range(n))
        idx_map = {i: i for i in range(n)}
        np.random.seed(self.random_state)

        used_indices = set()
        time = datetime.datetime.now()
        for _ in range(n - self.k):
            if len(not_in_cluster) >= self.sample_size:
                subsample = random.sample(list(not_in_cluster), self.sample_size)
                distances = np.sum((x[subsample][:, None] - x) ** 2, axis=2, dtype=float)
                for index, sample_index in enumerate(subsample):
                    distances[index, sample_index] = np.inf  # distance to self

                i, j = np.unravel_index(np.argmin(distances), distances.shape)
                point_index, cluster_index = subsample[i], idx_map[j]
                point = clusters.pop(point_index)
                clusters[cluster_index].merge(point)  # add the point to the cluster
                idx_map[point_index] = cluster_index  # update cluster indices

                not_in_cluster.remove(point_index)
                if j in not_in_cluster:
                    not_in_cluster.remove(j)

            else:
                subsample = random.sample(list(clusters), self.sample_size)
                min_dist = np.inf
                for cluster_idx in subsample:
                    if len(clusters) > (self.k + 1) * 2:
                        closest_clusters = find_k_min_distances(clusters, cluster_idx, (self.k + 1) * 2)
                    else:
                        closest_clusters = clusters
                    for other_cluster_idx, other_cluster in closest_clusters.items():
                        dist = clusters[cluster_idx].distance(other_cluster, linkage=self.linkage)
                        if dist < min_dist:
                            min_dist = dist
                            min_pair = cluster_idx, other_cluster_idx

                cluster, other_cluster = min_pair
                clusters[cluster].merge(clusters.pop(other_cluster))

            if _ % 1000 == 0:
                print(f"Step {_ // 1000}/{(n - self.k) // 1000}")
                print(f"Number of clusters: {len(clusters)}")
                print(f"Time in seconds: {(datetime.datetime.now() - time).seconds}")

                time = datetime.datetime.now()

        self.labels = np.zeros(n, dtype=int)
        for i, cluster in enumerate(clusters.values()):
            for idx in cluster.indices:
                self.labels[idx] = i

    def predict(self, img: np.ndarray):
        """
        Predicts the cluster labels for the image
        :param img: NumPy array of shape (H, W, C) where H is the height, W is the width and C is the number of channels
        of the image to cluster.
        :return: NumPy array of shape (H, W) with the cluster labels for each pixel.
        """
        return self.labels.reshape(img.shape[:2])


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img = plt.imread('star.jpg')

    agglo_dist = ImageAgglomerative(num_clusters=2, linkage='single', use_distance=True, distance_scale=0.5,
                                    random_state=42, sample_size=1)
    agglo_dist.fit(img)
