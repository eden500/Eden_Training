import numpy as np


class ImageKmeans:
    def __init__(self, num_clusters: int, max_iter: int = 100, use_distance: bool = False, random_state: int = None,
                 distance_scale: float = 1.0):
        """
        :param num_clusters: Number of clusters
        :param max_iter: Maximum number of iterations to run the algorithm before stopping or convergence
        :param use_distance: Add the pixel location as a feature to the image
        :param random_state: Seed for the random centroid initialization
        :param distance_scale: Weight of the pixel location feature relative to the pixel color features
        """
        self.k = num_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.use_distance = use_distance
        if random_state is not None:
            np.random.seed(random_state)
        self.distance_scale = distance_scale

    def fit(self, img: np.ndarray):
        """
        Fits the K-means algorithm to the image
        :param img: Numpy array of shape (H, W, C) where H is the height, W is the width and C is the number of channels
        of the image to cluster.
        """
        num_features = img.shape[-1]
        if self.use_distance:  # Adds the pixel location as feature
            indices = np.indices(img.shape[:2]).transpose(1, 2, 0) * self.distance_scale
            img = np.concatenate([img, indices], axis=-1)
            num_features += 2

        x = img.reshape(-1, num_features)
        self.centroids = x[np.random.choice(x.shape[0], self.k, replace=False)]

        for _ in range(self.max_iter):
            labels = np.argmin(np.linalg.norm(x[:, None] - self.centroids, axis=2), axis=1)
            new_centroids = np.array([x[labels == i].mean(axis=0) for i in range(self.k)])
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

    def predict(self, img: np.ndarray):
        """
        Predicts the cluster labels for the image
        :param img: NumPy array of shape (H, W, C) where H is the height, W is the width and C is the number of channels
        of the image to cluster.
        :return: NumPy array of shape (H, W) with the cluster labels for each pixel.
        """
        num_features = img.shape[-1]
        if self.use_distance:  # Adds the pixel location as feature
            indices = np.indices(img.shape[:2]).transpose(1, 2, 0)
            img = np.concatenate([img, indices], axis=-1)
            num_features += 2
        x = img.reshape(-1, num_features)
        return np.argmin(np.linalg.norm(x[:, None] - self.centroids, axis=2), axis=1).reshape(img.shape[:2])
