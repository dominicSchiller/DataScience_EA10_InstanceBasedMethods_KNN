import numpy as np
from numpy import ndarray

__author__ = 'Dominic Schiller'
__email__ = 'dominic.schiller@th-brandenburg.de'
__version__ = '1.0'
__license__ = 'MIT'

class KNNClassifier:
    """
    Custom implementation of an KNN classifier (k-nearest neighbor)
    """

    def __init__(self):
        """
        Constructor
        """
        self._k = 1

    def set_k(self, k):
        self._k = k

    def classify(self, sample: ndarray, features: ndarray, classes: ndarray):
        """
        Classify the given sample based on the features and the given classes
        :param sample: The sample set to classify
        :param features: List of features calculate the classification index from
        :param classes: List of classifications to look up the predicted class from
        :return: The predicted classification matching the given sample set
        """
        # (1) normalize all features and the sample
        normalized_data = self._normalize(features, sample)
        # (2) calculate all distances between the sample and all features
        distances = self._calc_distances(normalized_data[1], normalized_data[0])
        # (3) aggregate distances with all associated classifications
        aggregated_distances = np.column_stack((classes, distances))
        # (4) sort the aggregated distances by the distance only
        aggregated_distances = aggregated_distances[aggregated_distances[:, 1].argsort()]
        # (5) select the k-nearest neighbors
        k_nereast_neighbors = aggregated_distances[0:self._k, :]
        # (6) return the maximum from the selected k-nearest neighbors
        return k_nereast_neighbors[
            np.where(
                k_nereast_neighbors[:, 1] == np.max(k_nereast_neighbors[:, 1])
            )
        ][0][0]

    @staticmethod
    def _normalize(features: ndarray, sample: ndarray) -> (ndarray, ndarray):
        """
        Normalize all features as well as the sample set
        :param features: List with features to normalize
        :param sample: The sample set to normalize
        :return: Tuple with normalized features and the normalized sample set
        """
        features_normalized = np.copy(features)
        sample_normalized = np.copy(sample)

        for i in range(0, features_normalized.shape[1]):
            feature_data = features_normalized[:, i:i + 1]

            # calculate minimum and maximum per feature
            minimum = np.min(feature_data)
            maximum = np.max(feature_data)

            # calc normalized sample feature
            sn = (sample[i] - minimum) / (maximum - minimum)
            sample_normalized[i] = sn

            # do the normalization
            for j in range(0, features.shape[0]):
                # calc normalized feature
                fn = (feature_data[j] - minimum) / (maximum - minimum)
                # replacing the original value
                feature_data[j] = fn

        return features_normalized, sample_normalized

    @staticmethod
    def _calc_distances(sample, features) -> ndarray:
        """
        Calculate euclidean distances between the sample set and all features
        :param sample: The sample set to calculate the distance with
        :param features: List with features to calculate the distance from
        :return: Array containing all calculated distances
        """
        distances = np.empty(0)
        powers = np.power(np.subtract(sample, features), 2)

        for i in range(0, powers.shape[0]):
            d = np.sqrt(np.sum(powers[i]))
            distances = np.append(distances, d)

        return distances
