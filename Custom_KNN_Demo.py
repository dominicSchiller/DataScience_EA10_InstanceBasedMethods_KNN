__author__ = 'Dominic Schiller'
__email__ = 'dominic.schiller@th-brandenburg.de'
__version__ = '1.0'
__license__ = 'MIT'

# Step (1): Setup the environment
import numpy as np
from sklearn import datasets
from thb.datascience.ibm.KNNClassifier import KNNClassifier

# load the iris data set
iris = datasets.load_iris()

# Step (2): Define the Iris sample which we will classify in a second
iris_sample = np.array([4.8, 2.5, 5.3, 2.4])

# Step (3): Instantiate the custom KNN classifier and classify the sample
knn = KNNClassifier()
knn.k = 10

# perform the classification
predicted_class = knn.classify(iris_sample, iris.data, iris.target)
print('The predicted classification: %.1f' % predicted_class)
print('This means the Iris sample is an "iris-%s"' % iris.target_names[int(predicted_class)])
