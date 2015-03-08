import collections
import cv2
import numpy
import sys

import utils


if len(sys.argv) < 4:
    sys.exit('Usage: ./main.py corpus_path sift_contrast_threshold K')

corpus_path = sys.argv[1]
sift_contrast_threshold = float(sys.argv[2])
K = int(sys.argv[3])

################################################################################
# load the corpus as grey scale, detect its keypoints and compute its descriptors
corpus = cv2.imread(corpus_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
sift = cv2.SIFT(contrastThreshold=sift_contrast_threshold)
# keypoints: the input/output vector of keypoints
# descriptors: the output matrix of descriptors
keypoints, descriptors = sift.detectAndCompute(corpus, None)
keypoints = numpy.array(keypoints)
# utils.draw_keypoints(corpus, keypoints, 'keypoints.png')


# run k-means on descriptors space
max_iter = 10
epsilon = 1.0
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
attempts = 10
# compactness: the sum of squared distance from each point to their corresponding centers
# labels: the label array where each element marked '0', '1', ...
# centers: the array of centers of clusters
compactness, labels, centers = cv2.kmeans(descriptors, K, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)


# construct the codebook of K visual words
codebook = list()
Key = collections.namedtuple('Key', 'centroid features')
Feature = collections.namedtuple('Feature', 'keypoint descriptor')
for i in range(K):
    # for each region found by k-means group keypoints and their descriptors into keys
    centroid = centers[i]

    features = list()
    keypoints_i = keypoints[labels.ravel() == i]  # keypoints of the i-th region
    descriptors_i = descriptors[labels.ravel() == i]  # descriptors of the i-th region
    for keypoint, descriptor in zip(keypoints_i, descriptors_i):
        features.append(Feature(keypoint, descriptor))

    codebook.append(Key(centroid, features))
