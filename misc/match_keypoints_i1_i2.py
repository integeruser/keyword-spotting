#!/usr/bin/env python3
import argparse

import numpy
import matplotlib.pyplot

import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image1_filepath')
    parser.add_argument('query_filepath')
    parser.add_argument('contrast_threshold', type=float)
    parser.add_argument('n_octave_layers', type=int)
    args = parser.parse_args()

    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=args.contrast_threshold, nOctaveLayers=args.n_octave_layers)

    img1 = cv2.imread(args.query_filepath)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread(args.image1_filepath)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    matplotlib.pyplot.imshow(img3)
    matplotlib.pyplot.show()
