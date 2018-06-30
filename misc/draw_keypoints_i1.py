#!/usr/bin/env python3
import argparse

import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image1_filepath')
    parser.add_argument('contrast_threshold', type=float)
    parser.add_argument('n_octave_layers', type=int)
    args = parser.parse_args()

    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=args.contrast_threshold, nOctaveLayers=args.n_octave_layers)

    image1 = cv2.imread(args.image1_filepath)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    keypoints1 = sift.detect(image1, None)

    cv2.imshow('image1', cv2.drawKeypoints(image1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
    cv2.waitKey()
