#!/usr/bin/env python3
import collections
import json
import os
import pickle
import re
import sys

import cv2

KeyPoint = collections.namedtuple('KeyPoint', 'pt size angle response octave')


def cv2_to_namedtuple_keypoints(keypoints):
    return [
        KeyPoint(keypoint.pt, keypoint.size, keypoint.angle, keypoint.response, keypoint.octave)
        for keypoint in keypoints
    ]


def namedtuple_keypoints_to_cv2(keypoints):
    return [
        cv2.KeyPoint(
            x=keypoint[0][0],
            y=keypoint[0][1],
            _size=keypoint[1],
            _angle=keypoint[2],
            _response=keypoint[3],
            _octave=keypoint[4]) for keypoint in keypoints
    ]


def tuple_to_namedtuple_keypoints(keypoints):
    return [KeyPoint._make(keypoint) for keypoint in keypoints]


def matches_to_namedtuple_keypoints(keypoints):
    return [KeyPoint._make(keypoint) if keypoint else None for keypoint in keypoints]


################################################################################


def save_corpus(corpus, corpus_filepath):
    print(f'Saving: {corpus_filepath}')
    with open(corpus_filepath, 'wb') as f:
        pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_corpus(corpus_filepath):
    print(f'Loading: {corpus_filepath}')
    with open(corpus_filepath, 'rb') as f:
        corpus = pickle.load(f)

    corpus['keypoints'] = [tuple_to_namedtuple_keypoints(page_keypoints) for page_keypoints in corpus['keypoints']]
    return corpus


def save_codebook(codebook, codebook_filepath):
    print(f'Saving: {codebook_filepath}')
    with open(codebook_filepath, 'wb') as f:
        pickle.dump(codebook, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_codebook(codebook_filepath):
    print(f'Loading: {codebook_filepath}')
    with open(codebook_filepath, 'rb') as f:
        codebook = pickle.load(f)

    for codeword in codebook['codewords']:
        for page in codeword['keypoints']:
            codeword['keypoints'][page] = tuple_to_namedtuple_keypoints(codeword['keypoints'][page])
    return codebook


def save_matches(matches, matches_filepath):
    print(f'Saving: {matches_filepath}')
    with open(matches_filepath, 'w') as f:
        json.dump(matches, f, indent=4)


def load_matches(matches_filepath):
    print(f'Loading: {matches_filepath}')
    with open(matches_filepath) as f:
        matches = json.load(f)

    for page in matches:
        for i, match_scored in enumerate(matches[page]):
            match = match_scored[0]
            score = match_scored[1]
            matches[page][i] = (matches_to_namedtuple_keypoints(match), score)
    return matches
