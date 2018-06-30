#!/usr/bin/env python3
import argparse
import hashlib
import itertools
import os
import sys

import numpy

import cv2
import utils


def prune_near_keypoints(keypoints, descriptors, pixel_threshold=1):
    pruned_keypoints = list()
    pruned_descriptors = list()
    for i, keypoint1 in enumerate(keypoints):
        to_prune = False
        for keypoint2 in keypoints[i + 1:]:
            dist = abs(numpy.linalg.norm(numpy.asarray(keypoint1.pt) - numpy.asarray(keypoint2.pt)))
            if dist <= pixel_threshold:
                to_prune = True
                break
        if not to_prune:
            pruned_keypoints.append(keypoint1)
            pruned_descriptors.append(descriptors[i])

    return pruned_keypoints, pruned_descriptors


# def geometric_check(match, keypoint_to_add, query_keypoints, query_points_index, squared_tolerance_pixel=30):
# todo: for keypoint in itertools.ifilter(lambda x: x != [], match['keypoints']):
#     for i, match_keypoint in enumerate(match['keypoints']):
#         if match_keypoint != []:
#             query_keypoints_diff = abs(
#                 numpy.linalg.norm(query_keypoints[i] - query_keypoints[query_points_index]))
#             match_keypoints_diff = abs(
#                 numpy.linalg.norm(match_keypoint - keypoint_to_add))
#             if query_keypoints_diff - match_keypoints_diff > squared_tolerance_pixel:
#                 return False
#     return True


def geometric_check(match, keypoint_to_add, query_keypoints, i_q_new, angle_tolerance=20, tolerancesq=5):
    # todo
    c1 = keypoint_to_add.angle < (query_keypoints[i_q_new].angle - angle_tolerance / 2.0) % 360
    c2 = keypoint_to_add.angle > (query_keypoints[i_q_new].angle + angle_tolerance / 2.0) % 360
    if c1 or c2:
        return False

    for i in range(len(match)):
        if match[i]:
            a = numpy.asarray(match[i].pt)
            b = numpy.asarray(keypoint_to_add.pt)
            c = numpy.asarray(query_keypoints[i].pt)
            d = numpy.asarray(query_keypoints[i_q_new].pt)
            if (abs(numpy.linalg.norm(a - b)) - abs(numpy.linalg.norm(c - d))) > tolerancesq:
                return False
    return True


# todo: debug
def compute_match_err(query_keypoints, match):
    match_keypoints = [keypoint for keypoint in match if keypoint]
    if len(match_keypoints) == 0:
        return 0

    match_err = 0
    for match_keypoints_pair in itertools.product(match_keypoints, repeat=2):
        for query_keypoints_pair in itertools.product(query_keypoints, repeat=2):
            a = numpy.asarray(query_keypoints_pair[0].pt)
            b = numpy.asarray(query_keypoints_pair[1].pt)
            query_keypoints_diff = abs(numpy.linalg.norm(a - b))
            c = numpy.asarray(match_keypoints_pair[0].pt)
            d = numpy.asarray(match_keypoints_pair[1].pt)
            match_keypoints_diff = abs(numpy.linalg.norm(c - d))
            match_err += (query_keypoints_diff - match_keypoints_diff)**2
    match_len = sum([1 for keypoint in match if keypoint])
    match_err /= (match_len**2)
    return match_err


################################################################################


def run():
    matches_fingerprint = os.path.basename(args.codebook_filepath)
    matches_fingerprint += os.path.basename(args.query_filepath)
    matches_fingerprint += str(args.n_features)
    matches_fingerprint += str(args.rho)
    matches_fingerprint = matches_fingerprint.encode('ascii')

    matches_filepath = f'matches-{hashlib.sha256(matches_fingerprint).hexdigest()[:7]}.json'
    if os.path.isfile(matches_filepath):
        msg = 'Matches already exist. Overwrite?'
        overwrite = input('{} (y/N) '.format(msg)).lower() in ('y', 'yes')
        if not overwrite: sys.exit(1)

    sift = cv2.xfeatures2d.SIFT_create(nfeatures=args.n_features, nOctaveLayers=1)

    # load the query image as grey scale, detect its keypoints and compute its descriptors
    query_image = cv2.imread(args.query_filepath)
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)

    query_keypoints, query_descriptors = sift.detectAndCompute(query_image, None)
    assert len(query_keypoints) > 0
    assert len(query_keypoints) == len(query_descriptors)
    print(f'Query keypoints count: {len(query_keypoints)}')

    # prune near keypoints
    query_keypoints, query_descriptors = prune_near_keypoints(query_keypoints, query_descriptors)
    print(f'Query keypoints count after pruning: {len(query_keypoints)}')
    print(f'A match must contain at least {int(len(query_keypoints) * args.rho)} keypoints (since rho={args.rho})')

    codebook = utils.load_codebook(args.codebook_filepath)
    num_codebook_keypoints = sum(len(v) for codeword in codebook['codewords'] for v in codeword['keypoints'].values())
    print(f'Codebook keypoints count: {num_codebook_keypoints}')

    # find the nearest codewords to the query descriptors
    print('Finding nearest codewords...')
    query_points = list()
    for i, query_descriptor in enumerate(query_descriptors):
        # find nearest centroid
        query_descriptor_distances = [
            numpy.linalg.norm(query_descriptor - codeword['d']) for codeword in codebook['codewords']
        ]
        nearest_codeword = codebook['codewords'][numpy.argmin(query_descriptor_distances)]

        query_point = {'w': nearest_codeword, 'x': query_keypoints[i]}
        query_points.append(query_point)

    # find matches in each page
    print('Finding matches...')
    matches = dict()
    for page_filename in codebook['pages']:
        # find matches on current page, start with the empty match
        match = []
        page_matches = [match]

        s = 0
        query_keypoints_remaining = len(query_keypoints)
        for i, query_point in enumerate(query_points):
            curr_codeword = query_point['w']
            curr_codeword_keypoints = curr_codeword['keypoints'][page_filename]
            s += len(curr_codeword_keypoints)

            # try to expand current matches
            new_page_matches = []
            for match in page_matches:
                found_better_match = False

                for j, keypoint in enumerate(curr_codeword_keypoints):
                    if geometric_check(match, keypoint, query_keypoints, i):
                        # add the better match to the new matches
                        new_match = match + [keypoint]
                        new_page_matches.append(new_match)
                        found_better_match = True

                if not found_better_match:
                    # check if there are not enough keypoints remaining to mantain the threshold
                    match_len = sum([1 for keypoint in match if keypoint])
                    if match_len + query_keypoints_remaining - 1 >= args.rho * len(query_keypoints):
                        # ok, enough keypoints remaining, keep the old match going
                        new_match = match + [None]
                        new_page_matches.append(new_match)

            query_keypoints_remaining -= 1
            page_matches = new_page_matches

        assert all(len(match) >= args.rho * len(query_keypoints) for match in page_matches)
        print(f'{len(page_matches)} matches in page {page_filename}')
        if (len(page_matches) > 0):
            page_matches_scored = [(match, compute_match_err(query_keypoints, match)) for match in page_matches]
            matches[page_filename] = page_matches_scored

    utils.save_matches(matches, matches_filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('codebook_filepath')
    parser.add_argument('query_filepath')
    parser.add_argument('n_features', type=int)
    parser.add_argument('rho', type=float)
    args = parser.parse_args()
    run()
