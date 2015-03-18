#!/usr/bin/env python -u
import cv2
import itertools
import numpy
import pickle
import sys


class Match:
    keypoints = []
    length = 0

    def __str__(self):
        return 'keypoints: {0}, length: {1}'.format(self.keypoints, self.length)

    def __repr__(self):
        return 'keypoints: {0}, length: {1}'.format(self.keypoints, self.length)


def extract_and_save_matches(page_file_name, page_matches, max_extracted_matches=5, offset_pixel=15):
    for i in range(min(len(page_matches), max_extracted_matches)):
        match = page_matches[i]

        page_image = cv2.imread(
            '/Users/fcagnin/Documents/2015-ICIAP-keywordSpotting/gw_20p_wannot_pages/' + page_file_name)

        # keep only non-empy keypoints
        match_keypoints = [keypoint for keypoint in match.keypoints if keypoint != []]

        # draw keypoints on original image
        for keypoint in match_keypoints:
            pt = (int(keypoint[0]), int(keypoint[1]))
            cv2.circle(page_image, pt, radius=5, color=(0, 0, 255))

        # extract and save match from original image
        # find max and min keypoints
        x_min = numpy.min(numpy.asarray(match_keypoints)[:, 0]) - offset_pixel
        y_min = numpy.min(numpy.asarray(match_keypoints)[:, 1]) - offset_pixel
        x_max = numpy.max(numpy.asarray(match_keypoints)[:, 0]) + offset_pixel
        y_max = numpy.max(numpy.asarray(match_keypoints)[:, 1]) + offset_pixel

        extracted_match = page_image[y_min:y_max, x_min:x_max]
        match_file_path = '/Users/fcagnin/Desktop/output/{0}-{1}.png'.format(page_file_name, i)
        cv2.imwrite(match_file_path, extracted_match)


# def distance_check(q1, k1, q2, k2, squared_e=0):
# import math
#     return abs(numpy.linalg.norm(q1 - q2)) - abs(numpy.linalg.norm(k1 - k2)) <= squared_e
# return abs((q2[0] - q1[0]) ** 2 + (q2[1] - q1[1]) ** 2) - abs((k2[0] -
# k1[0]) ** 2 + (k2[1] - k1[1]) ** 2) <= squared_e
# return abs(math.hypot(q2[0] - q1[0], q2[1] - q1[1])) -
# abs(math.hypot(k2[0] - k1[0], k2[1] - k1[1])) <= squared_e


# def geometry_check(query_keypoints, match, new_keypoint):
#     for pair in itertools.product(query_keypoints, repeat=2):
#         for k2 in match:
#             if not distance_check(pair[0], new_keypoint, pair[1], k2):
#                 return False
#     return True


def geometric_check(match, keypoint_to_add, query_keypoints, i_q_new, tolerancesq=20):
    for i in range(len(match.keypoints)):
        if match.keypoints[i] != []:
            if (abs(numpy.linalg.norm(match.keypoints[i] - keypoint_to_add)) - abs(numpy.linalg.norm(query_keypoints[i] - query_keypoints[i_q_new]))) > tolerancesq:
                return False
    return True


def find_matches(query_keypoints, page, index_set, rho):
    # print 'index_set length: {0}'.format([len(l) for l in index_set])
    m = Match()
    matches = [m]
    query_keypoints_remaining = len(query_keypoints)

    for i, iq in enumerate(index_set):
        # draw_keypoints(query, query_keypoints, page, iq)

        new_matches = []
        # print '   len matches: {0}'.format(len(matches))
        # print '   query_keypoints_remaining: {0}'.format(query_keypoints_remaining)

        for match in matches:
            # for each keypoint, check if a new match can be created
            added = 0
            for keypoint in iq:
                if geometric_check(match, keypoint, query_keypoints, i):
                    # add the new match
                    new_match = Match()
                    new_match.keypoints = match.keypoints + [keypoint]
                    new_match.length = match.length + 1
                    new_matches.append(new_match)
                    added += 1
            # print '   new matches added: {0}/{1}'.format(added, len(iq))

            if match.length + query_keypoints_remaining >= len(query_keypoints) * rho:
                # print '      old match added: {0}, {1}, {2}, {3}'.format(match,
                # match.length, query_keypoints_remaining, len(query_keypoints))
                new_match = Match()
                new_match.keypoints = match.keypoints + [[]]
                new_match.length = match.length
                new_matches.append(new_match)
            # else:
                # print '      old match discarded: ' + str(match)

        query_keypoints_remaining -= 1
        matches = new_matches

    # print 'find_matches returns {0}'.format(matches)
    return matches


if len(sys.argv) != 4:
    sys.exit('Usage: {0} query_file_path codebook_file_path n_features'.format(sys.argv[0]))

query_file_path = sys.argv[1]
codebook_file_path = sys.argv[2]
n_features = int(sys.argv[3])
print 'Starting script...'
print '   {0: <18} = {1}'.format('query_file_path', query_file_path)
print '   {0: <18} = {1}'.format('codebook_file_path', codebook_file_path)
print '   {0: <18} = {1}'.format('n_features', n_features)

################################################################################

# load the query as grey scale, detect its keypoints and compute its descriptors
print 'Detecting keypoints and computing descriptors on \'{0}\'...'.format(query_file_path)
query = cv2.imread(query_file_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

sift = cv2.SIFT(nfeatures=n_features)
query_keypoints, query_descriptors = sift.detectAndCompute(query, None)

# save query keypoints for later comparison
cv2.imwrite('/Users/fcagnin/Desktop/output/query.png',
            cv2.drawKeypoints(query, query_keypoints))

query_keypoints = numpy.asarray([keypoint.pt for keypoint in query_keypoints])
print '   Keypoints detected: {0}'.format(len(query_keypoints))
# (sometimes the keypoints extrated are one more that requested)
assert len(query_keypoints) <= n_features + 1


# find the nearest codewords to the query descriptors
print 'Loading the codebook...'
with open(codebook_file_path, 'rb') as f:
    codebook = pickle.load(f)

print 'Finding the nearest codewords...'
q = list()
for i, query_descriptor in enumerate(query_descriptors):
    # find nearest centroid
    query_descriptor_distances = [
        numpy.linalg.norm(query_descriptor - codeword['d']) for codeword in codebook]
    nearest_codeword_index = numpy.argmin(query_descriptor_distances)

    q.append({'w': nearest_codeword_index, 'x': query_keypoints[i]})


# find and extract matches
print 'Finding and extracting matches...'
matches = dict()

pages = codebook[0]['features'].keys()
for page in pages:
    index_set = list()
    for i in range(len(q)):
        codeword_index = q[i]['w']
        codeword = codebook[codeword_index]
        index_set.append(codeword['features'][page])

    matches[page] = find_matches(query_keypoints, page, index_set, rho=0.8)
    print '   Found {0} matches in page {1}'.format(len(matches[page]), page, )

    extract_and_save_matches(page, matches[page])
