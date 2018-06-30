#!/usr/bin/env python2 -u
import cv2
import itertools
import numpy
import sys
import utils


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
    # match_keypoints = [keypoint for keypoint in match if keypoint]
    # if len(match_keypoints) == 0:
    #     return 0

    # match_err = 0
    # for match_keypoints_pair in itertools.product(match_keypoints, repeat=2):
    #     for query_keypoints_pair in itertools.product(query_keypoints, repeat=2):
    #         a = numpy.asarray(query_keypoints_pair[0].pt)
    #         b = numpy.asarray(query_keypoints_pair[1].pt)
    #         query_keypoints_diff = abs(
    #             numpy.linalg.norm(a - b))
    #         c = numpy.asarray(match_keypoints_pair[0].pt)
    #         d = numpy.asarray(match_keypoints_pair[1].pt)
    #         match_keypoints_diff = abs(
    #             numpy.linalg.norm(c - d))
    #         match_err += (query_keypoints_diff - match_keypoints_diff) ** 2
    # match_len = sum([1 for keypoint in match if keypoint])
    # match_err /= (match_len ** 2)
    # return match_err
    return 0

################################################################################

if len(sys.argv) != 6:
    sys.exit('Usage: {0} codebook_file_path query_file_path n_features rho'.format(sys.argv[0]))

codebook_file_path = sys.argv[1]
query_file_path = sys.argv[2]
contrast_threshold = float(sys.argv[3])
n_octave_layers = int(sys.argv[4])
rho = float(sys.argv[5])
print 'Starting script...'
print '   {0: <20} = {1}'.format('codebook_file_path', codebook_file_path)
print '   {0: <20} = {1}'.format('query_file_path', query_file_path)
print('   {0: <20} = {1}'.format('contrast_threshold', contrast_threshold))
print('   {0: <20} = {1}'.format('n_octave_layers', n_octave_layers))
print '   {0: <20} = {1}'.format('rho', rho)

################################################################################

# load the query as grey scale, detect its keypoints and compute its descriptors
print 'Detecting keypoints and computing descriptors on \'{0}\'...'.format(query_file_path)
sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=contrast_threshold, nOctaveLayers=n_octave_layers)

query_image = cv2.imread(query_file_path)
query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)

query_keypoints, query_descriptors = sift.detectAndCompute(query_image, None)
assert len(query_keypoints) > 0
assert len(query_keypoints) == len(query_descriptors)

print '   Keypoints detected: {0}'.format(len(query_keypoints))


# load the codebook
print 'Loading the codebook...'
codebook = utils.load_codebook(codebook_file_path)
print '   Codebook keypoints count: {0}'.format(sum([len(v) for codeword in codebook['codewords']
                                                     for v in codeword['keypoints'].viewvalues()]))

# find the nearest codewords to the query descriptors
print 'Finding the nearest codewords to the query...'
query_points = list()
for i, query_descriptor in enumerate(query_descriptors):
    # find nearest centroid
    query_descriptor_distances = [
        numpy.linalg.norm(query_descriptor - codeword['d']) for codeword in codebook['codewords']]
    nearest_codeword = codebook['codewords'][numpy.argmin(query_descriptor_distances)]

    query_point = {
        'w': nearest_codeword,
        'x': query_keypoints[i]
    }
    query_points.append(query_point)

### debug ###
utils.save_debug(codebook, query_points)
### debug ###


# find matches in each page
print 'Finding matches in each page...'
matches = dict()
for page in codebook['pages']:
    # find matches on current page
    match = []
    page_matches = [match]  # start with empty match

    query_keypoints_remaining = len(query_keypoints)
    for i, query_point in enumerate(query_points):
        curr_codeword = query_point['w']
        curr_codeword_keypoints = curr_codeword['keypoints'][page]

        # (try to) expand current matches
        new_page_matches = []
        for match in page_matches:
            found_better_match = False

            for j, keypoint in enumerate(curr_codeword_keypoints):
                if geometric_check(match, keypoint, query_keypoints, i):
                    # add the better match to new matches
                    new_match = match + [keypoint]
                    new_page_matches.append(new_match)
                    found_better_match = True

            if not found_better_match:
                match_len = sum([1 for keypoint in match if keypoint])
                if match_len + query_keypoints_remaining >= len(query_keypoints) * rho:
                    # keep the old match going
                    new_match = match + [None]
                    new_page_matches.append(new_match)

        query_keypoints_remaining -= 1
        page_matches = new_page_matches

    print '   Found {0} matches in page {1}'.format(len(page_matches), page)
    if (len(page_matches) > 0):
        page_matches_scored = [
            (match, compute_match_err(query_keypoints, match)) for match in page_matches]
        matches[page] = page_matches_scored


# save matches to file
utils.save_matches(matches, 'matches.json')
