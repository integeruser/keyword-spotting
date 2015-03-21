#!/usr/bin/env python2 -u
import cv2
import itertools
import numpy
import pickle
import sys


### debug ###
def draw_keypoints(image_file_path, image_keypoints, out_file_name):
    file_image = cv2.imread(image_file_path)
    image_keypoints = [keypoint for keypoint in image_keypoints if keypoint != []]
    for keypoint in image_keypoints:
        pt = (int(keypoint[0]), int(keypoint[1]))
        cv2.circle(file_image, pt, radius=10, color=(0, 0, 255), thickness=-1)
    cv2.imwrite('/Users/fcagnin/Desktop/debug/{0}.png'.format(out_file_name), file_image)
### debug ###


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

def geometric_check(match, keypoint_to_add, query_keypoints, i_q_new, tolerancesq=30):
    for i in range(len(match['keypoints'])):
        if match['keypoints'][i] != []:
            if (abs(numpy.linalg.norm(match['keypoints'][i] - keypoint_to_add)) - abs(numpy.linalg.norm(query_keypoints[i] - query_keypoints[i_q_new]))) > tolerancesq:
                return False
    return True


def find_matches(page_file_name, query_keypoints, index_set, rho):
    match = {'keypoints': [], 'length': 0}
    matches = [match]
    query_keypoints_remaining = len(query_keypoints)

    for i, codeword_keypoints in enumerate(index_set):
        ### debug ###
        # print '   len codeword_keypoints: {0}'.format(len(codeword_keypoints))
        ### debug ###
        new_matches = []

        for match in matches:
            found_better_match = False
            ### debug ###
            better_match_count = 0
            ### debug ###

            for codeword_keypoint in codeword_keypoints:
                if geometric_check(match, codeword_keypoint, query_keypoints, i):
                    # add the better match to the new matches
                    new_match = {
                        'keypoints': match['keypoints'] + [codeword_keypoint],
                        'length': match['length'] + 1
                    }
                    new_matches.append(new_match)
                    found_better_match = True

                    ### debug ###
                    draw_keypoints('/Users/fcagnin/Desktop/pages/{0}'.format(page_file_name),
                                   new_match['keypoints'],
                                   'new_match-{0}-{1}'.format(i, better_match_count))
                    better_match_count += 1
                    ### debug ###

            if not found_better_match:
                if match['length'] + query_keypoints_remaining >= len(query_keypoints) * rho:
                    # keep the old match going
                    new_match = {
                        'keypoints': match['keypoints'] + [[]],
                        'length': match['length']
                    }
                    new_matches.append(new_match)

        query_keypoints_remaining -= 1
        matches = new_matches

        ### debug ###
        # print '   len matches: {0}'.format(len(matches))
        # draw current query keypoint on query
        query_image = cv2.imread(query_file_path)
        pt = (int(query_keypoints[i][0]), int(query_keypoints[i][1]))
        cv2.circle(query_image, pt, radius=10, color=(0, 0, 255), thickness=-1)
        cv2.imwrite('/Users/fcagnin/Desktop/debug/query-{0}.png'.format(i), query_image)

        # draw codeword_keypoints on page
        draw_keypoints(
            '/Users/fcagnin/Desktop/pages/{0}'.format(page_file_name), codeword_keypoints, i)

        # print '   {0}/{1}'.format(i + 1, len(index_set))
        ### debug ###

    return matches


# todo: debug
def compute_match_err(query_keypoints, match):
    match_keypoints = [keypoint for keypoint in match['keypoints'] if keypoint != []]
    if len(match_keypoints) == 0:
        return 0

    match_err = 0
    for match_keypoints_pair in itertools.product(match_keypoints, repeat=2):
        for query_keypoints_pair in itertools.product(query_keypoints, repeat=2):
            query_keypoints_diff = abs(
                numpy.linalg.norm(query_keypoints_pair[0] - query_keypoints_pair[1]))
            match_keypoints_diff = abs(
                numpy.linalg.norm(match_keypoints_pair[0] - match_keypoints_pair[1]))
            match_err += (query_keypoints_diff - match_keypoints_diff) ** 2
    match_err /= (match['length'] ** 2)
    return match_err


# todo: debug
def extract_and_save_matches(page_file_name, page_matches, offset_pixel=15):
    for i in range(len(page_matches)):
        match = page_matches[i]

        page_image = cv2.imread('/Users/fcagnin/Desktop/pages/' + page_file_name)

        # keep only non-empy keypoints
        match_keypoints = [keypoint for keypoint in match['keypoints'] if keypoint != []]

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


if len(sys.argv) != 5:
    sys.exit('Usage: {0} query_file_path codebook_file_path n_features rho'.format(sys.argv[0]))

query_file_path = sys.argv[1]
codebook_file_path = sys.argv[2]
n_features = int(sys.argv[3])
rho = float(sys.argv[4])
print 'Starting script...'
print '   {0: <18} = {1}'.format('query_file_path', query_file_path)
print '   {0: <18} = {1}'.format('codebook_file_path', codebook_file_path)
print '   {0: <18} = {1}'.format('n_features', n_features)
print '   {0: <18} = {1}'.format('rho', rho)

################################################################################

# load the query as grey scale, detect its keypoints and compute its descriptors
print 'Detecting keypoints and computing descriptors on \'{0}\'...'.format(query_file_path)
query_image = cv2.imread(query_file_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

sift = cv2.SIFT(nfeatures=n_features)
query_keypoints, query_descriptors = sift.detectAndCompute(query_image, None)

### debug ###
cv2.imwrite('/Users/fcagnin/Desktop/debug/query.png',
            cv2.drawKeypoints(query_image, query_keypoints))
### debug ###

query_keypoints = numpy.asarray([keypoint.pt for keypoint in query_keypoints])
print '   Keypoints detected: {0}'.format(len(query_keypoints))
# (sometimes the keypoints extrated are one more that requested)
assert len(query_keypoints) <= n_features + 1


# find the nearest codewords to the query descriptors
print 'Loading the codebook...'
with open(codebook_file_path, 'rb') as f:
    codebook = pickle.load(f)

print 'Finding the nearest codewords to the query...'
query_points = list()
for i, query_descriptor in enumerate(query_descriptors):
    # find nearest centroid
    query_descriptor_distances = [
        numpy.linalg.norm(query_descriptor - codeword['d']) for codeword in codebook]
    nearest_codeword_index = numpy.argmin(query_descriptor_distances)

    query_point = {
        'w': nearest_codeword_index,
        'x': query_keypoints[i]
    }
    query_points.append(query_point)


# find and extract matches
k = 3
print 'Finding and extracting top-{0} matches in each page...'.format(k)

pages = codebook[0]['features'].keys()  # todo: rework
for page in pages:
    index_set = list()
    for query_point in query_points:
        codeword = codebook[query_point['w']]
        index_set.append(codeword['features'][page])

    page_matches = find_matches(page, query_keypoints, index_set, rho)
    print '   Found {0} matches in page {1}'.format(len(page_matches), page)

    if (len(page_matches) > 0):
        page_matches_scores = [
            compute_match_err(query_keypoints, page_match) for page_match in page_matches]
        top_k_page_matches = [x for (y, x) in sorted(zip(page_matches_scores, page_matches))][:k]
        extract_and_save_matches(page, top_k_page_matches)
        ### debug ###
        exit()
        ### debug ###
