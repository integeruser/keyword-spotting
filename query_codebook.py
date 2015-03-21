#!/usr/bin/env python2 -u
import cPickle
import cv2
import itertools
import numpy
import sys
import utils


import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


### debug ###
def draw_barplot(descriptor, query_desc, out_file_name):
    fig, ax = plt.subplots()
    w = 1
    ind1 = numpy.arange(0, 128 * (4 + w), 4 + w)
    ind2 = numpy.arange(w, 128 * (4 + w), 4 + w)
    ax.bar(ind1, descriptor, width=w, color='r', edgecolor='none')
    ax.bar(ind2, query_desc, width=w, color='b', edgecolor='none')
    plt.savefig(
        '/Users/fcagnin/Desktop/debug/{0}.jpg'.format(out_file_name))
    # plt.show()
    # exit()
    # plt.close(fig)
    # fig = plt.figure()
    # ax = plt.subplot(111)
    # ind = numpy.arange(len(descriptor))
    # ax.bar(ind, descriptor, width=1, color=col)
    # plt.savefig('/Users/fcagnin/Desktop/debug/{0}.png'.format(out_file_name))
    # plt.close(fig)


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

def geometric_check(match, keypoint_to_add, query_keypoints, i_q_new, angle_tolerance=40, tolerancesq=20):
    c1 = keypoint_to_add.angle < (query_keypoints[i_q_new].angle - angle_tolerance / 2) % 360
    c2 = keypoint_to_add.angle > (query_keypoints[i_q_new].angle + angle_tolerance / 2) % 360
    if c1 or c2:
        return False

    for i in range(len(match['keypoints'])):
        if match['keypoints'][i] != []:
            a = numpy.asarray(match['keypoints'][i].pt)
            b = numpy.asarray(keypoint_to_add.pt)
            c = numpy.asarray(query_keypoints[i].pt)
            d = numpy.asarray(query_keypoints[i_q_new].pt)
            if (abs(numpy.linalg.norm(a - b)) - abs(numpy.linalg.norm(c - d))) > tolerancesq:
                return False
    return True


# todo: debug
def compute_match_err(query_keypoints, match):
    match_keypoints = [keypoint for keypoint in match['keypoints'] if keypoint != []]
    if len(match_keypoints) == 0:
        return 0

    match_err = 0
    for match_keypoints_pair in itertools.product(match_keypoints, repeat=2):
        for query_keypoints_pair in itertools.product(query_keypoints, repeat=2):
            a = numpy.asarray(query_keypoints_pair[0].pt)
            b = numpy.asarray(query_keypoints_pair[1].pt)
            query_keypoints_diff = abs(
                numpy.linalg.norm(a - b))
            c = numpy.asarray(match_keypoints_pair[0].pt)
            d = numpy.asarray(match_keypoints_pair[1].pt)
            match_keypoints_diff = abs(
                numpy.linalg.norm(c - d))
            match_err += (query_keypoints_diff - match_keypoints_diff) ** 2
    match_err /= (match['length'] ** 2)
    return match_err


# todo: debug
def extract_and_save_matches(page_file_name, matches, offset_pixel=25):
    for i in range(len(matches)):
        match = matches[i]

        page_image = cv2.imread('{0}/{1}'.format(pages_directory_path, page_file_name))

        # keep only non-empy keypoints
        match_keypoints = [keypoint for keypoint in match['keypoints'] if keypoint != []]

        # draw keypoints on original image
        x_min = float('inf')
        y_min = float('inf')
        x_max = 0
        y_max = 0
        for keypoint in match_keypoints:
            x = int(keypoint.pt[0])
            y = int(keypoint.pt[1])
            if x < x_min:
                x_min = x
            if x > x_max:
                x_max = x
            if y < y_min:
                y_min = y
            if y > y_max:
                y_max = y
            cv2.circle(page_image, (x, y), radius=5, color=(0, 0, 255))

        x_min += - offset_pixel
        y_min += - offset_pixel
        x_max += + offset_pixel
        y_max += + offset_pixel

        extracted_match = page_image[y_min:y_max, x_min:x_max]
        match_file_path = '/Users/fcagnin/Desktop/output/{0}-{1}.png'.format(page_file_name, i)
        cv2.imwrite(match_file_path, extracted_match)

################################################################################

if len(sys.argv) != 6:
    sys.exit(
        'Usage: {0} pages_directory_path query_file_path codebook_file_path n_features rho'.format(sys.argv[0]))

pages_directory_path = sys.argv[1]
query_file_path = sys.argv[2]
codebook_file_path = sys.argv[3]
n_features = int(sys.argv[4])
rho = float(sys.argv[5])
print 'Starting script...'
print '   {0: <20} = {1}'.format('pages_directory_path', pages_directory_path)
print '   {0: <20} = {1}'.format('query_file_path', query_file_path)
print '   {0: <20} = {1}'.format('codebook_file_path', codebook_file_path)
print '   {0: <20} = {1}'.format('n_features', n_features)
print '   {0: <20} = {1}'.format('rho', rho)

################################################################################

# load the query as grey scale, detect its keypoints and compute its descriptors
print 'Detecting keypoints and computing descriptors on \'{0}\'...'.format(query_file_path)
sift = cv2.SIFT(nfeatures=n_features, nOctaveLayers=1)

query_image = cv2.imread(query_file_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
query_keypoints, query_descriptors = sift.detectAndCompute(query_image, None)
assert len(query_keypoints) > 0
assert len(query_keypoints) == len(query_descriptors)

print '   Keypoints detected: {0}'.format(len(query_keypoints))
# (sometimes the keypoints detected are one more that requested)
assert len(query_keypoints) <= n_features + 1

### debug ###
cv2.imwrite('/Users/fcagnin/Desktop/debug/query.png',
            cv2.drawKeypoints(query_image, query_keypoints))
### debug ###


# load the codebook
print 'Loading the codebook...'
with open(codebook_file_path, 'rb') as f:
    codebook = cPickle.load(f)

# convert back the serialized keypoints to KeyPoint instances
pages = codebook[0]['keypoints'].keys()  # todo: rework
for codeword in codebook:
    for page in pages:
        codeword['keypoints'][page] = utils.deserialize_keypoints(codeword['keypoints'][page])


# find the nearest codewords to the query descriptors
print 'Finding the nearest codewords to the query...'
query_points = list()
for i, query_descriptor in enumerate(query_descriptors):
    # find nearest centroid
    query_descriptor_distances = [
        numpy.linalg.norm(query_descriptor - codeword['d']) for codeword in codebook]
    nearest_codeword = codebook[numpy.argmin(query_descriptor_distances)]

    query_point = {
        'w': nearest_codeword,
        'x': query_keypoints[i]
    }
    query_points.append(query_point)


# find and extract matches in each page
k = 3
print 'Finding and extracting top-{0} matches in each page...'.format(k)
for page in pages:
    # extract codewords keypoints on the current page
    # index_set = list()
    # for query_point in query_points:
    #     codeword = query_point['w']
    #     index_set.append({
    #         'page_keypoints': codeword['keypoints'][page],
    #         'page_descriptors': codeword['descriptors'][page]
    #     })

    # find matches on current page
    match = {'keypoints': [], 'length': 0}
    matches = [match]  # start with empty match

    query_keypoints_remaining = len(query_keypoints)
    for i, query_point in enumerate(query_points):
        curr_codeword = query_point['w']
        curr_codeword_keypoints = curr_codeword['keypoints'][page]
        ### debug ###
        # print '   len codeword_keypoints: {0}'.format(len(curr_codeword_keypoints))
        ### debug ###

        new_matches = []
        for match in matches:
            found_better_match = False
            ### debug ###
            better_match_count = 0
            ### debug ###

            for j, keypoint in enumerate(curr_codeword_keypoints):
                if geometric_check(match, keypoint, query_keypoints, i):
                    # add the better match to the new matches
                    new_match = {
                        'keypoints': match['keypoints'] + [keypoint],
                        'length': match['length'] + 1
                    }
                    new_matches.append(new_match)
                    found_better_match = True

                    ### debug ###
                    # draw_keypoints('/Users/fcagnin/Desktop/pages/{0}'.format(page_file_name),
                    #                new_match['keypoints'],
                    #                'new_match-{0}-{1}'.format(i, better_match_count))
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
        # if i == 10:
        #     query_image = cv2.imread(query_file_path)
        #     pt = (int(query_keypoints[i][0]), int(query_keypoints[i][1]))
        #     cv2.circle(query_image, pt, radius=3, color=(0, 0, 255), thickness=1)
        #     cv2.imwrite('/Users/fcagnin/Desktop/debug/query-{0}.png'.format(i), query_image)

        #     # draw codeword_keypoints on page
        #     draw_keypoints(
        #         '{0}/{1}'.format(pages_directory_path, page_file_name), codeword_keypoints, i)

        # print '   {0}/{1}'.format(i + 1, len(index_set))
        ### debug ###

    print '   Found {0} matches in page {1}'.format(len(matches), page)
    if (len(matches) > 0):
        matches_scores = [
            (page_match, compute_match_err(query_keypoints, page_match)) for page_match in matches]

        top_k_matches = sorted(matches_scores, key=lambda x: x[1])[:k]
        extract_and_save_matches(page, [p[0] for p in top_k_matches])
