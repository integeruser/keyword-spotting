#!/usr/bin/env python2 -u
import cv2
import json
import sys
import utils


if len(sys.argv) != 3:
    sys.exit('Usage: {0} pages_directory_path query_file_path'.format(sys.argv[0]))

pages_directory_path = sys.argv[1]
query_file_path = sys.argv[2]
print 'Starting script...'
print '   {0: <19} = {1}'.format('pages_directory_path', pages_directory_path)
print '   {0: <19} = {1}'.format('query_file_path', query_file_path)

################################################################################

with open('debug.json') as f:
    debug = json.load(f)

query_image = cv2.imread(query_file_path)

page_index = 0
query_keypoint_index = 0
while True:
    page = debug.keys()[page_index]

    query_keypoint = debug[page][query_keypoint_index][0]
    codeword_keypoints = debug[page][query_keypoint_index][1]

    page_image = cv2.imread('{0}/{1}'.format(pages_directory_path, page))

    cv2.imshow('codeword-keypoints{0}-{1}'.format(page_index, query_keypoint_index),
               cv2.drawKeypoints(page_image, utils.namedtuple_keypoints_to_cv2(codeword_keypoints),
                                 None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))

    cv2.imshow('query-keypoints{0}-{1}'.format(page_index, query_keypoint_index),
               cv2.drawKeypoints(query_image, utils.namedtuple_keypoints_to_cv2([query_keypoint]),
                                 None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))

    k = cv2.waitKey()
    cv2.destroyAllWindows()

    if k == ord('j'):
        query_keypoint_index -= 1
        if query_keypoint_index == -1:
            query_keypoint_index = len(debug[page])-1
            page_index = (page_index-1) % len(debug.keys())
    elif k == ord('l'):
        query_keypoint_index += 1
        if query_keypoint_index == len(debug[page]):
            query_keypoint_index = 0
            page_index = (page_index+1) % len(debug.keys())


# for page in debug:
#     page_image = cv2.imread('{0}/{1}'.format(pages_directory_path, page))

#     for query_keypoint, codeword_keypoints in debug[page]:
#         print 'lollery ' + str(query_keypoint)
#         cv2.imshow('codeword-keypoints',
#                    cv2.drawKeypoints(page_image, utils.namedtuple_keypoints_to_cv2(codeword_keypoints),
#                                      None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))

#         cv2.imshow('query-keypoints',
#                    cv2.drawKeypoints(query_image, utils.namedtuple_keypoints_to_cv2([query_keypoint]),
#                                      None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
#         cv2.waitKey()
