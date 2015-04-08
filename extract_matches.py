#!/usr/bin/env python2 -u
import cv2
import sys
import utils


if len(sys.argv) != 3:
    sys.exit('Usage: {0} pages_directory_path matches_file_path'.format(sys.argv[0]))

pages_directory_path = sys.argv[1]
matches_file_path = sys.argv[2]
offset_pixel = 25
print 'Starting script...'
print '   {0: <20} = {1}'.format('pages_directory_path', pages_directory_path)
print '   {0: <20} = {1}'.format('matches_file_path', matches_file_path)

################################################################################

# load matches
print 'Loading matches...'
matches = utils.load_matches(matches_file_path)

# extract top-k matches
k = 10
print 'Extracting top-{0} matches in each page...'.format(k)
for page, page_matches in matches.viewitems():
    top_k_page_matches = sorted(page_matches, key=lambda x: x[1])[:k]
    assert len(top_k_page_matches) <= k

    page_image = cv2.imread('{0}/{1}'.format(pages_directory_path, page))
    for i, match_scored in enumerate(top_k_page_matches):
        match = match_scored[0]

        # keep only non-empy keypoints
        match_keypoints = [keypoint for keypoint in match if keypoint]

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

        # draw box on match
        page_image = cv2.rectangle(page_image, (x_min, y_min), (x_max, y_max),
                                   (0, 255, 0), 2)

    cv2.imwrite('/Users/fcagnin/Desktop/output/{0}.png'.format(page), page_image)
