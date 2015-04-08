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

matches = utils.load_matches(matches_file_path)

for page, page_matches in matches.viewitems():
    for i, match in enumerate(page_matches):
        pages_directory_path = 'data sets/typesetted/'
        page_image = cv2.imread('{0}/{1}'.format(pages_directory_path, page))

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

        extracted_match = page_image[y_min:y_max, x_min:x_max]
        match_file_path = '/Users/fcagnin/Desktop/output/{0}-{1}.png'.format(page, i)
        cv2.imwrite(match_file_path, extracted_match)
