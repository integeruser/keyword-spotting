#!/usr/bin/env python2 -u
import cv2
import sys
import utils


if len(sys.argv) != 4:
    sys.exit('Usage: {0} pages_directory_path matches_file_path output_directory_path'.format(sys.argv[0]))

pages_directory_path = sys.argv[1]
matches_file_path = sys.argv[2]
output_directory_path = sys.argv[3]
print 'Starting script...'
print '   {0: <21} = {1}'.format('pages_directory_path', pages_directory_path)
print '   {0: <21} = {1}'.format('matches_file_path', matches_file_path)
print '   {0: <21} = {1}'.format('output_directory_path', output_directory_path)

################################################################################

# load matches
print 'Loading matches...'
matches = utils.load_matches(matches_file_path)

# extract top-k matches
k = 20
print 'Extracting top-{0} matches in each page...'.format(k)
for page, page_matches in matches.viewitems():
    top_k_page_matches = sorted(page_matches, key=lambda x: x[1])[:k]
    assert len(top_k_page_matches) <= k

    page_image = cv2.imread('{0}/{1}'.format(pages_directory_path, page))
    for i, match_scored in enumerate(top_k_page_matches):
        match = match_scored[0]

        # keep only non-empy keypoints
        match_keypoints = [keypoint for keypoint in match if keypoint]
        match_keypoints = utils.namedtuple_keypoints_to_cv2(match_keypoints)

        # draw keypoints on original image
        page_image = cv2.drawKeypoints(page_image, match_keypoints, None,
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # draw box around match
        offset_pixel = 25
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

        x_min += - offset_pixel
        y_min += - offset_pixel
        x_max += + offset_pixel
        y_max += + offset_pixel
        cv2.rectangle(page_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.imwrite('{0}/{1}.png'.format(output_directory_path, page), page_image)
