#!/usr/bin/env python3
import argparse
import os
import random

import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gw_20p_wannot_dirpath')
    args = parser.parse_args()

    boxes_filename = random.choice(
        [filename for filename in os.listdir(args.gw_20p_wannot_dirpath) if 'boxes' in filename])
    print(f'Boxes: {boxes_filename}')

    with open(os.path.join(args.gw_20p_wannot_dirpath, boxes_filename), encoding='ascii') as f:
        boxes_filename_content = f.readlines()

    # retrieve the page filename from the boxes filename
    boxes_filename_name = boxes_filename_content[0]
    page_filename = boxes_filename_name[:boxes_filename_name.index('_')] + '.tif'
    print(f'Page: {page_filename}')

    # choose a random box
    print('Loading a random box...')
    random_box = random.choice(boxes_filename_content[1:])
    points = random_box.split()
    print(f'Points: {points}')

    # extract the box from its page
    page = cv2.imread(os.path.join(args.gw_20p_wannot_dirpath, page_filename), cv2.IMREAD_GRAYSCALE)
    page_height, page_width = page.shape[:2]
    x1 = int(float(points[0]) * page_width)
    x2 = int(float(points[1]) * page_width)
    y1 = int(float(points[2]) * page_height)
    y2 = int(float(points[3]) * page_height)
    pt1 = (x1, y1)
    pt2 = (x2, y2)
    print(f'Pt1: {pt1}')
    print(f'Pt2: {pt2}')

    cv2.imwrite('query.png', page[y1:y2, x1:x2])
