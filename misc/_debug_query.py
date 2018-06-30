#!/usr/bin/env python2 -u
import cv2
import math
import numpy
import PIL
import re
import sys
import Tkinter
import ttk
import utils

from PIL import Image, ImageTk


def draw(kp, nearest):
    image_color = image_color_original.copy()

    # draw nearest kp to click
    keypoint_pt = (int(kp.pt[0]), int(kp.pt[1]))
    cv2.circle(image_color, keypoint_pt, radius=5, color=(0, 255, 0), thickness=2)

    # draw nearest (in the descriptor sense) kps to kp
    angle_tolerance = 10
    for (index, descriptor) in nearest:
        keypoint = image_keypoints[index]
        pt = (int(keypoint.pt[0]), int(keypoint.pt[1]))

        # see http://code.opencv.org/issues/2987
        octave = keypoint.octave & 255
        layer = (keypoint.octave >> 8) & 255
        radius = 5 * layer

        # compute pt2 for drawing angles as lines
        theta = keypoint.angle * math.pi / 180.0
        pt2 = (int(pt[0] + radius * math.cos(theta)), int(pt[1] + radius * math.sin(theta)))

        # filter angles
        c1 = keypoint.angle >= (kp.angle - angle_tolerance) % 360
        c2 = keypoint.angle <= (kp.angle + angle_tolerance) % 360
        if c1 and c2:
            cv2.circle(image_color, pt, radius=radius, color=(0, 0, 255), thickness=2)
            cv2.line(image_color, pt, pt2, (0, 0, 255), thickness=2)
        else:
            cv2.circle(image_color, pt, radius=radius, color=(255, 0, 0), thickness=2)
            cv2.line(image_color, pt, pt2, (255, 0, 0), thickness=2)

    global image_color_pil
    global image_color_tk
    image_color_pil = Image.fromarray(image_color)
    image_color_tk = ImageTk.PhotoImage(image=image_color_pil)
    canvas.itemconfigure(canvas_image, image=image_color_tk)


def canvas_click(event):
    # translate click x and y into image space x and y
    x = image_color_tk.width() * xscrollbar.get()[0] + event.x
    y = image_color_tk.height() * yscrollbar.get()[0] + event.y

    # find nearest keypoint to the click point
    click_point = (int(x), int(y))
    keypoints_distances = [(keypoint_index, numpy.linalg.norm(click_point - numpy.asarray(keypoint.pt)))
                           for keypoint_index, keypoint in enumerate(image_keypoints)]
    nearest_keypoint_to_click = sorted(keypoints_distances, key=lambda x: x[1])[0]
    nearest_keypoint_to_click_index = nearest_keypoint_to_click[0]

    # find nearest keypoints (considering descriptor distance) to the selected keypoint
    descriptors_distances = [(index, numpy.linalg.norm(image_descriptors[nearest_keypoint_to_click_index] - descriptor))
                             for index, descriptor in enumerate(image_descriptors)]
    similar_keypoints = sorted(descriptors_distances, key=lambda x: x[1])[:200]

    # draw results on image
    draw(image_keypoints[nearest_keypoint_to_click_index], similar_keypoints)


def left_combobox_selected(event):
    page_file_name = left_combobox['values'][left_combobox.current()]

    global image_color_pil
    global image_color_tk
    image_color_pil = Image.fromarray(pages_color[page_file_name])
    image_color_tk = ImageTk.PhotoImage(image=image_color_pil)
    canvas.itemconfigure(canvas_image, image=image_color_tk)


if len(sys.argv) != 5:
    sys.exit(
        'Usage: {0} pages_directory_path corpus_file_path codebook_file_path query_file_path'.format(sys.argv[0]))

pages_directory_path = sys.argv[1]
corpus_file_path = sys.argv[2]
codebook_file_path = sys.argv[3]
query_file_path = sys.argv[4]
print 'Starting script...'
print '   {0: <20} = {1}'.format('pages_directory_path', pages_directory_path)
print '   {0: <20} = {1}'.format('corpus_file_path', corpus_file_path)
print '   {0: <20} = {1}'.format('codebook_file_path', codebook_file_path)
print '   {0: <20} = {1}'.format('query_file_path', query_file_path)

################################################################################

# load corpus, codebook, pages
assert corpus_file_path[corpus_file_path.find(
    '-'):] == codebook_file_path[codebook_file_path.find('-'):codebook_file_path.find('--')]

print 'Loading corpus...'
corpus = utils.load_corpus(corpus_file_path)

print 'Loading codebook...'
codebook = utils.load_codebook(codebook_file_path)

# contrast_threshold = float(re.findall(r'codebook-.*-([^(]*)-.*--', codebook_file_path)[0])
# n_octave_layers = int(re.findall(r'codebook-.*-([^(]*)--', codebook_file_path)[0])
# sift = cv2.SIFT(contrastThreshold=contrast_threshold, nOctaveLayers=n_octave_layers)

print 'Loading pages...'
pages = pages_color = dict()
for page_file_name in corpus['pages']:
    page_image = cv2.imread(
        '{0}/{1}'.format(pages_directory_path, page_file_name), cv2.CV_LOAD_IMAGE_GRAYSCALE)
    pages[page_file_name] = page_image

    page_image_color = cv2.imread('{0}/{1}'.format(pages_directory_path, page_file_name))
    assert page_image_color is not None
    pages_color[page_file_name] = page_image_color
assert len(pages) > 0

########################################

# set up the gui
print 'Starting gui...'
root = Tkinter.Tk()
root.attributes('-fullscreen', True)

initial_page = corpus['pages'][0]

###################
left_frame = Tkinter.Frame(root, bd=2, relief=Tkinter.SUNKEN)
left_frame.grid_rowconfigure(0, weight=1)
left_frame.grid_columnconfigure(0, weight=1)
left_frame.pack(fill=Tkinter.BOTH, expand=True, side=Tkinter.LEFT)

left_combobox = ttk.Combobox(left_frame, state='readonly')
left_combobox['values'] = corpus['pages']
left_combobox.set(initial_page)
left_combobox.bind("<<ComboboxSelected>>", left_combobox_selected)
left_combobox.grid(row=0, column=0, sticky=Tkinter.E + Tkinter.W)


xscrollbar = Tkinter.Scrollbar(left_frame, orient=Tkinter.HORIZONTAL)
xscrollbar.grid(row=2, column=0, sticky=Tkinter.E + Tkinter.W)

yscrollbar = Tkinter.Scrollbar(left_frame)
yscrollbar.grid(row=1, column=1, sticky=Tkinter.N + Tkinter.S)

canvas = Tkinter.Canvas(left_frame, xscrollcommand=xscrollbar.set, yscrollcommand=yscrollbar.set,
                        bd=0, highlightthickness=0)
canvas.grid(row=1, column=0, sticky=Tkinter.N + Tkinter.S + Tkinter.E + Tkinter.W)
canvas.bind("<Button-1>", canvas_click)

image_color = pages_color[initial_page].copy()
image_color_pil = Image.fromarray(image_color)
image_color_tk = ImageTk.PhotoImage(image=image_color_pil)
canvas_image = canvas.create_image(0, 0, image=image_color_tk)

xscrollbar.config(command=canvas.xview)
yscrollbar.config(command=canvas.yview)
canvas.config(scrollregion=canvas.bbox(Tkinter.ALL))

###################

right_frame = Tkinter.Frame(root, bd=2, relief=Tkinter.SUNKEN)
right_frame.grid_rowconfigure(0, weight=1)
right_frame.grid_columnconfigure(1, weight=1)
right_frame.pack(fill=Tkinter.BOTH, expand=True, side=Tkinter.RIGHT)

# combobox = ttk.Combobox(right_frame, state='readonly')
# combobox['values'] = corpus['pages']
# combobox.set(initial_page)
# combobox.bind("<<ComboboxSelected>>", combobox_selected)
# combobox.pack()

root.mainloop()
