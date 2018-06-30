#!/usr/bin/env python2 -Bu
import cv2
import math
import numpy
import PIL
import random
import sys
import Tkinter

from PIL import Image, ImageTk


def draw(kp, nearest):
    image_color = image_color_original.copy()

    # draw nearest kp to click
    keypoint_pt = (int(kp.pt[0]), int(kp.pt[1]))
    cv2.circle(image_color, keypoint_pt, radius=5, color=(255, 255, 255), thickness=2)

    # draw nearest (in the descriptor sense) kps to kp
    angle_tolerance = 10
    for (index, descriptor) in nearest:
        keypoint = image_keypoints[index]
        pt = (int(keypoint.pt[0]), int(keypoint.pt[1]))

        # see http://code.opencv.org/issues/2987
        # octave = keypoint.octave & 255
        # layer = (keypoint.octave >> 8) & 255
        radius = int(keypoint.size)

        # compute pt2 for drawing angles as lines
        theta = keypoint.angle * math.pi / 180.0
        pt2 = (int(pt[0] + radius * math.cos(theta)), int(pt[1] + radius * math.sin(theta)))

        # filter angles
        c1 = keypoint.angle >= (kp.angle - angle_tolerance) % 360
        c2 = keypoint.angle <= (kp.angle + angle_tolerance) % 360
        if c1 and c2:
            cv2.circle(image_color, pt, radius=radius, color=(0, 255, 0))
            cv2.line(image_color, pt, pt2, (0, 255, 0))
        else:
            cv2.circle(image_color, pt, radius=radius, color=(255, 0, 0))
            cv2.line(image_color, pt, pt2, (255, 0, 0))

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
    keypoints_distances = [(i, numpy.linalg.norm(click_point - numpy.asarray(keypoint.pt)))
                           for i, keypoint in enumerate(image_keypoints)]
    nearest_keypoint_to_click = sorted(keypoints_distances, key=lambda x: x[1])[0]
    nearest_keypoint_to_click_index = nearest_keypoint_to_click[0]

    # find nearest keypoints (considering descriptor distance) to the selected keypoint
    descriptors_distances = [(index, numpy.linalg.norm(image_descriptors[nearest_keypoint_to_click_index] - descriptor))
                             for index, descriptor in enumerate(image_descriptors)]
    similar_keypoints = sorted(descriptors_distances, key=lambda x: x[1])[:200]

    # draw results on image
    draw(image_keypoints[nearest_keypoint_to_click_index], similar_keypoints)


if len(sys.argv) != 4:
    sys.exit('Usage: {0} image_file_path contrast_threshold n_octave_layers'.format(sys.argv[0]))

image_file_path = sys.argv[1]
contrast_threshold = float(sys.argv[2])
n_octave_layers = int(sys.argv[3])

print 'Starting script...'
print '   {0: <18} = {1}'.format('image_file_path', image_file_path)
print '   {0: <18} = {1}'.format('contrast_threshold', contrast_threshold)
print '   {0: <18} = {1}'.format('n_octave_layers', n_octave_layers)

################################################################################

# load the image
sift = cv2.SIFT(contrastThreshold=contrast_threshold, nOctaveLayers=n_octave_layers)

image_color_original = cv2.imread(image_file_path)
image = cv2.cvtColor(image_color_original, cv2.COLOR_BGR2GRAY)

image_keypoints, image_descriptors = sift.detectAndCompute(image, None)
assert len(image_keypoints) > 0


# set up the gui
root = Tkinter.Tk()
root.attributes('-fullscreen', True)

frame = Tkinter.Frame(root, bd=2, relief=Tkinter.SUNKEN)
frame.grid_rowconfigure(0, weight=1)
frame.grid_columnconfigure(0, weight=1)
frame.pack(fill=Tkinter.BOTH, expand=True)

xscrollbar = Tkinter.Scrollbar(frame, orient=Tkinter.HORIZONTAL)
xscrollbar.grid(row=1, column=0, sticky=Tkinter.E + Tkinter.W)

yscrollbar = Tkinter.Scrollbar(frame)
yscrollbar.grid(row=0, column=1, sticky=Tkinter.N + Tkinter.S)

canvas = Tkinter.Canvas(frame, xscrollcommand=xscrollbar.set, yscrollcommand=yscrollbar.set,
                        bd=0, highlightthickness=0)
canvas.grid(row=0, column=0, sticky=Tkinter.N + Tkinter.S + Tkinter.E + Tkinter.W)
canvas.bind("<Button-1>", canvas_click)

image_color = image_color_original.copy()
image_color_pil = Image.fromarray(image_color)
image_color_tk = ImageTk.PhotoImage(image=image_color_pil)
canvas_image = canvas.create_image(0, 0, image=image_color_tk)

xscrollbar.config(command=canvas.xview)
yscrollbar.config(command=canvas.yview)
canvas.config(scrollregion=canvas.bbox(Tkinter.ALL))

root.mainloop()
