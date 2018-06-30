#!/usr/bin/env python -u
import cv2
import PIL
import sys
import Tkinter

from PIL import Image, ImageTk


if len(sys.argv) != 6:
    sys.exit('Usage: {0} page_file_path query_file_path contrast_threshold n_octave_layers n_features'.format(
        sys.argv[0]))

page_file_path = sys.argv[1]
query_file_path = sys.argv[2]
contrast_threshold = float(sys.argv[3])
n_octave_layers = int(sys.argv[4])
n_features = int(sys.argv[5])

print 'Starting script...'
print '   {0: <15} = {1}'.format('page_file_path', page_file_path)
print '   {0: <15} = {1}'.format('query_file_path', query_file_path)

################################################################################

print 'Detecting keypoints...'
contrast_sift = cv2.SIFT(contrastThreshold=contrast_threshold, nOctaveLayers=n_octave_layers)
features_sift = cv2.SIFT(nfeatures=n_features, nOctaveLayers=n_octave_layers)

# load page
page_image = cv2.imread(page_file_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
page_keypoints = contrast_sift.detect(page_image, None)
assert len(page_keypoints) > 0

# load query
query_image = cv2.imread(query_file_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
query_keypoints_contrast = contrast_sift.detect(query_image, None)
query_keypoints_features = features_sift.detect(query_image, None)
assert len(query_keypoints_contrast) > 0
assert len(query_keypoints_features) > 0

########################################

# set up the gui
print 'Starting gui...'
root = Tkinter.Tk()
root.attributes('-fullscreen', True)

###################
left_frame = Tkinter.Frame(root, bd=2, relief=Tkinter.SUNKEN)
left_frame.grid_rowconfigure(0, weight=1)
left_frame.grid_columnconfigure(0, weight=1)
left_frame.pack(fill=Tkinter.BOTH, expand=True, side=Tkinter.LEFT)


xscrollbar = Tkinter.Scrollbar(left_frame, orient=Tkinter.HORIZONTAL)
xscrollbar.grid(row=1, column=0, sticky=Tkinter.E + Tkinter.W)

yscrollbar = Tkinter.Scrollbar(left_frame)
yscrollbar.grid(row=0, column=1, sticky=Tkinter.N + Tkinter.S)

canvas = Tkinter.Canvas(left_frame,
                        xscrollcommand=xscrollbar.set, yscrollcommand=yscrollbar.set,
                        bd=0, highlightthickness=0)
canvas.grid(row=0, column=0, sticky=Tkinter.N + Tkinter.S + Tkinter.E + Tkinter.W)

page_image_keypoints = ImageTk.PhotoImage(
    image=Image.fromarray(cv2.drawKeypoints(page_image, page_keypoints)))
canvas_image = canvas.create_image(0, 0, image=page_image_keypoints)

xscrollbar.config(command=canvas.xview)
yscrollbar.config(command=canvas.yview)
canvas.config(scrollregion=canvas.bbox(Tkinter.ALL))

###################

right_frame = Tkinter.Frame(root, bd=2, relief=Tkinter.SUNKEN)
right_frame.grid_rowconfigure(0, weight=1)
right_frame.grid_columnconfigure(0, weight=1)
right_frame.pack(fill=Tkinter.BOTH, expand=True, side=Tkinter.RIGHT)


query_image_keypoints = ImageTk.PhotoImage(
    image=Image.fromarray(cv2.drawKeypoints(query_image, query_keypoints_contrast)))
label_query_contrast = Tkinter.Label(right_frame, image=query_image_keypoints)
label_query_contrast.image = query_image_keypoints
label_query_contrast.pack()


query_image_keypoints = ImageTk.PhotoImage(
    image=Image.fromarray(cv2.drawKeypoints(query_image, query_keypoints_features)))
label_query_features = Tkinter.Label(right_frame, image=query_image_keypoints)
label_query_features.image = query_image_keypoints
label_query_features.pack()

root.mainloop()
