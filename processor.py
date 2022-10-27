import numpy as np
import cv2 as cv
import os
import time
from tqdm import tqdm

in_folder = "../R0008026"
out_folder = "../R0008026-out"

im_count = len(os.listdir(in_folder))

lo_val = 25
hi_val = 50

dark_im: np.array = None
first = True
for idx, filename in tqdm(enumerate(os.listdir(in_folder)), desc="Processing images", total=im_count):
    filepath = os.path.join(in_folder, filename)
    im = cv.cvtColor(cv.imread(filepath), cv.COLOR_BGR2GRAY)
    
    im = np.minimum(im, hi_val)
    im = np.maximum(im, lo_val)

    if first:
        dark_im = im
        first = False
    else:
        dark_im = np.minimum(dark_im, im)

    image_path_out = os.path.join(out_folder, filename)
    cv.imwrite(image_path_out, dark_im)

    if idx % 10:
        continue

    cv.imshow("Trails", dark_im)
    if 27 == cv.waitKey(2):
        exit()

print("Finished")
cv.imshow("Trails", dark_im)
cv.waitKey()