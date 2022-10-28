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
idx = 0
first = True
prev_im: np.array = None

for filename in tqdm(os.listdir(in_folder), desc="Processing images", total=im_count, disable=False):
    filepath = os.path.join(in_folder, filename)
    im = cv.cvtColor(cv.imread(filepath), cv.COLOR_BGR2GRAY)
    
    im = np.minimum(im, hi_val)
    im = np.maximum(im, lo_val)
    im = (im - lo_val) * (256. - hi_val)

    if first:
        dark_im = im
        prev_im = im
        first = False
        continue

    dark_im = np.minimum(im, dark_im)
    diff = np.sum(prev_im - dark_im)

    if diff < 25:
        continue

    prev_im = dark_im
    idx += 1

    image_path_out = os.path.join(out_folder, f"trails-{idx:05d}.jpg")
    cv.imwrite(image_path_out, dark_im)

    if idx % 10:
        continue

    cv.imshow("Trails", dark_im)
    if 27 == cv.waitKey(2):
        print(f"Aborting the process at {filename}")
        exit()

print("Finished")
cv.imshow("Trails", dark_im)
cv.waitKey()