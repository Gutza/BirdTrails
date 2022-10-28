import numpy as np
import cv2 as cv
import os
from tqdm import tqdm

in_folder = "../R0008026"
out_folder = "../R0008026-preprocessed"

im_count = len(os.listdir(in_folder))

lo_val = 25
hi_val = 45
factor = 255. / (hi_val - lo_val)
idx = 0

with tqdm(os.listdir(in_folder), desc="Processing", total=im_count, disable=False) as tq:
    for filename in tq:
        filepath = os.path.join(in_folder, filename)
        im = cv.cvtColor(cv.imread(filepath), cv.COLOR_BGR2GRAY)
        
        im = np.minimum(im, hi_val)
        im = np.maximum(im, lo_val)
        im = (im - lo_val) * factor

        if np.sum(im < 128) < 10: # Skip images with less than 10 dark pixels
            tq.desc = "Skipping"
            continue

        tq.desc = "Processing"

        image_path_out = os.path.join(out_folder, f"preprocessed-{idx:05d}-{os.path.splitext(filename)[0]}.png")
        cv.imwrite(image_path_out, im)
        idx += 1

        if idx % 10:
            continue

        cv.imshow("Trails", im)
        if 27 == cv.waitKey(2):
            print(f"Aborting the process at {filename}")
            exit()

print("Finished")
