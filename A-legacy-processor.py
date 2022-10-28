import numpy as np
import cv2 as cv
import os
from tqdm import tqdm

in_folder = "../R0008026-preprocessed"
temp_folder = "../R0008026-temp"
out_folder = "../R0008026-out"

image_list = os.listdir(in_folder).sort()

dark_im: np.array = None
idx = 0
first = True
prev_im: np.array = None
history: np.array = None

latency = 10

for idx, filename in tqdm(enumerate(image_list), desc="Processing images", disable=False):
    filepath = os.path.join(in_folder, filename)
    im = cv.imread(filepath)
    
    if first:
        dark_im = im
        prev_im = im
        history = np.zeros(im.shape)

        first = False
        continue

    dark_im = np.minimum(im, dark_im)

    history[im < 128] = idx
    dark_im[history < idx - latency] += 0.01

    prev_im = dark_im

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