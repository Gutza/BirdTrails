import math
import numpy as np
import cv2 as cv
import os
from tqdm import tqdm

in_folder = "../R0008026-preprocessed"
temp_folder = "../R0008026-temp"
out_folder = "../R0008026-out"

image_list = os.listdir(in_folder)
image_list.sort()

dark_im: np.array = None
idx = 0
first = True
prev_im: np.array = None

latency = 10
duration = 100

cache_size = 25

for idx, filename in tqdm(enumerate(image_list), total=len(image_list), desc="Processing images", disable=True):
    im = cv.imread(os.path.join(in_folder, filename))[:, :, 0]
    cv.imwrite(os.path.join(temp_folder, filename), im)

    if first:
        first = False
        continue

    final_im = None

    minI = max(0, idx-latency-duration)
    for i in range(idx, minI - 1, -1):
        if i < idx - latency:
            trY = -(idx - latency - i)/10.
            translation_matrix = np.eye(3)
            translation_matrix[1, 2] = trY
            old_imagepath = os.path.join(temp_folder, image_list[i])
            old_im = np.minimum(255., cv.imread(old_imagepath)[:, :, 0] - trY/10.)
            old_im = cv.GaussianBlur(old_im, (3, 3), 0)
            old_im = cv.warpPerspective(old_im, translation_matrix, (old_im.shape[1], old_im.shape[0])) # Actually just translation
            old_im[math.floor(trY):, :] = 255
            cv.imwrite(old_imagepath, old_im)
            if final_im is None:
                final_im = old_im
            else:
                final_im = np.minimum(final_im, old_im)
        elif final_im is None:
            final_im = cv.imread(os.path.join(temp_folder, image_list[i]))[:, :, 0]
        else:
            final_im = np.minimum(final_im, cv.imread(os.path.join(temp_folder, image_list[i]))[:, :, 0])

    cv.imwrite(os.path.join(out_folder, f"trails-{idx:05d}.png"), final_im)

    if idx % 10:
        continue

    cv.imshow("Trails", final_im)
    if 27 == cv.waitKey(2):
        print(f"Aborting the process at {filename}")
        exit()

print("Finished")
cv.imshow("Trails", dark_im)
cv.waitKey()