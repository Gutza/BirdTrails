import numpy as np
import cv2 as cv
import os
from tqdm import tqdm
from wakepy import set_keepawake, unset_keepawake

# Configure these
empty_duration = 120
"""The longest empty image sequence to allow before skipping ahead"""

minimum_pixel_count = 10
"""How many pixels have to be dark for an image to be considered non-empty"""

lo_val = 25
"""The darkest value you want to keep (this remaps to black)"""

hi_val = 45
"""The lightest value you want to keep (this remaps to white)"""

raw_footage_folder = "../R0008026"
"""The folder containing your raw footage. The files must be exported as sequential images. All images in this folder are processed in alphabetical order as a single sequence."""

preprocessed_folder = "../R0008026-postprocessed"
"""A temporary folder used to export the pre-processed files: greyscale, and with limited empty images."""
# End of configuration

os.makedirs(preprocessed_folder, exist_ok=True)

image_list = os.listdir(raw_footage_folder)
image_list.sort()

factor = 255. / (hi_val - lo_val)
idx = 0
empty_streak = 0

set_keepawake(keep_screen_awake=False)
with tqdm(image_list, desc="Processing", total=len(image_list), disable=False) as tq:
    for filename in tq:
        filepath = os.path.join(raw_footage_folder, filename)
        im = cv.cvtColor(cv.imread(filepath), cv.COLOR_BGR2GRAY)

        im = np.minimum(im, hi_val)
        im = np.maximum(im, lo_val)
        im = (im - lo_val) * factor

        if np.sum(im < 128) > 10:
            empty_streak = 0
        else:
            empty_streak += 1
            if empty_streak > empty_duration:
                cv.destroyAllWindows()
                tq.desc = "Skipping"
                continue

        tq.desc = "Processing"

        image_path_out = os.path.join(preprocessed_folder, f"preprocessed-{idx:05d}-{os.path.splitext(filename)[0]}.png")
        cv.imwrite(image_path_out, im)
        idx += 1

        if idx % 10:
            continue

        cv.imshow("Trails", im)
        if 27 == cv.waitKey(1):
            print(f"Aborting the process at {filename}")
            exit()
unset_keepawake()

print("Finished")
