from genericpath import isfile
import math
from typing import List
import numpy as np
import cv2 as cv
import os
from tqdm import tqdm
from threading import Thread, Lock
import time
from wakepy import set_keepawake, unset_keepawake

# Configure these
max_thread_count = 12
"""How many threads to execute concurrently"""

latency = 500
"""How many frames to delay applying blur/fade/translate effects (500)"""

duration = 100
"""How many frames to apply the effects for (100)"""

preprocessed_folder = "../R0008026-postprocessed-v2"
"""A temporary folder used to export the pre-processed files: greyscale, and with limited empty images."""

output_folder = "../R0008026-out-production01"
"""The final output folder."""

pause_file = "pause"
"""Touch this file to pause processing."""
# End of configuration

os.makedirs(output_folder, exist_ok=True)

image_list = os.listdir(preprocessed_folder)
image_list.sort()
for i in range(latency + duration):
    image_list.append(None)

log_mutex = Lock()

print("Allocating the history arrays... ", end="", flush=True)
tmp = cv.imread(os.path.join(preprocessed_folder, image_list[0]))
valid_layers: np.array = np.zeros((latency + duration), dtype=bool)
history_layers: np.array = 255 + np.zeros((latency + duration, tmp.shape[0], tmp.shape[1]))
tmp = None
print("done.")

def shIdx(n: int) -> int:
    """Computes a safe history index for a given value. Simply applies modulo.

    Args:
        n (int): The desired index (linear)

    Returns:
        int: The correct index (circular)
    """
    return (n + len(history_layers)) % len(history_layers)

def process_history_layer(lIdx: int, factor: float):
    """Processes a single history layer, applying Gaussian blur and translation, and fading it out. Thread-safe.

    Args:
        lIdx (int): The layer index.
        factor (float): The translation factor (also produces the fade out factor)
    """
    translation_matrix = np.eye(3)
    translation_matrix[0, 2] = factor # translate left
    # translation_matrix[1, 2] = factor # translate up

    log_mutex.acquire()
    old_im = history_layers[lIdx, :, :]
    log_mutex.release()

    old_im = np.minimum(255., old_im - factor/10.)
    old_im = cv.GaussianBlur(old_im, (7, 3), 0)

    old_im = cv.warpPerspective(old_im, translation_matrix, (old_im.shape[1], old_im.shape[0])) # Actually just translation
    # old_im[math.floor(trY):, :] = 255 # Fill with white at the bottom
    old_im[:, math.floor(factor):] = 255 # Fill with white on the right

    log_mutex.acquire()
    history_layers[lIdx, :, :] = old_im
    log_mutex.release()

    if np.min(old_im) == 255:
        valid_layers[lIdx] = False

threads: List[Thread] = []
def herd_thread(new_thread: Thread):
    """Thread herder. Caps the number of active threads.

    Args:
        new_thread (Thread): The new thread to queue when it's go time.
    """
    global threads
    go_time = len(threads) < max_thread_count
    while not go_time:
        live_threads = []
        for t in threads:
            if t.is_alive():
                live_threads.append(t)
            else:
                t.join()
                go_time = True
        threads = live_threads

        if not go_time:
            time.sleep(0.001) # Wait a millisecond there, mister!

    new_thread.start()
    threads.append(new_thread)

print(f"Create file «{pause_file}» at any time to pause processing.")
set_keepawake(keep_screen_awake=False)
with tqdm(enumerate(image_list), total=len(image_list), desc="Processing images", disable=False) as tq:
    for idx, filename in tq:
        hIdx = shIdx(idx)
        if filename is None:
            history_layers[hIdx, :, :] = 255 + np.zeros((history_layers.shape[1], history_layers.shape[2]))
            valid_layers[hIdx] = False
        else:
            history_layers[hIdx, :, :] = cv.imread(os.path.join(preprocessed_folder, filename))[:, :, 0]
            valid_layers[hIdx] = True

        minI = max(0, idx - latency - duration + 1)
        maxI = max(0, idx - latency)
        for i in range(minI, maxI):
            lIdx = shIdx(i)
            if not valid_layers[lIdx]:
                continue
            herd_thread(Thread(target = process_history_layer, args = [lIdx, -(idx - latency - i)/10.]))

        for t in threads:
            t.join()
        threads = []

        if len(valid_layers[valid_layers]):
            final_im = np.min(history_layers[valid_layers, :, :], 0)
        else:
            final_im = history_layers[0, :, :] # If all layers are white, just return the first one

        cv.imwrite(os.path.join(output_folder, f"trails-{idx:05d}.png"), final_im)

        if os.path.isfile(pause_file):
            tq.write(f"Paused; remove file «{pause_file}» to resume.")
            while True:
                time.sleep(0.1)
                if not os.path.isfile(pause_file):
                    break
            tq.write("Resumed.")
            tq.unpause()
unset_keepawake()

print("Finished.")
