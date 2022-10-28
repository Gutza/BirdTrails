import math
from typing import List
import numpy as np
import cv2 as cv
import os
from tqdm import tqdm
from threading import Thread, Lock
import time

in_folder = "../R0008026-preprocessed"
out_folder = "../R0008026-out"

latency = 750
duration = 100

image_list = os.listdir(in_folder)
image_list.sort()

log_mutex = Lock()

tmp = cv.imread(os.path.join(in_folder, image_list[0]))
print("Allocating the history array... ", end="", flush=True)
history: np.array = 255 + np.zeros((latency + duration, tmp.shape[0], tmp.shape[1]))
print("done.")

def shIdx(n: int) -> int:
    """Computes a safe history index for a given value. Simply applies modulo.

    Args:
        n (int): The desired index (linear)

    Returns:
        int: The correct index (circular)
    """
    return (n + len(history)) % len(history)

def process_history_layer(lIdx: int, trY: float):
    translation_matrix = np.eye(3)
    translation_matrix[1, 2] = trY

    log_mutex.acquire()
    old_im = history[lIdx, :, :]
    log_mutex.release()

    if np.min(old_im) == 255:
        return

    old_im = np.minimum(255., old_im - trY/10.)
    old_im = cv.GaussianBlur(old_im, (7, 3), 0)
    old_im = cv.warpPerspective(old_im, translation_matrix, (old_im.shape[1], old_im.shape[0])) # Actually just translation
    old_im[math.floor(trY):, :] = 255 # Fill with white

    log_mutex.acquire()
    history[lIdx, :, :] = old_im
    log_mutex.release()

threads: List[Thread] = []
max_thread_count = 12
def herd_thread(new_thread: Thread):
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

for idx, filename in tqdm(enumerate(image_list), total=len(image_list), desc="Processing images", disable=False):
    hIdx = shIdx(idx)
    history[hIdx, :, :] = cv.imread(os.path.join(in_folder, filename))[:, :, 0]

    minI = max(0, idx - latency - duration + 1)
    maxI = max(0, idx - latency)
    for i in range(minI, maxI):
        herd_thread(Thread(target = process_history_layer, args = [shIdx(i), -(idx - latency - i)/10.]))

    for t in threads:
        t.join()
    threads = []

    final_im = np.min(history, 0)
    cv.imwrite(os.path.join(out_folder, f"trails-{idx:05d}.png"), final_im)

print("Finished.")
