import os
# numerical libs
import math
import numpy as np
import random
import PIL
import cv2
import pandas as pd
import matplotlib.pyplot as plt


CARVANA_HEIGHT = 1280
CARVANA_WIDTH  = 1918

def im_show(name, image, resize=1, cmap = ''):
    H,W = image.shape[0:2]
    if cmap=='gray':
        plt.imshow(image.astype(np.uint8),cmap='gray')
    plt.imshow(image.astype(np.uint8))


#https://www.kaggle.com/stainsby/fast-tested-rle
def run_length_encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle


def run_length_decode(rel, H, W, fill_value=255):
    mask = np.zeros((H * W), np.uint8)
    rel = np.array([int(s) for s in rel.split(' ')]).reshape(-1, 2)
    for r in rel:
        start = r[0]
        end = start + r[1]
        mask[start:end] = fill_value
    mask = mask.reshape(H, W)
    return mask




def ensemble_csv(csv_list, gz_file):
    rles = []
    names = []

    for n in range(0,100064):
        accu_mask = 0
        for file_indices in range(0, len(csv_list)):
            df = pd.read_csv(csv_list[file_indices], compression='gzip', header=0, skiprows= n, nrows = 1)
            rle = df.values[0][1]
            accu_mask += run_length_decode(rle, H=CARVANA_HEIGHT, W=CARVANA_WIDTH)/255

        names.append(df.values[0][0]) # Append name
        accu_mask = accu_mask/len(csv_list)

        im_show('mask', accu_mask, resize=0.25, cmap='gray')
        plt.waitforbuttonpress()

        rle = run_length_encode(accu_mask)
        rles.append(rle)


    df = pd.DataFrame({'img': names, 'rle_mask': rles})
    df.to_csv(gz_file, index=False, compression='gzip')

    # im_show('mask', accu_mask, resize=0.25, cmap='gray')
    # plt.waitforbuttonpress()





if __name__ == "__main__":
    csv_list = [
        "/Users/Eugene/Documents/Git/fold1/results-final.csv.gz",
        "/Users/Eugene/Documents/Git/fold2/results-final.csv.gz",
    ]

    ensemble_csv(
        csv_list,
        "/Users/Eugene/Documents/Git/results-ensemble.csv.gz"
        )
