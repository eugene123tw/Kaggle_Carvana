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
        accu_mask = accu_mask > len(csv_list)/2

        # im_show('mask', accu_mask, resize=0.25, cmap='gray')
        # plt.waitforbuttonpress()

        rle = run_length_encode(accu_mask)
        rles.append(rle)
        print('rle : b/num_test = %06d/%06d' % (n, 100064))


    df = pd.DataFrame({'img': names, 'rle_mask': rles})
    df.to_csv(gz_file, index=False, compression='gzip')



def ensemble_csv_v2(csv_list, out_dir):

    gz_file = out_dir + "/results-ensemble.csv.gz"

    rles = []
    names = []

    num = 100064
    H, W = CARVANA_HEIGHT, CARVANA_WIDTH
    binary_pred = np.memmap(out_dir + '/preds.npy', dtype=np.uint8, mode='w+', shape=(num, H, W))

    print("Do merge CSV")
    for file_indices in range(0, len(csv_list)):
        df = pd.read_csv(csv_list[file_indices], compression='gzip', header=0)
        for n in range(0, num):
            rle = df.values[n][1]
            binary_pred[n] = binary_pred[n] + (run_length_decode(rle, H=CARVANA_HEIGHT, W=CARVANA_WIDTH)/255)
            if file_indices==0:
                names.append(df.values[n][0])
            if n%1000==0:
                print('File indices: %06d/%06d, rle : b/num_test = %06d/%06d' %
                      (file_indices, len(csv_list), n + 1, num)
                )

    binary_pred = binary_pred > (len(csv_list)/2)

    print("Do run_length_encode")
    for n in range(0, num):
        mask = binary_pred[n]
        rle = run_length_encode(mask)
        rles.append(rle)
        print('rle : b/num_test = %06d/%06d' % (n+1, 100064))

    df = pd.DataFrame({'img': names, 'rle_mask': rles})
    df.to_csv(gz_file, index=False, compression='gzip')



def ensemble_csv_v3(csv_list, out_dir, chunk_size=1000.0):

    gz_file = out_dir + "/results-ensemble.csv.gz"

    rles = []
    names = []

    num = 100064
    H, W = CARVANA_HEIGHT, CARVANA_WIDTH
    chunk_batch = int(math.ceil(num/chunk_size)) # 100064/1000. = 100.064

    begin, end = 0, 0

    for chunk in range(0, chunk_batch):
        begin = end
        end = begin + chunk_size
        if end > num : end = num

        binary_pred = np.memmap(out_dir + '/preds.npy', dtype=np.uint8, mode='w+', shape=(end-begin, H, W))
        for file_indices in range(0, len(csv_list)):
            df = pd.read_csv(csv_list[file_indices], compression='gzip', header=0)
            for n in range(begin, end):
                rle = df.values[n][1]
                binary_pred[n] = binary_pred[n] + (run_length_decode(rle, H=CARVANA_HEIGHT, W=CARVANA_WIDTH) / 255)
                if file_indices == 0:
                    names.append(df.values[n][0])
                if n % 1000 == 0:
                    print('File indices: %02d/%06d, rle : b/num_test = %06d/%06d' %
                          (file_indices+1, len(csv_list), n + 1, num)
                          )

        binary_pred = binary_pred > (len(csv_list) / 2)
        print("Do run_length_encode")
        for n in range(begin, end):
            mask = binary_pred[n]
            rle = run_length_encode(mask)
            rles.append(rle)
            print('rle : b/num_test = %06d/%06d' % (n + 1, num))

    df = pd.DataFrame({'img': names, 'rle_mask': rles})
    df.to_csv(gz_file, index=False, compression='gzip')



if __name__ == "__main__":
    csv_list = [
        "/Users/Eugene/Documents/Git/fold1/results-final.csv.gz",
        "/Users/Eugene/Documents/Git/fold2/results-final.csv.gz",
    ]

    ensemble_csv(csv_list,"/Users/Eugene/Documents/Git/results-ensemble.csv.gz")

    ensemble_csv_v2(csv_list, "/Users/Eugene/Documents/Git")
