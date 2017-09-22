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


def predict_asIntType(net, test_loader, out_dir):

    test_dataset = test_loader.dataset

    num = len(test_dataset)
    H, W = CARVANA_HEIGHT, CARVANA_WIDTH
    predictions = np.memmap(out_dir + '/preds.npy', dtype=np.uint8, mode='w+', shape=(num, H, W))

    test_num  = 0
    for it, (images, indices) in enumerate(test_loader, 0):
        images = Variable(images.cuda(),volatile=True)

        # forward
        logits = net(images)
        probs  = F.sigmoid(logits)

        batch_size = len(indices)
        test_num  += batch_size
        start = test_num-batch_size
        end   = test_num
        probs = probs.data.cpu().numpy().reshape(-1, H, W)*255
        predictions[start:end] = probs.astype(np.uint8)

    assert(test_num == len(test_loader.sampler))
    return predictions


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


def ensemble_csv(csv_list, out_dir):

    gz_file = out_dir + "/results-ensemble.csv.gz"

    num     = 100064
    H, W    = CARVANA_HEIGHT, CARVANA_WIDTH
    predictions = np.memmap(out_dir + '/preds.npy', dtype=np.uint8, mode='w+', shape=(num, H, W))

    for file_indices in range(0, len(csv_list)):
        df = pd.read_csv(csv_list[file_indices], compression='gzip', header=0) # fold1/results-final.csv.gz
        for n in range(0, num):
            rle  = df.values[n][1]
            mask = run_length_decode(rle, H=CARVANA_HEIGHT, W=CARVANA_WIDTH)/255
            predictions[n] = predictions[n] + mask

    predictions = predictions > (len(csv_list)/2)
    names       = []

    for n in range(0, num):
        name  = df.values[n][0]
        names.append(name)

    rles  = []


    for n in range(0, num):
        if (n % 1000 == 0):
            print('rle : b/num_test = %06d/%06d' % (n, num))

        pred = predictions[n]
        rle = run_length_encode(pred)
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

    ensemble_csv(csv_list,"/Users/Eugene/Documents/Git")