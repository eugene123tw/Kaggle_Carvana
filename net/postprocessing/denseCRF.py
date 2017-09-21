from net.dataset.dataprocess import *

import numpy as np
import pydensecrf.densecrf as dcrf
import cv2

import multiprocessing
from itertools import product

from pydensecrf.utils import unary_from_softmax, unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

TEST_DIR = '/home/eugene/Documents/Kaggle_Carvana/data/image/test-jpg'
LABEL_DIR = '/home/eugene/Documents/Kaggle_Carvana/results/unet-2-1024/pred_mask'
PREDICT_ARR_DIR = '/home/eugene/Documents/Kaggle_Carvana/results/unet-2-1024'
OUT_DIR = '/home/eugene/Documents/Kaggle_Carvana/results/unet-2-1024/pred_mask_crf'


def doCRF(n,name):

    predict = np.memmap(PREDICT_ARR_DIR + '/preds.npy', dtype=np.uint8, mode='r', shape=(100064, CARVANA_H, CARVANA_W))
    prob = np.asarray(predict[n], dtype=np.float32)
    prob = cv2.resize(prob, (CARVANA_WIDTH, CARVANA_HEIGHT), interpolation=cv2.INTER_LINEAR) / 255.

    img = cv2.imread(TEST_DIR + '/' + name)

    prob = np.tile(prob[np.newaxis, :, :], (2, 1, 1))
    prob[1, :, :] = 1 - prob[0, :, :]

    U = unary_from_softmax(prob)

    pairwise_energy = create_pairwise_bilateral(sdims=(10, 10), schan=(0.01,), img=img, chdim=2)

    d = dcrf.DenseCRF2D(CARVANA_WIDTH, CARVANA_HEIGHT, 2)
    d.setUnaryEnergy(U)
    d.addPairwiseEnergy(pairwise_energy, compat=10)

    Q, tmp1, tmp2 = d.startInference()

    for _ in range(5):
        d.stepInference(Q, tmp1, tmp2)
    kl3 = d.klDivergence(Q) / (CARVANA_HEIGHT * CARVANA_WIDTH)
    map_soln3 = np.argmax(Q, axis=0).reshape((CARVANA_HEIGHT, CARVANA_WIDTH))

    # plt.imshow(map_soln3)
    # plt.waitforbuttonpress()
    cv2.imwrite(OUT_DIR + '/%s' % name, ((1 - map_soln3) * 255).astype(np.uint8))


def denseCRF_nonRGB_multipro():
    test_dataset = ImageDataset('test-INTER_LINEAR-1024x1024-100064',
                                is_mask=False,
                                is_preload=False,  # True,
                                type='test',
                                )

    names = test_dataset.names

    for n in range(len(test_dataset)):
        names[n] = names[n].split('/')[-1].replace('<mask>','').replace('<ext>','jpg')

    index = range(len(test_dataset))

    with multiprocessing.Pool(10) as p:
        p.starmap(doCRF, zip(index,names))




def denseCRF_nonRGB():
    test_dataset = ImageDataset('test-INTER_LINEAR-1024x1024-100064',
                                is_mask=False,
                                is_preload=False,  # True,
                                type='test',
                                )

    names = test_dataset.names

    # Read prediction mask = 100064 * 1024 * 1024 (N * W * H)
    predict = np.memmap(PREDICT_ARR_DIR + '/preds.npy', dtype=np.uint8, mode='r',shape=(100064, CARVANA_H, CARVANA_W))

    NLABELS = 2

    for n in range(len(predict)):

        prob = np.asarray(predict[n], dtype=np.float32)
        prob = cv2.resize(prob, (CARVANA_WIDTH, CARVANA_HEIGHT), interpolation=cv2.INTER_LINEAR)/255.

        name = names[n].split('/')[-1].replace('<mask>','').replace('<ext>','jpg')
        img  = cv2.imread(TEST_DIR + '/' + name)

        prob = np.tile(prob[np.newaxis, :, :], (2, 1, 1))
        prob[1, :, :] = 1 - prob[0, :, :]

        U = unary_from_softmax(prob)

        pairwise_energy = create_pairwise_bilateral(sdims=(10, 10), schan=(0.01,), img=img, chdim=2)

        d = dcrf.DenseCRF2D(CARVANA_WIDTH, CARVANA_HEIGHT, NLABELS)
        d.setUnaryEnergy(U)
        d.addPairwiseEnergy(pairwise_energy, compat=10)

        Q, tmp1, tmp2 = d.startInference()

        for _ in range(50):
            d.stepInference(Q, tmp1, tmp2)
        kl3 = d.klDivergence(Q) / (CARVANA_HEIGHT * CARVANA_WIDTH)
        map_soln3 = np.argmax(Q, axis=0).reshape((CARVANA_HEIGHT, CARVANA_WIDTH))

        plt.imshow(map_soln3)
        plt.waitforbuttonpress()
        # cv2.imwrite(OUT_DIR + '/%s' % name, ((1-map_soln3)*255).astype(np.uint8))

if __name__ == '__main__':

    if 0:
        # denseCRF_nonRGB_multipro()
        denseCRF_nonRGB()