import numpy as np
from utils import *
from constants import *

def corner_detector(image, blockSize, kSize):
    """image should be grayscale"""
    height = image.shape[0]
    width = image.shape[1]
  
    gx = image[:, 1:width] - image[:, 0:width-1]
    zero = np.zeros((height, 1)) 
    gx = np.append(gx, zero, axis=1)
    gy = image[1:height, :] - image[0:height-1, :]
    zero = np.zeros((1, width))
    gy = np.append(gy, zero, axis=0)

    I_xx = np.multiply(gx, gx)
    I_xy = np.multiply(gx, gy)
    I_yy = np.multiply(gy, gy)

    gkern = gaussWin(kSize);
    W_xx = conv2d(I_xx, gkern, 'same')
    W_xy = conv2d(I_xy, gkern, 'same')
    W_yy = conv2d(I_yy, gkern, 'same')

    print('convo finish...')

    eig_min = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            W = np.array([[W_xx[i][j], W_xy[i][j]], [W_xy[i][j], W_yy[i][j]]])
            d, w = np.linalg.eig(W)
            eig_min[i][j] = min(d)

    print('eig_min finish...')

    for i in range(0, height, kSize):
        for j in range(0, width, kSize):
            size_a = min(i+kSize, height)
            size_b = min(j+kSize, width)
            E = eig_min[i:size_a, j:size_b]
            max_eig = np.amax(E)
            E[E < max_eig] = 0
            eig_min[i:size_a, j:size_b] = E

    print('eig_max finish...')

    cutoff = min(Constant.NUM_FEATURE_TO_TRACK, height * width - 1)
    cutoff_eig = np.sort(eig_min, axis=None)[::-1][cutoff-1]
    bitmap = np.zeros((height, width))
    bitmap[eig_min >= cutoff_eig] = 1

    return bitmap

