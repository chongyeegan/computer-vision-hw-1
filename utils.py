import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage import util

__author__ = 'jhh283'


# function which normalizes the values in a provided matrix
def norml_mtx(matrix):
    summed = np.sum(matrix)
    if summed > 0.:
        return matrix / summed
    else:
        return matrix


# calculate minimum padding (one side of the image)
def getSmallPad(MaskDim):
    pad = int(MaskDim / 2) - (1 - (MaskDim % 2))
    if (pad < 0):
        pad = 0
    return pad


# calculate total dimensional padding (both sides)
def getTotalPad(MaskDim, ImgDim):
    smallPad = getSmallPad(MaskDim)
    bigPad = MaskDim / 2
    return ImgDim + smallPad + bigPad, smallPad, bigPad


# fetch the neighbors for a particular value in a matrix
def fetch_neighbors(row, col, padded, ker_w, ker_h):
    neighbors = padded[row:row + ker_w, col:col + ker_h]
    return neighbors


# perform cross correlation
# requires the same size n and kernel
def sum_cross_mtx(n, kernel):
    summed = 0
    for (row, col), value in np.ndenumerate(n):
        summed += n[row, col] * kernel[row, col]
    return summed


# find the median value of a matrix
# requires the same size n and kernel
def median_cross_mtx(n, kernel):
    return np.median(n)


# runs a matrix operation on a large image based on the provided fun function
# by default will perform the flips expected by convolution
# implementation of zero pad boundary handling
def handle_window(img, kernel, fun, flip=True):
    img = np.array(img, dtype=np.float)
    kernel = np.array(kernel, dtype=np.float)
    result = np.array(img, copy=True)

    total_width, l_pad, r_pad = getTotalPad(kernel.shape[0], img.shape[0])
    total_height, t_pad, b_pad = getTotalPad(kernel.shape[1], img.shape[1])
    padded = np.zeros((total_width, total_height), dtype=np.float)
    padded[l_pad:-r_pad, t_pad:-b_pad] = img

    flipped = kernel
    if flip:
        flipped = np.flipud(flipped)
        flipped = np.fliplr(flipped)
    flipped = norml_mtx(flipped)

    for (row, col), value in np.ndenumerate(img):
        neighbors = fetch_neighbors(row, col, padded, kernel.shape[0], kernel.shape[1])
        result[row, col] = fun(neighbors, kernel)

    return result


# add noise helper function - uses gaussian by default
def AddNoise(img):
    # mean = 0
    # sigma = 0.045
    # gauss = np.random.normal(mean, sigma, img.shape)
    # gauss = gauss.reshape(img.shape[0], img.shape[1])
    # noisy = img + gauss
    noisy = util.random_noise(img, mode='gaussian', var=0.002)
    # noisy = util.random_noise(img, mode='gaussian', var=0.0015)

    # noisy = util.random_noise(img, mode='s&p')
    return noisy


# plot grayscale image helper
def plot_im_grey(img, title, fn):
    fig = plt.figure(1)
    plt.imshow(img, cmap=cm.Greys_r)
    plt.title(title)
    plt.savefig(fn)
    plt.close(fig)


# plot image helper
def plot_im(img, title, fn):
    fig = plt.figure(1)
    plt.imshow(img)
    plt.title(title)
    plt.savefig(fn)
    plt.close(fig)
