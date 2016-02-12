import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib.pylab import imshow, figure, show, savefig, title
import matplotlib.cm as cm
from scipy.ndimage import filters as filters
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import math
import os
# import cv2

__author__ = 'jhh283'

IMAGE = [[4., 1., 6., 1., 3.],
         [3., 2., 7., 7., 2.],
         [2., 5., 7., 3., 7.],
         [1., 4., 7., 1., 3.],
         [0., 1., 6., 4., 4.]]


def MeanFilter(img):
    # img = np.array(img)
    mean33 = [[1., 1., 1.],
              [1., 1., 1.],
              [1., 1., 1.]]
    win = np.array(mean33) / 9
    # assumes zero pad. need to think about
    return filters.convolve(img, win)


def MedianFilter(img):
    return filters.median_filter(img, size=(3, 3))


def SobelFilter(img):
    x_filter = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    y_filter = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    g_x = filters.convolve(img, x_filter)
    g_y = filters.convolve(img, y_filter)
    # f_x = filters.sobel(img, axis=1, mode='constant', cval=0.0)
    # f_y =  filters.sobel(img, axis=0, mode='constant', cval=0.0)
    g_mag = np.hypot(g_x, g_y)
    g_dir = np.arctan2(g_y, g_x)
    return g_dir, g_mag


def DistanceFilter(img):
    dfilter_33 = [[0.0625, 0.125, 0.0625],
                  [0.125, 0.25, 0.125],
                  [0.0625, 0.125, 0.0625]]
    win = np.array(dfilter_33).astype(float)
    return filters.convolve(img, win)


def PixelFilter(img):
    pfilter_33 = [[-1, -1, -1],
                  [-1, 8, -1],
                  [-1, -1, -1]]
    win = np.array(pfilter_33).astype(float)
    return filters.convolve(img, win)


def ComboFilter(img):
    identity = [[0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]]
    identity = np.array(identity).astype(float)
    kernel = DistanceFilter(identity)
    kernel = PixelFilter(kernel)
    # print kernel
    return filters.convolve(img, kernel)


def AddNoise(img):
    row, col, ch = img.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = img + gauss
    return noisy


def RunFilters(img_path, filename):
    img = io.imread(img_path, as_grey=True)
    # print img[0][1]
    # pixels = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    # print pixels[0][1]
    dfilt_img = DistanceFilter(img)
    print dfilt_img
    # print dfilt_img.shape
    pfilt_img = PixelFilter(img)
    cfilt_img = ComboFilter(img)
    # dfilt_img = np.reshape(pixels, (img.shape[0], img.shape[1], img.shape[2]))

    print 'Distance'
    figure(1)
    imshow(dfilt_img, cmap=cm.Greys_r)
    title("Distance Filtered - No Noise")
    savefig('Q1/' + filename + '_distance_nonoise.png')

    print 'Pixel'
    figure(1)
    imshow(pfilt_img, cmap=cm.Greys_r)
    title("Pixel Filtered - No Noise")
    savefig('Q1/' + filename + '_pixel_nonoise.png')

    print 'Combo'
    figure(1)
    imshow(cfilt_img, cmap=cm.Greys_r)
    title("Combo Filtered - No Noise")
    savefig('Q1/' + filename + '_combo_nonoise.png')


def UnsharpMasking(img, gauss_sd):
    gauss = filters.gaussian_filter(img, gauss_sd)
    return (2 * img) - gauss


# def blurThenUnsharpen(img_path, sds, filename):
#     print filename
#     img = io.imread(img_path)
#     for sd in sds:
#         img = filters.gaussian_filter(img, sd)
#     figure(1)
#     imshow(img)
#     title("Blurred Image")
#     savefig('Q1/' + filename + '_blurred.png')
#     for sd in sds:
#         unsharpened = UnsharpMasking(img, sd)
#         figure(1)
#         imshow(unsharpened)
#         title("Unsharpened where sigma is " + str(sd))
#         savefig('Q1/' + filename + '_' + str(sd) + '.png')


def blurThenUnsharpen(img_path, sds, filename):
    img = io.imread(img_path, as_grey=True)
    # for sd in sds:
    #     img = filters.gaussian_filter(img, sd)
    # figure(1)
    # imshow(img)
    # title("Blurred Image")
    # savefig('Q1/' + filename + '_blurred.png')
    for sd in sds:
        blurred = filters.gaussian_filter(img, sd)
        unsharpened = UnsharpMasking(blurred, sd)
        figure(1)
        imshow(unsharpened, cmap=cm.Greys_r)
        title("Unsharpened where sigma is " + str(sd))
        savefig('Q1/' + filename + '_' + str(sd) + '.png')


if __name__ == '__main__':
    if not os.path.exists("Q1"):
        os.makedirs("Q1")
    # img = np.array(IMAGE)
    # img = img.astype('int32')
    # mean = MeanFilter(IMAGE)
    # print mean
    # median = MedianFilter(IMAGE)
    # print median
    # print mean - median
    # g_mag, g_dir = SobelFilter(IMAGE)
    # print g_mag[2][2]
    # print g_dir[2][2]

    images = ['Images/Q1/cameraman.jpg',
              'Images/Q1/house.jpg',
              'Images/Q1/lena.jpg']
    for image in images:
        fn = image.split('/')[-1]
        print fn
        RunFilters(image, fn)

    # sds = [0.75, 2.5]
    # for image in images:
    #     fn = image.split('/')[-1]
    #     print fn
    #     blurThenUnsharpen(image, sds, fn)
