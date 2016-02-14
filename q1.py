import matplotlib as mpl
mpl.use('TkAgg')
# from matplotlib.pylab import imshow, figure, show, savefig, title
import matplotlib.cm as cm
from scipy.ndimage import filters as filters
from skimage import io
import numpy as np
import math
import os
import utils as custom_utils

__author__ = 'jhh283'

IMAGE = [[4., 1., 6., 1., 3.],
         [3., 2., 7., 7., 2.],
         [2., 5., 7., 3., 7.],
         [1., 4., 7., 1., 3.],
         [0., 1., 6., 4., 4.]]
# BLUR33 = [[0.0625, 0.125, 0.0625],
#           [0.125, 0.25, 0.125],
#           [0.0625, 0.125, 0.0625]]
BLUR33 = [[0.102059, 0.115349, 0.102059],
          [0.115349, 0.130371, 0.115349],
          [0.102059, 0.115349, 0.102059]]

DEFAULT33 = np.zeros((3, 3))


# add noise
def MeanFilter(img):
    # img = np.array(img)
    mean33 = [[1., 1., 1.],
              [1., 1., 1.],
              [1., 1., 1.]]
    win = np.array(mean33) / 9
    # assumes zero pad. need to think about
    # print filters.convolve(img, win, mode='constant', cval=0)
    # print utils.convolve(img, win)
    return custom_utils.convolve(img, win, custom_utils.sum_cross_mtx)


def MedianFilter(img):
    # out = filters.median_filter(img, size=(3, 3), mode='constant', cval=0)
    return custom_utils.convolve(img, DEFAULT33, custom_utils.median_cross_mtx)


# http://homepages.inf.ed.ac.uk/rbf/HIPR2/sobel.htm
def SobelFilter(img):
    x_filter = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    y_filter = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    # f_x = filters.sobel(img, axis=1, mode='constant', cval=0.0)
    # f_y = filters.sobel(img, axis=0, mode='constant', cval=0.0)
    g_x = custom_utils.convolve(img, x_filter, custom_utils.sum_cross_mtx)
    g_y = custom_utils.convolve(img, y_filter, custom_utils.sum_cross_mtx)
    u_mag = np.hypot(g_x, g_y)
    u_dir = np.arctan2(g_y, g_x)
    return u_mag, u_dir


def DistanceFilter(img):
    win = np.array(BLUR33).astype('float32')
    blurred = filters.convolve(img, win, mode='constant', cval=0)
    return blurred


def GeneratePixelGauss(img, mid):
    # can play with this
    # inten_sig = 0.8
    # inten_sig = 2.0
    inten_sig = 0.08
    weight = np.exp(-(img - img[mid[0]][mid[1]])**2 / inten_sig)
    return weight


def get_middle(mtx):
    return math.ceil(mtx.shape[0] / 2), math.ceil(mtx.shape[1] / 2)


def pixel_func(img, kernel):
    mid = get_middle(kernel)
    weight = GeneratePixelGauss(img, mid)
    weight_norm = custom_utils.norml_mtx(weight)
    apply_weights = custom_utils.sum_cross_mtx(img, weight)
    return apply_weights


def PixelFilter(img):
    out = custom_utils.convolve(img, DEFAULT33, pixel_func)
    return out


def combo_func(img, kernel):
    mid = get_middle(kernel)
    pixel = GeneratePixelGauss(img, mid)
    dist = np.array(BLUR33).astype('float32')
    weight = pixel * dist
    weight_norm = custom_utils.norml_mtx(weight)
    apply_weights = custom_utils.sum_cross_mtx(img, weight)
    return apply_weights


def ComboFilter(img):
    out = custom_utils.convolve(img, DEFAULT33, combo_func)
    return out


def RunFilters(img_path, filename, plot_title):
    img = io.imread(img_path, as_grey=True)
    dfilt_img = DistanceFilter(img)
    pfilt_img = PixelFilter(img)
    cfilt_img = ComboFilter(img)

    custom_utils.plot_im_grey(dfilt_img,
                              plot_title + "Distance Filtered",
                              filename + '_distance.png')

    custom_utils.plot_im_grey(pfilt_img,
                              plot_title + 'Pixel Filtered',
                              filename + '_pixel.png')

    custom_utils.plot_im_grey(cfilt_img,
                              plot_title + 'Combo Filtered',
                              filename + '_combo.png')


def UnsharpMasking(img, gauss_sd):
    gauss = filters.gaussian_filter(img, gauss_sd)
    return (2 * img) - gauss


def blurThenUnsharpen(img_path, sds, filename):
    img = io.imread(img_path, as_grey=True)
    for sd in sds:
        print 'Sigma:', str(sd)
        blurred = filters.gaussian_filter(img, sd)
        unsharpened = UnsharpMasking(blurred, sd)
        figure(1)
        imshow(unsharpened, cmap=cm.Greys_r)
        title("Unsharpen Masking where sigma is " + str(sd))
        savefig('Q1/part5_' + filename + '_' + str(sd) + '.png')


if __name__ == '__main__':
    if not os.path.exists("Q1"):
        os.makedirs("Q1")

    # Part 1
    # mean = MeanFilter(IMAGE)
    # print 'Mean Filter Output'
    # print mean

    # Part 2
    # median = MedianFilter(IMAGE)
    # print 'Median Filter Output'
    # print median
    # print 'Median Median Difference'
    # print mean - median

    # Part 3
    # g_mag, g_dir = SobelFilter(IMAGE)
    # print 'Gradiant Magnitude @ Center', g_mag[2][2]
    # print 'Gradiant Direction @ Center', g_dir[2][2]

    images = ['Images/Q1/cameraman.jpg',
              'Images/Q1/house.jpg',
              'Images/Q1/lena.jpg']

    # Part 4
    for image in images:
        fn = image.split('/')[-1]
        print fn
        filename = 'Q1/part4_nonoise_' + fn
        plot_title = 'No Noise - '
        RunFilters(image, filename, plot_title)
        noisy = custom_utils.AddNoise(image)
        filename = 'Q1/' + fn + '_noisy.png'
        custom_utils.plot_im_grey(noisy,
                                  "Noise - No Filter",
                                  filename)
        filename = 'Q1/part4_noise_' + fn
        plot_title = 'Gaussian Noise - '
        RunFilters(image, filename)

    # pixel = PixelFilter(IMAGE)
    # pixel = ComboFilter(IMAGE)
    # print IMAGE
    # print pixel

    # Part 5
    # sds = [0.75, 2.5]
    # for image in images:
    #     fn = image.split('/')[-1]
    #     print fn
    #     blurThenUnsharpen(image, sds, fn)
