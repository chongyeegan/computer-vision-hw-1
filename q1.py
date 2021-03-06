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


# provided image matrix from hw
IMAGE = [[4., 1., 6., 1., 3.],
         [3., 2., 7., 7., 2.],
         [2., 5., 7., 3., 7.],
         [1., 4., 7., 1., 3.],
         [0., 1., 6., 4., 4.]]

# this blur kernel has sigma = 2
BLUR33 = [[0.102059, 0.115349, 0.102059],
          [0.115349, 0.130371, 0.115349],
          [0.102059, 0.115349, 0.102059]]

# this blur kernel has sigma = 1
# BLUR33 = [[0.077847, 0.123317, 0.077847],
#           [0.123317, 0.195346, 0.123317],
#           [0.077847, 0.123317, 0.077847]]

# default 3x3 matrix of zeros
DEFAULT33 = np.zeros((3, 3))


# function applies a 3x3 mean filter on a provided image
def MeanFilter(img):
    # img = np.array(img)
    mean33 = [[1., 1., 1.],
              [1., 1., 1.],
              [1., 1., 1.]]
    win = np.array(mean33) / 9
    # assumes zero pad. need to think about
    # print filters.convolve(img, win, mode='constant', cval=0)
    return custom_utils.handle_window(img, win, custom_utils.sum_cross_mtx)


# function applies a 3x3 median filter on a provided image
def MedianFilter(img):
    # out = filters.median_filter(img, size=(3, 3), mode='constant', cval=0)
    return custom_utils.handle_window(img, DEFAULT33, custom_utils.median_cross_mtx, False)


# function applies a 3x3 sobel filter on a provided image (in both directions)
# calculates gradiant magnitude and direction and returns that
def SobelFilter(img):
    x_filter = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    y_filter = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    # f_x = filters.sobel(img, axis=1, mode='constant', cval=0.0)
    # f_y = filters.sobel(img, axis=0, mode='constant', cval=0.0)
    g_x = custom_utils.handle_window(img, x_filter, custom_utils.sum_cross_mtx)
    g_y = custom_utils.handle_window(img, y_filter, custom_utils.sum_cross_mtx)
    u_mag = np.hypot(g_x, g_y)
    u_dir = np.arctan2(g_y, g_x)
    return u_mag, u_dir


# calls a distance-based filter on a provided image
# the distance filter used is a 3 x 3 gaussian blur kernel listed above
def DistanceFilter(img):
    win = np.array(BLUR33).astype(np.float)
    # normalize the window and apply convolution with zero padding
    win = custom_utils.norml_mtx(win)
    blurred = filters.convolve(img, win, mode='constant', cval=0)
    # clip to ensure grayscale values
    blurred = np.clip(blurred, 0, 1)
    return blurred


# helper function for generating gaussian sub-windows based on pixel intensities off a provided image
def GeneratePixelGauss(img, mid):
    inten_sig = 0.1
    # inten_sig = 0.5
    # inten_sig = 1
    # inten_sig = 2
    weight = np.exp(-((img - img[mid[0]][mid[1]])**2) / (2 * (inten_sig)**2))
    return weight


# returns the 'center' of a provided image
def get_middle(mtx):
    return math.floor(mtx.shape[0] / 2), math.floor(mtx.shape[1] / 2)


# run a pixel-value based filter on a provided image
# by default uses the GeneratePixelGauss function to generate sub filter windows for every window of the image
def pixel_func(img, kernel):
    mid = get_middle(kernel)
    weight = GeneratePixelGauss(img, mid)
    # need to normalize and apply cross correlation
    weight_norm = custom_utils.norml_mtx(weight)
    apply_weights = custom_utils.sum_cross_mtx(img, weight_norm)
    return apply_weights


# calls a pixel value-based filter on a provided image
def PixelFilter(img):
    out = custom_utils.handle_window(img, DEFAULT33, pixel_func, False)
    out = np.clip(out, 0, 1)
    return out


# run a value and distance based filter on provided subimage
def combo_func(img, kernel):
    mid = get_middle(kernel)
    # generate pixel gaussian
    pixel = GeneratePixelGauss(img, mid)
    # combine with distance gaussian
    dist = np.array(BLUR33).astype(np.float)
    weight = pixel * dist
    # apply the new window
    weight_norm = custom_utils.norml_mtx(weight)
    apply_weights = custom_utils.sum_cross_mtx(img, weight_norm)
    return apply_weights


# calls a pixel and distance-based filter on a provided image
def ComboFilter(img):
    out = custom_utils.handle_window(img, DEFAULT33, combo_func, False)
    out = np.clip(out, 0, 1)
    return out


# helper that runs all 3 filters on the provided images
def RunFilters(img, filename, plot_title):
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


# implementation of unsharp masking based on provided formula
def UnsharpMasking(img, gauss_sd):
    gauss = filters.gaussian_filter(img, gauss_sd)
    return (2 * img) - gauss


# helper that executes "blur" then "unsharp masking" on provided images
def blurThenUnsharpen(img_path, sds, filename):
    img = io.imread(img_path, as_grey=True)
    for sd in sds:
        print 'Sigma:', str(sd)
        blurred = filters.gaussian_filter(img, sd)
        custom_utils.plot_im_grey(blurred,
                                  'Blurred where sigma is ' + str(sd),
                                  'Q1/blurred_' + filename + '_' + str(sd) + '.png')
        unsharpened = UnsharpMasking(blurred, sd)
        unsharpened = np.clip(unsharpened, 0, 1)
        custom_utils.plot_im_grey(unsharpened,
                                  'Unsharpen Masking where sigma is ' + str(sd),
                                  'Q1/part5_' + filename + '_' + str(sd) + '.png')

if __name__ == '__main__':
    if not os.path.exists("Q1"):
        os.makedirs("Q1")

    # Part 1
    mean = MeanFilter(IMAGE)
    print 'Mean Filter Output'
    print mean

    # Part 2
    median = MedianFilter(IMAGE)
    print 'Median Filter Output'
    print median
    print 'Median Median Difference'
    print mean - median

    # Part 3
    g_mag, g_dir = SobelFilter(IMAGE)
    print 'Gradiant Magnitude @ Center', g_mag[2][2]
    print 'Gradiant Direction @ Center', g_dir[2][2]

    images = ['Images/Q1/cameraman.jpg',
              'Images/Q1/house.jpg',
              'Images/Q1/lena.jpg']

    # Part 4
    for image in images:
        img = io.imread(image, as_grey=True)
        fn = image.split('/')[-1]
        print fn
        filename = 'Q1/part4_nonoise_' + fn
        plot_title = 'No Noise - '
        RunFilters(img, filename, plot_title)
        noisy = custom_utils.AddNoise(img)
        filename = 'Q1/' + fn + '_noisy.png'
        custom_utils.plot_im_grey(noisy,
                                  "Noise - No Filter",
                                  filename)
        filename = 'Q1/part4_noise_' + fn
        plot_title = 'Gaussian Noise - '
        RunFilters(noisy, filename, plot_title)

    # Part 5
    sds = [0.75, 2.5]
    for image in images:
        fn = image.split('/')[-1]
        print fn
        blurThenUnsharpen(image, sds, fn)
