import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib.pylab import imshow, figure, savefig, title
from scipy.cluster.vq import kmeans, vq
from skimage import io, color
import numpy as np
import matplotlib.pyplot as plt
import os

__author__ = 'jhh283'

MAX_COLOR_VAL = 255.0


def cluster_pixels(pixels, k, img_dim):
    codebook, _ = kmeans(pixels, k)
    quantz, _ = vq(pixels, codebook)
    cent_id = np.reshape(quantz, img_dim)
    clustered = codebook[cent_id]
    return clustered


def RGB(img, k, filename):
    # print 'rgb'
    pixels = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    pixels = pixels.astype('float32')

    clustered = cluster_pixels(pixels, k, (img.shape[0], img.shape[1]))
    clustered = clustered.astype('uint8')
    figure(1)
    imshow(clustered)
    title("RGB quantization where k is " + str(k))
    savefig('Q2/' + filename + '_RGB.png')
    return clustered


def LAB(img, k, filename):
    # print 'lab'
    img = img * 1.0 / MAX_COLOR_VAL
    pixels_lab = color.rgb2lab(img)
    L = pixels_lab[:, :, 0]
    pixels_l = np.reshape(L, (L.shape[0] * L.shape[1], 1))

    clustered = cluster_pixels(pixels_l, k, (L.shape[0], L.shape[1]))
    pixels_lab[:, :, 0] = clustered[:, :, 0]
    quanted_img = color.lab2rgb(pixels_lab) * MAX_COLOR_VAL
    quanted_img = quanted_img.astype('uint8')

    figure(1)
    imshow(quanted_img)
    title("LAB quantization where k is " + str(k))
    savefig('Q2/' + filename + '_LAB.png')
    return quanted_img


def SSD(img1, img2):
    diff = img1 - img2
    diff2 = diff * diff
    return diff2.sum()


def HIST(img, k, filename):
    img = img * 1.0 / MAX_COLOR_VAL
    pixels_lab = color.rgb2lab(img)
    L = pixels_lab[:, :, 0]
    pixels_l = np.reshape(L, (L.shape[0] * L.shape[1], 1))

    plt.figure()
    plt.hist(pixels_l, bins=100)
    plt.title("L before Quantization")
    plt.savefig('Q2/' + filename + '_histPre.png')

    clustered = cluster_pixels(pixels_l, k, (L.shape[0], L.shape[1]))
    clustered = clustered.flatten()

    plt.figure()
    plt.hist(clustered, bins=100)
    plt.title("L after Quantization")
    plt.savefig('Q2/' + filename + '_histPost.png')


def runPrePost(filename, k, name):
    img = io.imread(filename)
    qRGB = RGB(img, k, name)
    qLAB = LAB(img, k, name)
    ssd_rgb = SSD(img, qRGB)
    ssd_lab = SSD(img, qLAB)
    print "SSD of RGB quantization: ", ssd_rgb
    print "SSD of L channel quantization: ", ssd_lab
    HIST(img, k, name)


if __name__ == '__main__':
    if not os.path.exists("Q2"):
        os.makedirs("Q2")

    images = ['Images/Q2/colorful1.jpg',
              'Images/Q2/colorful2.jpg',
              'Images/Q2/colorful3.jpg']
    clusters = [4, 6]
    for image in images:
        fn = image.split('/')[-1]
        for k in clusters:
            name = fn + '_k' + str(k)
            print name
            runPrePost(image, k, name)
