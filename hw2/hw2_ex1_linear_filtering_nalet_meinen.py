import pdb
import time
from mpl_toolkits.mplot3d import Axes3D
""" 1 Linear filtering """

# Imports
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

if __name__ == '__main__':
    img = plt.imread('cat.jpg').astype(np.float32)

    plt.imshow(img)
    plt.axis('off')
    plt.title('original image')
    plt.show()

# 1.1


def boxfilter(n):
    # this function returns a box filter of size nxn

    ### your code should go here ###
    return np.ones((n, n)) / (n*n)

# 1.2
# Implement full convolution


def myconv2(image, filt):
    # This function performs a 2D convolution between image and filt, image being a 2D image. This
    # function should return the result of a 2D convolution of these two images. DO
    # NOT USE THE BUILT IN SCIPY CONVOLVE within this function. You should code your own version of the
    # convolution, valid for both 2D and 1D filters.
    # INPUTS
    # @ image         : 2D image, as numpy array, size mxn
    # @ filt          : 1D or 2D filter of size kxl
    # OUTPUTS
    # img_filtered    : 2D filtered image, of size (m+k-1)x(n+l-1)

    ### your code should go here ###
    # adding checks for 2D
    if np.ndim(filt) < 2:
        filt = filt.reshape(1, len(filt))
    if np.ndim(image) < 2:
        image = image.reshape(1, len(image))
    if np.ndim(filt) >= 2:
        filt = np.flip(filt, 0)
        filt = np.flip(filt, 1)
        filtered_img = np.zeros(
            (image.shape[0] + filt.shape[0] - 1, image.shape[1] + filt.shape[1] - 1))
        image = np.pad(image, ((filt.shape[0] - 1, filt.shape[0] - 1),
                               (filt.shape[1] - 1, filt.shape[1] - 1)), mode='constant')
        for row in range(filtered_img.shape[0]):
            for col in range(filtered_img.shape[1]):
                filtered_img[row, col] = np.sum(np.multiply(
                    image[row:row + filt.shape[0], col:col + filt.shape[1]], filt))
    return filtered_img


if __name__ == '__main__':
    # 1.3
    # create a boxfilter of size 10 and convolve this filter with your image - show the result
    bsize = 10

    ### your code should go here ###
    plt.imshow(myconv2(img, boxfilter(bsize)))
    plt.show()

# 1.4
# create a function returning a 1D gaussian kernel


def gauss1d(sigma, filter_length=20):
    # INPUTS
    # @ sigma         : sigma of gaussian distribution
    # @ filter_length : integer denoting the filter length, default is 10
    # OUTPUTS
    # @ gauss_filter  : 1D gaussian filter

    ### your code should go here ###
    if filter_length % 2 == 0:
        filter_length += 1
    x = np.linspace(int(-filter_length / 2),
                    int(filter_length / 2), filter_length)
    gauss_filter = np.exp(-(x*x) / (2 * sigma*sigma))
    gauss_filter = gauss_filter / sum(gauss_filter)
    return gauss_filter


# 1.5
# create a function returning a 2D gaussian kernel
def gauss2d(sigma, filter_size=20):
    # INPUTS
    # @ sigma         : sigma of gaussian distribution
    # @ filter_size   : integer denoting the filter size, default is 10
    # OUTPUTS
    # @ gauss2d_filter  : 2D gaussian filter

    ### your code should go here ###
    g1d = gauss1d(sigma, filter_size)
    return myconv2(g1d, g1d.reshape(len(g1d), 1))


if __name__ == '__main__':
    # Display a plot using sigma = 3
    sigma = 3

    ### your code should go here ###
    gauss2D = gauss2d(sigma)

    plt.figure()
    ax = plt.gca(projection='3d')
    x2d, y2d = np.meshgrid(np.linspace(
        int(-21 / 2), int(21 / 2), 21), np.linspace(int(-21 / 2), int(21 / 2), 21))
    ax.plot_surface(x2d, y2d, gauss2D, cmap=plt.get_cmap("jet"))
    plt.show()

# 1.6
# Convoltion with gaussian filter


def gconv(image, sigma):
    # INPUTS
    # image           : 2d image
    # @ sigma         : sigma of gaussian distribution
    # OUTPUTS
    # @ img_filtered  : filtered image with gaussian filter

    ### your code should go here ###
    return myconv2(image, gauss2d(sigma, 30))


if __name__ == '__main__':
    # run your gconv on the image for sigma=3 and display the result
    sigma = 3

    ### your code should go here ###
    plt.imshow(gconv(img, sigma))
    plt.show()

    # 1.7
    # Convolution with a 2D Gaussian filter is not the most efficient way
    # to perform Gaussian convolution with an image. In a few sentences, explain how
    # this could be implemented more efficiently and why this would be faster.
    #
    # HINT: How can we use 1D Gaussians?

    ### your explanation should go here ###
    # For each pixel the operations for convolution are n*n, n equals the size, width,
    # and height. Usually, this can be made faster with performing a 1d convolution in
    # both directions, horizontal and vertical, resulting in 2*n operations for a pixel.
    # Convolutions can then be split up. Looking at SVD, with one non-zero
    # value the separations in 1d can be made.

    # 1.8
    # Computation time vs filter size experiment
    size_range = np.arange(3, 100, 5)
    t1d = []
    t2d = []
    for size in size_range:

        ### your code should go here ###
        f1d = gauss2d(sigma, size)[int(size / 2), :]
        # 1d
        start = time.time()
        myconv2(img, gauss2d(sigma, size))
        t2d.append(time.time() - start)
        # 2d
        start = time.time()
        myconv2(myconv2(img, f1d), np.transpose(f1d))
        t1d.append(time.time() - start)

    # plot the comparison of the time needed for each of the two convolution cases
    plt.plot(size_range, t1d, label='1D filtering')
    plt.plot(size_range, t2d, label='2D filtering')
    plt.xlabel('Filter size')
    plt.ylabel('Computation time')
    plt.legend(loc=0)
    plt.show()
