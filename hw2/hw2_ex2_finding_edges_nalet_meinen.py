""" 2 Finding edges """

import numpy as np
from skimage import color, io
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
import pdb

# load image
img = io.imread('bird.jpg')
img = color.rgb2gray(img)


### copy functions myconv2, gauss1d, gauss2d and gconv from exercise 1 ###
import hw2_ex1_linear_filtering_nalet_meinen as ex1

# 2.1
# Gradients
# define a derivative operator
dx = np.array([1, 0, -1]) ### your code should go here ###
dy = np.array([1, 0, -1]).reshape(3, 1) ### your code should go here ###

# convolve derivative operator with a 1d gaussian filter with sigma = 1
# You should end up with 2 1d edge filters,  one identifying edges in the x direction, and
# the other in the y direction
sigma = 1
### your code should go here ###
gdx = ex1.myconv2(dx, ex1.gauss1d(sigma)) ### your code should go here ###
gdy = ex1.myconv2(dy.reshape(1, 3), ex1.gauss1d(sigma))
gdy = gdy.reshape(gdy.shape[1], 1) ### your code should go here ###


# 2.2
# Gradient Edge Magnitude Map
def create_edge_magn_image(image, dx, dy):
    # this function created an eddge magnitude map of an image
    # for every pixel in the image, it assigns the magnitude of gradients
    # INPUTS
    # @image  : a 2D image
    # @gdx     : gradient along x axis
    # @gdy     : geadient along y axis
    # OUTPUTS
    # @ grad_mag_image  : 2d image same size as image, with the magnitude of gradients in every pixel
    # @grad_dir_image   : 2d image same size as image, with the direcrion of gradients in every pixel

    ### your code should go here ###
    if np.ndim(dx) < 2:
        dx = dx.reshape(1, len(dx))
        #print("dx dim < 2")
        #print("img dim < 2")
    Dx = ex1.myconv2(image, dx)
    print("Dx nach convolution: " + str(Dx.shape))
    Dx = Dx[dx.shape[0] - 1:dx.shape[0] - 1 + image.shape[0],
            dx.shape[1] - 1:dx.shape[1] - 1 + image.shape[1]]
    print("Dx nach cut: " + str(Dx.shape))
    if np.ndim(dy) < 2:
        dy = dy.reshape(len(dy), 1)
    Dy = ex1.myconv2(image, dy)
    print("Dy nach convolution: " + str(Dy.shape))
    Dy = Dy[dy.shape[0] - 1:dy.shape[0] - 1 + image.shape[0],
            dy.shape[1] - 1:dy.shape[1] - 1 + image.shape[1]]
    print("Dy nach cut: " + str(Dy.shape))
    # plt.subplot(121)
    # plt.imshow(Dx)
    # plt.subplot(122)
    # plt.imshow(Dy)
    # plt.show()
    E = np.sqrt(Dx * Dx + Dy * Dy)
    E = 255 * E / np.max(E)
    print(E.shape)
    phi = np.arctan2(Dy, Dx)
    # plt.imshow(phi)
    # plt.show()
    grad_mag_image = E
    grad_dir_image = phi

    return grad_mag_image, grad_dir_image


# create an edge magnitude image using the derivative operator
img_edge_mag, img_edge_dir = create_edge_magn_image(img, dx, dy)

# show all together
plt.subplot(121)
plt.imshow(img)
plt.axis('off')
plt.title('Original image')
plt.subplot(122)
plt.imshow(img_edge_mag)
plt.axis('off')
plt.title('Edge magnitude map')
plt.show()

# 2.3
# Edge images of particular directions
def make_edge_map(image, dx, dy):
    # INPUTS
    # @image        : a 2D image
    # @gdx          : gradient along x axis
    # @gdy          : geadient along y axis
    # OUTPUTS:
    # @ edge maps   : a 3D array of shape (image.shape, 8) containing the edge maps on 8 orientations

    ### your code should go here ###
    img_edge_mag, img_edge_dir = create_edge_magn_image(image, dx, dy)
    threshold = 3
    mask = img_edge_mag >= threshold
    edge_maps = np.zeros((image.shape[0], image.shape[1], 8))

    for i in range(0, 8, 1):  # range(-7*math.pi/8, 7*math.pi/8, math.pi/8):
        angle = -7 * np.pi / 8 + 2 * i * np.pi / 8
        print("angle = " + str(angle / np.pi * 180))
        print("range: " + str((angle - np.pi / 8) / np.pi * 180) +
              " - " + str((angle + np.pi / 8) / np.pi * 180))
        edge_maps[:, :, i] = np.where((img_edge_dir > angle - np.pi /
                                       8) * (img_edge_dir < angle + np.pi / 8), 255, 0)
        edge_maps[:, :, i] = edge_maps[:, :, i] * mask
    return edge_maps


# verify with circle image
circle = plt.imread('circle.jpg')
edge_maps = make_edge_map(circle, dx, dy)
edge_maps_in_row = [edge_maps[:, :, i] for i in range(edge_maps.shape[2])]
all_in_row = np.concatenate((edge_maps_in_row), axis=1)
plt.imshow(np.concatenate((circle, all_in_row), axis=1))
plt.title('Circle and edge orientations')
# plt.imshow(np.concatenate(np.dsplit(edge_maps, edge_maps.shape[2]), axis=0))
plt.show()

# now try with original image
edge_maps = make_edge_map(img, dx, dy)
edge_maps_in_row = [edge_maps[:, :, i] for i in range(edge_maps.shape[2])]
all_in_row = np.concatenate((edge_maps_in_row), axis=1)
plt.imshow(np.concatenate((img, all_in_row), axis=1))
plt.title('Original image and edge orientations')
plt.show()


# 2.4
# Edge non max suppresion
def edge_non_max_suppression(img_edge_mag, edge_maps):
    # This function performs non maximum suppresion, in order to reduce the width of the edge response
    # INPUTS
    # @img_edge_mag   : 2d image, with the magnitude of gradients in every pixel
    # @edge_maps      : 3d image, with the edge maps
    # OUTPUTS
    # @non_max_sup    : 2d image with the non max suppresed pixels

    ### your code should go here ###
    non_max_sup = np.zeros(img_edge_mag.shape)
    total_edge_map = edge_maps[:, :, 0] + edge_maps[:, :, 1] + edge_maps[:, :, 2] + edge_maps[:, :, 3] + \
        edge_maps[:, :, 4] + edge_maps[:, :, 5] + \
        edge_maps[:, :, 6] + edge_maps[:, :, 7]
    total_mask = total_edge_map > 0
    img_edge_mag_total_masked = img_edge_mag * total_mask
    plt.imshow(img_edge_mag_total_masked)
    plt.show()
    for i in range(0, 8, 1):  # range(-7*math.pi/8, 7*math.pi/8, math.pi/8):
        angle = -7 * np.pi / 8 + 2 * i * np.pi / 8
        mask = edge_maps[:, :, i] > 0
        toLookAt = img_edge_mag * mask
        img_edge_mag_copy = np.copy(img_edge_mag)

        if i == 0 or i == 4:  # horizontal -> go line by line
            x, y = (toLookAt > 0).nonzero()
            indlist = list(zip(x, y))
            for pos in indlist:
                #plt.imshow(np.concatenate(
                #    (img_edge_mag_copy, toLookAt, edge_maps[:, :, i], total_edge_map), axis=1))
                #plt.title(
                #    'img_edge_mag_copy, toLookAt, and edge_maps[:, :, ' + str(i) + '], and total_edge_map')
                #plt.show()
                if pos[0] >= img_edge_mag.shape[0] - 1 or pos[1] >= img_edge_mag.shape[1] - 1:
                    print("break1 Position: " + str(pos))
                    break
                elif (img_edge_mag_total_masked[pos[0], pos[1] - 1] > img_edge_mag_total_masked[pos]) + (img_edge_mag_total_masked[pos[0], pos[1] + 1] > img_edge_mag_total_masked[pos]):
                    edge_maps[pos[0], pos[1], i] = 0
                else:
                    img_edge_mag_copy[pos[0], pos[1] - 1] = 0
                    img_edge_mag_copy[pos[0], pos[1] + 1] = 0
        if i == 1 or i == 5:  # 45° -> 1
            x, y = (toLookAt > 0).nonzero()
            indlist = list(zip(x, y))
            for pos in indlist:
                if pos[0] >= img_edge_mag.shape[0] - 1 or pos[1] >= img_edge_mag.shape[1] - 1:
                    print("break2 Position: " + str(pos))
                    break
                elif (img_edge_mag_total_masked[pos[0] - 1, pos[1] - 1] > img_edge_mag_total_masked[pos]) + (img_edge_mag_total_masked[pos[0] + 1, pos[1] + 1] > img_edge_mag_total_masked[pos]):
                    edge_maps[pos[0], pos[1], i] = 0
                else:
                    img_edge_mag_copy[pos[0] - 1, pos[1] - 1] = 0
                    img_edge_mag_copy[pos[0] + 1, pos[1] + 1] = 0

        if i == 2 or i == 6:  # 90° ->
            x, y = (toLookAt > 0).nonzero()
            indlist = list(zip(x, y))
            for pos in indlist:
                if pos[0] >= img_edge_mag.shape[0] - 1 or pos[1] >= img_edge_mag.shape[1] - 1:
                    print("break3 Position: " + str(pos))
                    break
                elif (img_edge_mag_total_masked[pos[0] - 1, pos[1]] > img_edge_mag_total_masked[pos]) + (img_edge_mag_total_masked[pos[0] + 1, pos[1]] > img_edge_mag_total_masked[pos]):
                    edge_maps[pos[0], pos[1], i] = 0
                else:
                    img_edge_mag_copy[pos[0] - 1, pos[1]] = 0
                    img_edge_mag_copy[pos[0] + 1, pos[1]] = 0
        if i == 3 or i == 7:  # 45° -> 2
            x, y = (toLookAt > 0).nonzero()
            indlist = list(zip(x, y))
            for pos in indlist:
                if pos[0] >= img_edge_mag.shape[0] - 1 or pos[1] >= img_edge_mag.shape[1] - 1:
                    print("break4 Position: " + str(pos))
                    break
                elif (img_edge_mag_total_masked[pos[0] - 1, pos[1] + 1] > img_edge_mag_total_masked[pos]) + (img_edge_mag_total_masked[pos[0] + 1, pos[1] - 1] > img_edge_mag_total_masked[pos]):
                    edge_maps[pos[0], pos[1], i] = 0
                else:
                    img_edge_mag_copy[pos[0] - 1, pos[1] + 1] = 0
                    img_edge_mag_copy[pos[0] + 1, pos[1] - 1] = 0
        #plt.imshow(edge_maps[:, :, i])
        #plt.show()
        non_max_sup = non_max_sup + edge_maps[:, :, i]

    return non_max_sup


# show the result
img_non_max_sup = edge_non_max_suppression(img_edge_mag, edge_maps)
plt.imshow(np.concatenate((img, img_edge_mag, img_non_max_sup), axis=1))
plt.title('Original image, magnitude edge, and max suppresion')
plt.show()


# # 2.5
# # Canny edge detection (BONUS)
# def canny_edge(image, sigma=2):
#     # implementation of canny edge detector
#     # INPUTS
#     # @image      : 2d image
#     # @sigma      : sigma parameter of gaussian
#     # OUTPUTS
#     # @canny_img  : 2d image of size same as image, with the result of the canny edge detection

#     ### your code should go here ###

#     return canny_img

# canny_img = canny_edge(img)
# plt.imshow(np.concatenate((img, canny_img), axis=1))
# plt.title('Original image and canny edge detector')
# plt.show()
