import hw2_ex1_linear_filtering_nalet_meinen as ex1
import pdb
""" 2 Finding edges """

import numpy as np
from skimage import color, io, feature
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

# load image
img = io.imread('bird.jpg')
img = color.rgb2gray(img)


### copy functions myconv2, gauss1d, gauss2d and gconv from exercise 1 ###
#see import

# 2.1
# Gradients
# define a derivative operator
dx = np.array([1, 0, -1])  # your code should go here ###
dy = np.array([1, 0, -1]).reshape(3, 1)  # your code should go here ###

# convolve derivative operator with a 1d gaussian filter with sigma = 1
# You should end up with 2 1d edge filters,  one identifying edges in the x direction, and
# the other in the y direction
sigma = 1
### your code should go here ###
gdx = ex1.myconv2(dx, ex1.gauss1d(sigma))  # your code should go here ###
gdy_tmp = ex1.myconv2(dy.reshape(1, 3), ex1.gauss1d(sigma))
gdy = gdy_tmp.reshape(gdy_tmp.shape[1], 1)  # your code should go here ###


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
    Dx = ex1.myconv2(image, dx)
    Dx = Dx[dx.shape[0] - 1:dx.shape[0] - 1 + image.shape[0],
            dx.shape[1] - 1:dx.shape[1] - 1 + image.shape[1]]
    if np.ndim(dy) < 2:
        dy = dy.reshape(len(dy), 1)
    Dy = ex1.myconv2(image, dy)
    Dy = Dy[dy.shape[0] - 1:dy.shape[0] - 1 + image.shape[0],
            dy.shape[1] - 1:dy.shape[1] - 1 + image.shape[1]]
    E = np.sqrt(Dx * Dx + Dy * Dy)
    E = 255 * E / np.max(E)
    phi = np.arctan2(Dy, Dx)
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

    for i in range(0, 8, 1):
        angle = -7 * np.pi / 8 + 2 * i * np.pi / 8
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
    for i in range(0, 8, 1):
        mask = edge_maps[:, :, i] > 0
        toLookAt = img_edge_mag * mask
        img_edge_mag_copy = np.copy(img_edge_mag)

        if i == 0 or i == 4:
            x, y = (toLookAt > 0).nonzero()
            indlist = list(zip(x, y))
            for pos in indlist:
                if pos[0] >= img_edge_mag.shape[0] - 1 or pos[1] >= img_edge_mag.shape[1] - 1:
                    break
                elif (img_edge_mag_total_masked[pos[0], pos[1] - 1] > img_edge_mag_total_masked[pos]) + (img_edge_mag_total_masked[pos[0], pos[1] + 1] > img_edge_mag_total_masked[pos]):
                    edge_maps[pos[0], pos[1], i] = 0
                else:
                    img_edge_mag_copy[pos[0], pos[1] - 1] = 0
                    img_edge_mag_copy[pos[0], pos[1] + 1] = 0
        if i == 1 or i == 5:
            x, y = (toLookAt > 0).nonzero()
            indlist = list(zip(x, y))
            for pos in indlist:
                if pos[0] >= img_edge_mag.shape[0] - 1 or pos[1] >= img_edge_mag.shape[1] - 1:
                    break
                elif (img_edge_mag_total_masked[pos[0] - 1, pos[1] - 1] > img_edge_mag_total_masked[pos]) + (img_edge_mag_total_masked[pos[0] + 1, pos[1] + 1] > img_edge_mag_total_masked[pos]):
                    edge_maps[pos[0], pos[1], i] = 0
                else:
                    img_edge_mag_copy[pos[0] - 1, pos[1] - 1] = 0
                    img_edge_mag_copy[pos[0] + 1, pos[1] + 1] = 0

        if i == 2 or i == 6:
            x, y = (toLookAt > 0).nonzero()
            indlist = list(zip(x, y))
            for pos in indlist:
                if pos[0] >= img_edge_mag.shape[0] - 1 or pos[1] >= img_edge_mag.shape[1] - 1:
                    break
                elif (img_edge_mag_total_masked[pos[0] - 1, pos[1]] > img_edge_mag_total_masked[pos]) + (img_edge_mag_total_masked[pos[0] + 1, pos[1]] > img_edge_mag_total_masked[pos]):
                    edge_maps[pos[0], pos[1], i] = 0
                else:
                    img_edge_mag_copy[pos[0] - 1, pos[1]] = 0
                    img_edge_mag_copy[pos[0] + 1, pos[1]] = 0
        if i == 3 or i == 7:
            x, y = (toLookAt > 0).nonzero()
            indlist = list(zip(x, y))
            for pos in indlist:
                if pos[0] >= img_edge_mag.shape[0] - 1 or pos[1] >= img_edge_mag.shape[1] - 1:
                    break
                elif (img_edge_mag_total_masked[pos[0] - 1, pos[1] + 1] > img_edge_mag_total_masked[pos]) + (img_edge_mag_total_masked[pos[0] + 1, pos[1] - 1] > img_edge_mag_total_masked[pos]):
                    edge_maps[pos[0], pos[1], i] = 0
                else:
                    img_edge_mag_copy[pos[0] - 1, pos[1] + 1] = 0
                    img_edge_mag_copy[pos[0] + 1, pos[1] - 1] = 0

        non_max_sup = non_max_sup + edge_maps[:, :, i]

    return non_max_sup


# show the result
img_non_max_sup = edge_non_max_suppression(img_edge_mag, edge_maps)
plt.imshow(np.concatenate((img, img_edge_mag, img_non_max_sup), axis=1))
plt.title('Original image, magnitude edge, and max suppresion')
plt.show()


# 2.5
# Canny edge detection (BONUS)
def canny_edge(image, sigma=2):
    # implementation of canny edge detector
    # INPUTS
    # @image      : 2d image
    # @sigma      : sigma parameter of gaussian
    # OUTPUTS
    # @canny_img  : 2d image of size same as image, with the result of the canny edge detection

    ### your code should go here ###
    gaux_1dfilt = ex1.gauss1d(sigma,np.size(dx))
    gauy_1dfilt = ex1.gauss1d(sigma,np.size(dy))
    gradx = ex1.myconv2(dx,gaux_1dfilt)                             # Gdx
    grady = ex1.myconv2(gauy_1dfilt,dy)                             # Gdy

    g_mag, g_dir = create_edge_magn_image(image, gradx, grady)  # gradient magnitude, direction
    ed_map = make_edge_map(image, gradx, grady)                 # make_edge_map
    m_sup = edge_non_max_suppression(g_mag, ed_map)             # non_maxima_suprema

    hg_tsh = 0.8*(np.amax(g_mag)-np.amin(g_mag))               # strong edges > high threshold = 20
    lw_tsh = 0.5*(np.amax(g_mag)-np.amin(g_mag))                # soft edges   < low threshold  = 10

    canny_img = np.zeros((image.shape))                         # output declaration

    e_map = np.swapaxes(ed_map,1,2)                             # swap axis (255,255,8) -> (255,8,255)
    maps  = np.swapaxes(e_map,0,1)                              # swap axis (255,8,255) -> (8,255,255)

    size_row = np.size(g_mag,0)
    size_clm = np.size(g_mag,1)
    max_val  = np.amax(g_mag)
    i = 0

    for i_map in maps:
        mask     = np.column_stack((np.where(i_map[::]>lw_tsh)))
        for coordinates in mask:
            y = coordinates[0]
            x = coordinates[1]

            if (i_map[y,x]>=hg_tsh):
                canny_img[y,x]=m_sup[y,x]
                
            elif (y < size_row-1 and x < size_clm-1):
                if (i==0 or i== 4):
                    if (i_map[y+1,x]>=hg_tsh or i_map[y-1,x]>=hg_tsh):
                        canny_img[y,x]=m_sup[y,x]

                elif (i==1 or i== 5):
                    if (i_map[y+1,x-1]>=hg_tsh or i_map[y-1,x+1]>=hg_tsh):
                        canny_img[y,x]=m_sup[y,x]

                elif (i==2 or i== 6):
                    if (i_map[y,x-1]>=hg_tsh or i_map[y,x+1]>=hg_tsh):
                        canny_img[y,x]=m_sup[y,x]

                else:
                    if (i_map[y+1,x+1]>=hg_tsh or i_map[y-1,x-1]>=hg_tsh):
                        canny_img[y,x]=m_sup[y,x]

        i+=1

    return canny_img

canny_img = canny_edge(img)
plt.imshow(np.concatenate((img, canny_img), axis=1))
plt.title('Original image and canny edge detector')
plt.show()
