import sys
import glob
import os
import numpy as np
import pdb
import matplotlib.pyplot as plt
from scipy import misc
import random

def distances(in_map):
    # get distances from top left to bottom right
    for x in range(1, len(in_map[:,0]) - 1):
        for y in range(1, len(in_map[0,:]) - 1):
            b = in_map[x + 1, y + 1] + 2
            l = in_map[x, y - 1] + 1
            t = in_map[x - 1, y - 1] + 2
            t = in_map[x - 1, y] + 1
            c = in_map[x, y]
            in_map[x,y] = np.amin(np.array([c, t, l, b, t]))
    return in_map

# load shapes
shapes = glob.glob(os.path.join('shapes', '*.png'))
for i, shape in enumerate(shapes):
    # load the edge map
    edge_map = plt.imread(shapes[i])

    # caclulate distance map
    # set edges to '0' and background to 'inf'
    map_inf = edge_map.copy()
    map_inf[map_inf == 0] = np.inf
    map_inf[map_inf == 1] = 0
    
    # append 2 rows and 2 columns
    app_column = np.ones((len(map_inf[:,0]), 2)) * np.inf   
    app_row = np.ones((2, len(map_inf[0,:]) + 2)) * np.inf
    in_map = np.c_[map_inf, app_column]
    in_map = np.r_[in_map, app_row]

    in_map = distances(in_map)
    in_map = distances(np.rot90(np.rot90(in_map)))       
    in_map = np.rot90(np.rot90(in_map))  

    # distance_map: array_like, same size as edge_map
    distance_map = in_map[1:len(in_map[:,0])-1, 1:len(in_map[0,:])-1]

    # the top row of the plots should be the edge maps, and on the bottom the corresponding distance maps
    k, l = i + 1, len(shapes) + i + 1
    plt.subplot(2, len(shapes), k)
    plt.imshow(edge_map, cmap='gray')
    plt.subplot(2, len(shapes), l)
    plt.imshow(distance_map, cmap='gray')

plt.show()
