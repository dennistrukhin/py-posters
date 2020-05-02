from operator import itemgetter

from sklearn.cluster import KMeans
import utils
import cv2
from PIL import Image
from math import pow, sqrt
from data import posters, dimensions
import numpy as np

KEY = "0019"
POSTER = posters[KEY]

PIXEL_FACTOR = 5
SUBSTITUTE_PALETTE = POSTER['substitutePalette']

if POSTER['substituteSource']:
    image = cv2.imread("./input/003.jpg")
else:
    image = cv2.imread("./input/" + KEY + ".jpg")

# if image.shape[0] != image.shape[1]:
#     raise Exception("Input image should be square")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
if POSTER['usePredefinedCentroids']:
    centroids = POSTER['centroids']
else:
    paletteSourceImage = cv2.resize(image, (300, 300))
    paletteSourceImage = paletteSourceImage.reshape((paletteSourceImage.shape[0] * paletteSourceImage.shape[1], 3))

    clt = KMeans(n_clusters=5)
    clt.fit(paletteSourceImage)

    centroids = sorted(clt.cluster_centers_, key=itemgetter(0))

scheme = utils.scheme_from_hexes(POSTER['scheme'])

# for dim in dimensions:

dim = dimensions[5]

border_size = 0  # floor(dim['border'] * dim['points']) * (1 if withBorder else 0)
w = dim['points']
h = dim['points']
u = dim['upscale']
im = cv2.resize(image, (w, h))
img = Image.new(mode='RGB', size=(w * u, h * u), color=(0, 0, 0))
pix_new = img.load()

for y in range(0, h):
    if y % 100 == 0:
        print(dim['name'] + ': ' + str(y))
    for x in range(0, w):
        min_dist = 1000000
        c_i = 0
        (r, g, b) = im[y, x]
        for c in range(0, 5):
            dist = sqrt(pow(centroids[c][0] - r, 2) + pow(centroids[c][1] - g, 2) + pow(centroids[c][2] - b, 2))
            if dist < min_dist:
                c_i = c
                min_dist = dist
        for dux in range(0, u):
            for duy in range(0, u):
                pix_new[x*u + border_size + dux, y*u + border_size + duy] = \
                    (
                        scheme[c_i][0] if SUBSTITUTE_PALETTE else int(centroids[c_i][0]),
                        scheme[c_i][1] if SUBSTITUTE_PALETTE else int(centroids[c_i][1]),
                        scheme[c_i][2] if SUBSTITUTE_PALETTE else int(centroids[c_i][2])
                    )

img.save("./output/[" + KEY + '] ' + POSTER['name'] + ' - ' + dim['name'] + '.jpg',
         quality=dim['quality'], dpi=(dim['resolution'], dim['resolution']))
img.close()

