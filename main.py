from operator import itemgetter

from sklearn.cluster import KMeans
import utils
import cv2
from PIL import Image
from math import pow, sqrt
from data import posters, dimensions
from math import floor

KEY = "012"
POSTER = posters[KEY]

PIXEL_FACTOR = 1
SUBSTITUTE_PALETTE = POSTER['substitutePalette']

if POSTER['substituteSource']:
    image = cv2.imread("./input/003.jpg")
else:
    image = cv2.imread("./input/" + KEY + ".jpg")

if image.shape[0] != image.shape[1]:
    raise Exception("Input image should be square")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
if POSTER['usePredefinedCentroids']:
    centroids = POSTER['centroids']
else:
    exit()
    paletteSourceImage = cv2.resize(image, (300, 300))
    paletteSourceImage = paletteSourceImage.reshape((paletteSourceImage.shape[0] * paletteSourceImage.shape[1], 3))

    clt = KMeans(n_clusters=5)
    clt.fit(paletteSourceImage)

    centroids = sorted(clt.cluster_centers_, key=itemgetter(0))

scheme = utils.scheme_from_hexes(POSTER['scheme'])

# for withBorder in [False, True]:
for withBorder in [False]:
    for dim in dimensions:
        border_size = floor(dim['border'] * dim['points']) * (1 if withBorder else 0)
        s = dim['points']
        im = cv2.resize(image, (s - 2 * border_size, s - 2 * border_size))
        img = Image.new(mode='RGB', size=(s, s), color=(0, 0, 0))
        pix_new = img.load()

        for y in range(0, s - 2 * border_size):
            for x in range(0, s - 2 * border_size):
                min_dist = 1000000
                c_i = 0
                (r, g, b) = im[y, x]
                for c in range(0, 5):
                    dist = sqrt(pow(centroids[c][0] - r, 2) + pow(centroids[c][1] - g, 2) + pow(centroids[c][2] - b, 2))
                    if dist < min_dist:
                        c_i = c
                        min_dist = dist
                pix_new[x + border_size, y + border_size] = \
                    (
                        scheme[c_i][0] if SUBSTITUTE_PALETTE else int(centroids[c_i][0]),
                        scheme[c_i][1] if SUBSTITUTE_PALETTE else int(centroids[c_i][1]),
                        scheme[c_i][2] if SUBSTITUTE_PALETTE else int(centroids[c_i][2])
                    )

        img.save("./output/" + KEY + '-' + dim['name'] + '-' + ('border' if withBorder else '') + "-test.jpg",
                 quality=100, dpi=(300, 300))
        img.close()
