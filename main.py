from operator import itemgetter

from sklearn.cluster import KMeans
import utils
import cv2
from PIL import Image
from math import pow, sqrt
from data import posters

KEY = "005"
POSTER = posters[KEY]

PIXEL_FACTOR = 1
SUBSTITUTE_PALETTE = POSTER['substitutePalette']

# image = cv2.imread("./input-3.jpg")
if POSTER['substituteSource']:
    image = cv2.imread("./input/003.jpg")
else:
    image = cv2.imread("./input/" + KEY + ".jpg")


image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (300, 300))

image = image.reshape((image.shape[0] * image.shape[1], 3))

clt = KMeans(n_clusters=5)
clt.fit(image)

centroids = sorted(clt.cluster_centers_, key=itemgetter(0))

scheme = utils.scheme_from_hexes(POSTER['scheme'])

im = Image.open("./input/" + KEY + ".jpg")
(sx, sy) = im.size
img = Image.new(im.mode, (sx * PIXEL_FACTOR, sy * PIXEL_FACTOR))
pix = im.load()
pix_new = img.load()
(w, h) = im.size

for y in range(0, h):
    print(y)
    for x in range(0, w):
        min_dist = 1000000
        c_i = 0
        (r, g, b) = pix[x, y]
        for c in range(0, 5):
            dist = sqrt(pow(centroids[c][0] - r, 2) + pow(centroids[c][1] - g, 2) + pow(centroids[c][2] - b, 2))
            if dist < min_dist:
                c_i = c
                min_dist = dist
        for dx in range(0, PIXEL_FACTOR):
            for dy in range(0, PIXEL_FACTOR):
                pix_new[x * PIXEL_FACTOR + dx, y * PIXEL_FACTOR + dy] = \
                    (
                        scheme[c_i][0] if SUBSTITUTE_PALETTE else int(centroids[c_i][0]),
                        scheme[c_i][1] if SUBSTITUTE_PALETTE else int(centroids[c_i][1]),
                        scheme[c_i][2] if SUBSTITUTE_PALETTE else int(centroids[c_i][2])
                    )

im.close()
img.save("./output/" + KEY + "-test.jpg", quality=100, dpi=(72, 72))
img.close()
