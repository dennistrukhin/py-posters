from operator import itemgetter

from sklearn.cluster import KMeans
import utils
import cv2
from PIL import Image
from math import pow, sqrt

PIXEL_FACTOR = 1
USE_AUTO_PALETTE = True
SUBSTITUTE_PALETTE = True
# SUBSTITUTE_PALETTE = False

if USE_AUTO_PALETTE:
    # image = cv2.imread("./input-3.jpg")
    image = cv2.imread("./input/0404-04-1.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (300, 300))

    image = image.reshape((image.shape[0] * image.shape[1], 3))

    clt = KMeans(n_clusters=5)
    clt.fit(image)

    print(clt.cluster_centers_)
    centroids = sorted(clt.cluster_centers_, key=itemgetter(0))
else:
    centroids = utils.get_palette([10, 150, 180, 210, 255])

scheme = utils.scheme_from_hexes([
    # 'BDBAAA', 'B9AA7A', 'BB803B', '976634', '252D1B',
    # '82C5A2', 'D79057', 'F0C582', 'FDEDB8', '1C2536',  # Мичуринка
    # '253239', '4D684F', 'C1AA64', 'EDE6B0', 'AAC5AA',
    # '253239', '3E6352', 'A1C576', 'EDE6B0', 'D04E33',
    # '253239', '3E6352', '15BDC1', 'EDE6B0', 'D04E33',  # Office BEST
    # '15BDC1', '3E6352', '253239', 'EDE6B0', 'D04E33',
    # '45826C', '8AD392', 'F2D068', 'F49A4C', 'E65549',  # Tree
    # '162F38', '427A55', '9EC135', 'D44A26', 'F6E87B',  # Рябина
    # 'D39A66', 'E4D988', 'C59566', 'AF484E', '442237',  # Киса сверху
    # 'CCC9CF', '09335B', '204D81', 'E6E7D7', 'F3AC88',  # Дерево и забор
    # '83A184', 'F0D7A6', 'EF8C2E', 'AB8F79', '6E2E2E',  # веточка в стакане
    # '532943', 'E0DFB6', '842F5C', 'BEC89A', 'D1C3B6',  # Цветы в офисе на подоконнике
    '406C4D', '7B8169', 'BDC1B9', 'D26B45', 'BDBFBF',  # Цветы в офисе на подоконнике
])
print(scheme)

im = Image.open("./input/0404-04-1.jpg")
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
img.save("./output/0404-04-1.jpg", quality=100, dpi=(300, 300))
img.close()
