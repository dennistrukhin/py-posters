import numpy as np
import cv2


def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    num_labels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=num_labels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    # return the histogram
    return hist


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    start_x = 0
    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        end_x = start_x + (percent * 300)
        cv2.rectangle(bar, (int(start_x), 0), (int(end_x), 50),
                      color.astype("uint8").tolist(), -1)
        start_x = end_x

    # return the bar chart
    return bar


def scheme_from_hexes(hexes):
    scheme = []
    for i in range(0, 5):
        scheme.append([
            int(hexes[i][0:2], 16),
            int(hexes[i][2:4], 16),
            int(hexes[i][4:6], 16),
        ])
    return scheme


def get_palette(thresholds):
    palette = []
    for i in range(0, 5):
        palette.append([
            thresholds[i], thresholds[i], thresholds[i],
        ])
    return palette
