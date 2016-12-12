from __future__ import absolute_import, print_function
import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import time
import threading
import sys

####### NOTE #######

#Legacy code - all functions are unstable predev versions.

####### NOTE #######

#Prints the result from HDBSCAN.
def printHdbscanResult(hd_cluster, feature_data, stat=True, print_all=False, visulize_clustering=False):
    histo = np.bincount(hd_cluster.labels_ + 1)
    nbr_of_datapoints = max(feature_data.shape)
    not_classified = histo[0]
    nbr_of_classes = max(histo.shape)
    proc = not_classified * 100 / nbr_of_datapoints
    if nbr_of_classes <= 1:
        return 100, 100

    largest_class = (max(histo[1::]) * 100 / nbr_of_datapoints)
    if largest_class > 75 and nbr_of_classes > 5:
        return 100, 100
    elif largest_class > 75:
        return 100, 100

    if stat:
        print("number of classes(including not a class): ", nbr_of_classes)
        print("Of ", nbr_of_datapoints, " datapoints, ", not_classified, " was not classified.  ", proc, "%")

    if print_all:
        sum_all = 0
        print("class    members     %")
        for x in range(1, nbr_of_classes):
            proc = histo[x] * 100 / nbr_of_datapoints
            sum_all += histo[x]
            print(x, "     ", histo[x], "       ", proc)
        sum_all_p = sum_all * 100 / nbr_of_datapoints
        print("sum    ", sum_all, "      ", sum_all_p)

    if visulize_clustering:
        color_palette = sns.color_palette('Paired', nbr_of_classes + 1)
        cluster_colors = [color_palette[x] if x >= 0
                          else (0.5, 0.5, 0.5)
                          for x in hd_cluster.labels_]
        cluster_member_colors = [sns.desaturate(x, p) for x, p in
                                 zip(cluster_colors, hd_cluster.probabilities_)]
        projection = TSNE().fit_transform(feature_data)
        plt.scatter(*projection.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)
        plt.show()

    return proc, nbr_of_classes

#Returns best parameters for HDBSCAN.
def findOptimalHdbParameters(cluster_data, save=False, mcs_delta=1, ms_delta=1,
                             cls_low=4, cls_high=5, proc=50):
    if max(cluster_data.shape) < 2500:
        mcs_start = 4
        mcs_end = 100
    elif max(cluster_data.shape) < 7500:
        mcs_start = 60
        mcs_end = 140
    else:
        mcs_start = 90
        mcs_end = 140

    if save is False:
        best_parameters = np.load("./HdbP/HdbParameters_12461.npy")
        return best_parameters.item(0), best_parameters.item(1)

    print("Sub Clustering")
    mcs_end = min(mcs_end, int(max(cluster_data.shape) / 2))

    best_mcs, best_ms, best_P = findClusterParameters(cluster_data, mcs_start, mcs_end, mcs_delta,
                                                      ms_delta, cls_low, cls_high, proc)
    best_parameters = np.array([best_mcs, best_ms, best_P])
    print(best_parameters)
    if save:
        name = "./HdbP/HdbParameters_" + str(max(cluster_data.shape)) + str(int(time.time())) + ".npy"
        np.save(name, best_parameters)

    print(best_parameters)
    return best_parameters.item(0), best_parameters.item(1)

#Returns best parameters for clusterings.
def findClusterParameters(cluster_data, mcs_start, mcs_end, mcs_delta, ms_delta, nbr_cls_low,
                          nbr_cls_high, proc_high):
    best_mcs = 0
    best_ms = 0
    best_P = 90

    for mcs in range(mcs_start, mcs_end, mcs_delta):
        good_param_found = False
        if not mcs % 20:
            print("MCS: ", mcs)
            ms_delta += 1
        for ms in range(1, max(mcs, 80), ms_delta):
            # print(" Starting HDBSCAN, data size:", cluster_data.shape)
            hd_cluster = hdbscan.HDBSCAN(algorithm='best', metric='euclidean', min_cluster_size=mcs, min_samples=ms,
                                         alpha=1.0)
            hd_cluster.fit(cluster_data)

            # Lables
            proc, nbr_cls = printHdbscanResult(hd_cluster, cluster_data, stat=False)

            if nbr_cls >= nbr_cls_low and nbr_cls <= nbr_cls_high and proc < proc_high:
                # print("MCS: ", mcs, " & MS: ", ms, "Gives best %: ", proc, " w/ ", nbr_cls, " classes")
                if proc <= best_P:
                    best_mcs = mcs
                    best_ms = ms
                    best_P = proc
                    print("MCS: ", best_mcs, " & MS: ", best_ms, "Gives best %: ", proc, " w/ ", nbr_cls, " classes")
                    printHdbscanResult(hd_cluster, cluster_data, print_all=True)
                    # if best_P < 3:
                    # print("Breaking search: Proc < 5%")
                    # good_param_found = True
                    # break
        if good_param_found:
            break

    return int(best_mcs), int(best_ms), int(best_P)

#Prints best HDBSCAN result
def printOptimalHdb(cluster_data, mcs, ms, stat=True, print_all_statistic=True, visulize_clustering=False):
    print("optimal: ", mcs, ms)

    hd_cluster = hdbscan.HDBSCAN(algorithm='best', metric='euclidean', min_cluster_size=mcs, min_samples=ms,
                                 alpha=1.0)
    hd_cluster.fit(cluster_data)

    # Lables
    proc, nbr_cls = printHdbscanResult(hd_cluster, cluster_data,
                                       stat, print_all_statistic, visulize_clustering)
    return hd_cluster
