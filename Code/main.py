#Import libraries

from __future__ import absolute_import, print_function
import cv2
import numpy as np
import random
import visulation_export_image as vei
from subprocess import call
from PIL import Image
from PIL import ImageOps
import time
from matplotlib import pyplot as plt
import seaborn as sns
import hdbscan
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from itertools import chain
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

#Import local files
import init_v
import object
import extract_buildings
import help_functions
import multiprocessing
import cluster

#Set constants
CORES = multiprocessing.cpu_count()
print("Number of System Cores: ", CORES, "\n")
NUMBER_OF_FEATURES = 13

#########################################
# Load map names
map_source_directory = init_v.init_map_directory()

# Extract features
print("Loading feature data... ")
feature_data = object.getFeatures(map_source_directory,
    CORES,new_markers=False,filename='./numpy_arrays/feature_data_all_threads_final.npy',load=True)


rows = max(feature_data.shape)
cols = min(feature_data.shape)



# Plot area / area
# object.scatterPlot(feature_data, 0, 1, name_x='mead heaght',name_y='water')

first_cluster = np.empty([rows, 4])

first_cluster[:, 0] = np.copy(feature_data[:, 0])
first_cluster[:, 1] = np.copy(np.multiply(feature_data[:, 4], feature_data[:, 5]))
first_cluster[:, 2] = np.copy(feature_data[:, cols-2])
first_cluster[:, 3] = np.copy(feature_data[:, cols-1])
# object.scatterPlot(first_cluster, 0, 1,figure=1, name_x='real area',name_y='conture area')

X = np.copy(first_cluster)



# Initialize the clusterer with n_clusters value and a random generator
# seed of 10 for reproducibility.
cluster_labels = KMeans(n_clusters=2, random_state=10).fit(X[:, 0:2])
cluster_data_2 = np.copy(np.hstack((X, np.transpose(np.array([cluster_labels.labels_])))))
#object.colorCluster(cluster_data_2, map_source_directory, CORES, scale=0.5, save=True, im_name="K_means_2")


histo = np.bincount(cluster_data_2[:, -1].astype(int))
print("c_D_2", histo)

cluster_labels = KMeans(n_clusters=3, random_state=10).fit(X[:, 0:2])
cluster_data_3 = np.copy(np.hstack((X, np.transpose(np.array([cluster_labels.labels_])))))

histo = np.bincount(cluster_data_3[:, -1].astype(int))
print("c_D_3", histo)
#object.colorCluster(cluster_data_3, map_source_directory, CORES, scale=0.125, save=True, im_name="K_means_3_new_color")

label_holder = np.empty([rows, 2])
label_holder[:, 0] = cluster_data_2[:, -1]
label_holder[:, 1] = cluster_data_3[:, -1]


sec_cluster = np.empty([rows, 7])
sec_cluster[:, 0] = np.copy(feature_data[:, 1]/np.amax(feature_data[:, 1]))
sec_cluster[:, 1] = np.copy(feature_data[:, 2]/np.amax(feature_data[:, 2]))
sec_cluster[:, 2] = np.copy(feature_data[:, 4]/np.amax(feature_data[:, 4]))
sec_cluster[:, 3] = np.copy(feature_data[:, 5]/np.amax(feature_data[:, 5]))
sec_cluster[:, 4] = np.copy(feature_data[:, cols-2])
sec_cluster[:, 5] = np.copy(feature_data[:, cols-1])


for nbr_sub_cluster in range(3, 4):  # Number of subclasses to each class
    for first_cluster_size in range(3, 4):  # Number of classes before
        sec_cluster[:, 6] = label_holder[:, first_cluster_size-2]
        X = np.copy(sec_cluster)
        X[:, -1] = X[:, -1] * nbr_sub_cluster
        histo = np.bincount(X[:, -1].astype(int))
        print("Histo X BF000 last FL", histo)
        for current_label in range(0, nbr_sub_cluster*first_cluster_size, nbr_sub_cluster):

            index_labels = np.where(X[:, -1] != current_label)
            index_rest = np.where(X[:, -1] == current_label)
            # Separate Currrent sub cluster data & the rest of the data
            cluster_current = np.delete(X, index_labels, 0)
            X = np.delete(X, index_rest, 0)



            cluster_labels = KMeans(n_clusters=nbr_sub_cluster, random_state=10).fit(cluster_current[:, 0:4])
            cluster_current[:, -1] = cluster_labels.labels_ + current_label

            X = np.vstack((X, cluster_current))

            histo = np.bincount(X[:, -1].astype(int))
            print("Histo X BF111 last FL", histo)
            print("Xsample: ", X[0::700, -1])
        #histo = np.bincount(X[:, -1].astype(int))
        #print(histo)
        #object.colorCluster(X, map_source_directory, CORES, scale=0.5, save=True,
        #                    im_name="K_means_test_3_3_D20_K55")


nbr_sub_cluster = 2
first_cluster_size = 9
X = np.copy(np.hstack((feature_data[:, 6:11], X[:, -3::])))
X[:, -1] = X[:, -1] * 2


for current_label in range(0, nbr_sub_cluster * first_cluster_size, nbr_sub_cluster):
    print("Current_label: ", current_label)
    index_labels = np.where(X[:, -1] != current_label)
    index_rest = np.where(X[:, -1] == current_label)
    # Separate Currrent sub cluster data & the rest of the data
    cluster_current = np.delete(X, index_labels, 0)
    X = np.delete(X, index_rest, 0)

    cluster_labels = KMeans(n_clusters=nbr_sub_cluster, random_state=10).fit(cluster_current[:, 0:-3])
    cluster_current[:, -1] = cluster_labels.labels_ + current_label


    X = np.vstack((X, cluster_current))

object.colorCluster(X, map_source_directory, CORES, scale=0.5, save=True, im_name="K_means_3_3_2_D20_K55")
histo = np.bincount(X[:, -1].astype(int))
print("Histo X BF last FL", histo)


"""
print("\n")

Jobba pa 3 eller 4

cluster_data1 = cluster.cluster_data(feature_data,
    save_cluster_data=True,save_filename='cd_sub_pre_meter_new3.npy',sub_clustering=False)
object.colorCluster(cluster_data1, map_source_directory,CORES,scale=0.5,save=True,sub_c=False,im_name="1_sub_meter_new3")

last_feat = min(cluster_data1.shape) - 1
a = cluster_data1[:,last_feat]
counts = np.bincount(a.astype(int) )
print(counts)
print(feature_data.shape)




print("\n")
"""
