import numpy as np
from PIL import Image
import cv2
import threading
import time
from sklearn.cluster import KMeans

def kMeansSubCluster(data, normalize=False):

    if normalize:
        for i in range(min(data.shape)-2):
            data[:, i] = np.copy(data[:, i] / np.amax(data[:, i]))

    nbr_sub_cluster = 3
    first_cluster_size = 3

    labels = KMeans(n_clusters=first_cluster_size, random_state=10).fit(data[:, 0:1])
    data = np.copy(np.hstack((data, np.transpose(np.array([labels.labels_])))))

    data[:, -1] = data[:, -1] * nbr_sub_cluster

    histo = np.bincount(data[:, -1].astype(int))
    print(histo)

    for current_label in range(0, nbr_sub_cluster * first_cluster_size, nbr_sub_cluster):
        # Find cluster to subcluster
        index_labels = np.where(data[:, -1] != current_label)
        index_rest = np.where(data[:, -1] == current_label)

        # Separate Currrent sub cluster data & the rest of the data
        cluster_current = np.delete(data, index_labels, 0)
        data = np.delete(data, index_rest, 0)

        labels = KMeans(n_clusters=nbr_sub_cluster, random_state=10).fit(cluster_current[:, 1:7])
        cluster_current[:, -1] = labels.labels_ + current_label

        data = np.vstack((data, cluster_current))

        histo = np.bincount(data[:, -1].astype(int))
        print(histo)


    return data




def labeler(TREAHD_ID, cluster_data, map_c, markers, CORES, cls_mask):
    nbr_feat_max = max(cluster_data.shape) - 1
    for feat in range(TREAHD_ID, nbr_feat_max, CORES):
        if cluster_data[feat, -2] == map_c: # Find correct map
            if not feat % (nbr_feat_max / 10):
                print("Map: ", map_c, " || Thread: ", TREAHD_ID, " || done: ", ((feat * 100) / nbr_feat_max), "%")

            # Add one to labels to destinct from BG
            marker_id = cluster_data[feat, -3]
            label = cluster_data[feat, -1] + 1
            index_pos = np.where(markers == marker_id)
            cls_mask[index_pos] = label
    if TREAHD_ID == 0:
        print("map: ", map_c, " is done!\n\n")


def exportCluster2PNG(cluster_data, map_source_directory, CORES):

    TIME = time.time()

    for map_c in range(len(map_source_directory)):
        print("Start coloring map ", map_c)

        map_name = map_source_directory[map_c]

        name = "./markers/markers_" + str(map_c) + ".png"
        markers = np.asarray(Image.open(name))
        cls_mask = np.empty([max(markers.shape), max(markers.shape), 1], dtype=int) * 0

        threads = []
        for i in range(0, CORES):
            t = threading.Thread(target=labeler, args=(i, cluster_data, map_c, markers, CORES, cls_mask))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        print("Saving Class map: ", map_c)
        print("Shape FD = ", cls_mask.shape)
        Image.fromarray(cls_mask.astype('uint8')[:, :, 0]).save("../Data/cls/" + map_name + "ccls.png")

    print("Time:")
    print(time.time() - TIME)

