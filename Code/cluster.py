import object
import numpy as np


def cluster_data(cluster_data, save_cluster_data=False, save_filename='',sub_clustering=True):

    if not save_cluster_data:
        return np.load(save_filename)

    print("Cluster Data Shape Before: ", cluster_data.shape)
    last_feat = min(cluster_data.shape) - 1

    if sub_clustering:
        print("Mode: Sub Clustering")
        nbr_cls_high = 7
        nbr_cls_low = 5
        # Use Area & avg height / Vol
        cluster_first = np.copy(cluster_data[:, 0:last_feat - 1])
    else:
        print("Mode: Full Clustering")
        nbr_cls_high = 15
        nbr_cls_low = 8
        # Use all features
        cluster_first = np.copy(cluster_data[:, 0:last_feat - 1])



    # Find Optimal HDBSCAN parameters
    print("Find Optimal Parameters fist It")

    print(cluster_data.shape)
    print(cluster_first.shape)

    mcs, ms = object.findOptimalHdbParameters(cluster_first, save=True, cls_high=nbr_cls_high,
                                              cls_low=nbr_cls_low,mcs_delta=1,ms_delta=1)
    print(mcs, ms)

    hd_cluster = object.printOptimalHdb(cluster_first, mcs, ms, print_all_statistic=True)

    # Add labels to the Features
    cluster_data = np.hstack((cluster_data, np.transpose(np.array([hd_cluster.labels_]))))
    last_feat = min(cluster_data.shape) - 1

    print(cluster_data.shape)

    # Make unclassified data label 0
    cluster_data[:, last_feat] = cluster_data[:, last_feat] + 1

    # New code
    """
    a = cluster_data[:,last_feat]
    counts = np.bincount(a.astype(int) )
    print(counts)
    house_index = np.argmax(counts)
    not_house = np.where(cluster_data[:, last_feat] != house_index)
    house = np.where(cluster_data[:, last_feat] == house_index)
    cluster_data[not_house, last_feat] = 0
    cluster_data[house, last_feat] = 1
    a = cluster_data[:, last_feat]
    counts = np.bincount(a.astype(int))
    print(counts)
    """
    # ------

    for_range = range(0, 1)

    if sub_clustering:
        for_range = list(range(1, int(max(cluster_data[:, last_feat]) + 1))) + list(for_range)

    print("Find Optimal Parameters sub Clustering")
    for label in for_range:
        print("For label: ", label)
        # Find where current labels datapoints are in cluster data
        index_labels = np.where(cluster_data[:, last_feat] != label)
        index_rest = np.where(cluster_data[:, last_feat] == label)

        # Separate Currrent sub cluster data & the rest of the data
        cluster_current = np.delete(cluster_data, index_labels, 0)
        cluster_data = np.delete(cluster_data, index_rest, 0)

        # Find optimal Parameters
        print("For label: ", label, "  || cluster size:  ", cluster_current.shape)
        mcs, ms = object.findOptimalHdbParameters(cluster_current[:, 0:last_feat-2], save=True,
                                                  cls_low=3,cls_high=4, mcs_delta=1,ms_delta=1)
        if mcs == 0:
            # Put back sub cluster and whole cluster
            cluster_current[:, last_feat] = 0
        else:
            hd_cluster = object.printOptimalHdb(cluster_current[:, 0:last_feat-2], mcs, ms)
            cluster_current[:, last_feat] = hd_cluster.labels_ + 1
            cluster_current[:, last_feat] += 100 * (label + 1)

        cluster_data = np.vstack((cluster_data, cluster_current))

    if sub_clustering:
        index_pos = np.where(cluster_data[:, last_feat] >= 0)
        cluster_data[index_pos, last_feat] -= 100
    else:
        histo = np.bincount(cluster_data[:, last_feat].astype(int) - 1)
        index_pos = np.where(cluster_data[:, last_feat] > 15)
        cluster_data[index_pos, last_feat] -= (99 - np.where(histo == 0)[0][0])
        cluster_data[:, last_feat] = cluster_data[:, last_feat] - 1

    if save_cluster_data:
        print("Saving Clustered Data: Succes!\n\n\n\n")
        np.save(save_filename, cluster_data)

    return cluster_data
