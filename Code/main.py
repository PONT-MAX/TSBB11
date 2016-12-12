#Import libraries
from __future__ import absolute_import, print_function
import datetime

#Import local files
import init_v
import object
import help_functions
import multiprocessing
import sub_clustering
import visualization

#Set constants
CORES = multiprocessing.cpu_count()
MIN_BLOB_SIZE = 300
PERCENTAGE_OF_ARC1 = 0.01
PERCENTAGE_OF_ARC2 = 0.005
NUMBER_OF_FEATURES = 9
#Needs to be changed if additional features are added
QUANTIZE_ANGLES = True
#Makes angles straighter
#########################################

print("Number of System Cores: ", CORES, "\n")

# Load map names
map_source_directory = init_v.init_map_directory()

date = datetime.datetime.now()
#feature_data_filename = './numpy_arrays/feature_data_all_threads_final_' + \
#str(date.month) + str(date.day) + str(date.hour) + str(date.minute) + '.npy'
feature_data_filename = './numpy_arrays/feature_data_all_threads_final.npy'

# Extract features
print("Loading feature data... ")
feature_data = object.getFeatures(map_source_directory, CORES,
	NUMBER_OF_FEATURES, MIN_BLOB_SIZE, PERCENTAGE_OF_ARC1, PERCENTAGE_OF_ARC2, 
	QUANTIZE_ANGLES, new_markers=True, filename=feature_data_filename,
	load_features=False, save_features=True)
print("Done!")
print("Clustering data... ")
data = sub_clustering.kMeansSubCluster(feature_data, normalize=True)
print("Done!")
print("Coloring data... ")
visualization.colorCluster(data, map_source_directory, CORES, save=True, 
	im_name='New_data2', scale=0.125)
sub_clustering.exportCluster2PNG(data, map_source_directory, CORES)
print("Done!")

print("Number of buildings: ", max(feature_data.size))

print("Print monopoly houses...")
help_functions.printMonopolyHouses(map_source_directory, MIN_BLOB_SIZE, 
	PERCENTAGE_OF_ARC1,	PERCENTAGE_OF_ARC2, QUANTIZE_ANGLES)
print("Done!")

