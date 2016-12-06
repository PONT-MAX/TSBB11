import numpy as np
from PIL import Image

def init_map_directory():
    map_source_directory = []
    "Collects all names to the maps in a numpy char array"
    map_source_directory.append('0152945e_582034n_20160905T073402Z_')
    map_source_directory.append('0152947e_582246n_20160905T073405Z_')
    map_source_directory.append('0152949e_582459n_20160905T073406Z_')
    map_source_directory.append('0152951e_582711n_20160905T073406Z_')
    map_source_directory.append('0153357e_582033n_20160905T073402Z_')
    map_source_directory.append('0153359e_582245n_20160905T073406Z_')
    map_source_directory.append('0153401e_582458n_20160905T073406Z_')
    map_source_directory.append('0153403e_582710n_20160905T073405Z_')
    map_source_directory.append('0153405e_582922n_20160905T073406Z_')
    map_source_directory.append('0153809e_582032n_20160905T073406Z_')
    map_source_directory.append('0153811e_582244n_20160905T073406Z_')
    map_source_directory.append('0153813e_582456n_20160905T073406Z_')
    map_source_directory.append('0153816e_582709n_20160905T073404Z_')
    map_source_directory.append('0153818e_582921n_20160905T073406Z_')
    map_source_directory.append('0154220e_582030n_20160905T073406Z_')
    map_source_directory.append('0154223e_582243n_20160905T073406Z_')
    map_source_directory.append('0154226e_582455n_20160905T073406Z_')
    return map_source_directory;

def get_map_array(type,map_name,print_info):
    "Returns ortho map as numphy array, call type w/: dtm,dsm,dhm,cls,ortho"

    if(type == 'ortho'):
        type2 = 'tex'
    elif(type == 'cls'):
        type2 = type
        type = 'auxfiles'
    else:
        type2 = type

    map_array = np.array(Image.open('../Data/' + type + '/' + map_name + type2 + '.tif'))

# if(print_info):
#        print('')
#        print(map_array.shape)
#        print(map_array.dtype)

    return map_array
