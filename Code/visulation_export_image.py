from numpy import *
from PIL import Image

def visulation_export( map_name ):
    "This function exports the tiff maps to png to feed the visulation"
    print("Start EXPORT")
    # Data

    # Ortho
    ortho = array(Image.open('../Data/ortho/' + map_name + 'tex.tif'))
    ortho_out = Image.fromarray(ortho.astype(uint8))
    ortho_out.save('visulation/images/ortho.png')
    print("Export Ortho png")

    # DSM
    dsm = array(Image.open('../Data/dsm/' + map_name + 'dsm.tif'))
    dsm[dsm < 0] = 0
    dsm_n = (dsm/dsm.max())*255.0
    # dsm_out = Image.fromarray(dsm.astype(uint8))
    # dsm_out.save('visulation/images/dsm.png')
    dsm_out_n = Image.fromarray(dsm_n.astype(uint8))
    dsm_out_n.save('visulation/images/dsm_n.png')
    print("Export DSM png")

    # DHM
    dhm = array(Image.open('../Data/dhm/' + map_name + 'dhm.tif'))
    dhm[dhm < 0] = 0
    dhm_n = (dhm/dhm.max())*255.0
    #dhm_out = Image.fromarray(dhm.astype(uint8))
    # dhm_out.save('visulation/images/dhm.png')
    dhm_out_n = Image.fromarray(dhm_n.astype(uint8))
    dhm_out_n.save('visulation/images/dhm_n.png')
    print("Export DHM png")

    # DTM
    dtm = array(Image.open('../Data/dtm/' + map_name + 'dtm.tif'))
    dtm[dtm < 0] = 0
    dtm_n = (dtm/dtm.max())*255.0
    #dtm_out = Image.fromarray(dtm.astype(uint8))
    # dtm_out.save('visulation/images/dtm.png')
    dtm_out_n = Image.fromarray(dtm_n.astype(uint8))
    dtm_out_n.save('visulation/images/dtm_n.png')
    print("Export DTM png")

    # AUX
    aux = array(Image.open('../Data/auxfiles/' + map_name + 'cls.tif'))
    aux_out = Image.fromarray(aux.astype(uint8),"L")
    aux_out.getpalette()
    print(aux_out.mode)
    aux_out.save('visulation/images/aux.png')
    print("Export AUX png")
    return

def visulation_export_result(output_name, output_array):
    output_image = Image.fromarray(output_array.astype(uint8),"L")
    output_image.save('visulation/images/' + output_name + '.png')
    return



