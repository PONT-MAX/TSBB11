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
    aux = visulation_export_cls(aux)
    aux_out = Image.fromarray(aux.astype(uint8))
    aux_out.save('visulation/images/aux.png')
    print("Export AUX png")
    return

def visulation_export_result(output_name, output_array):
    print("Export to png")
    output_image = Image.fromarray(output_array.astype(uint8))
    output_image.save('visulation/images/' + output_name + '.png')
    print("Export done!")
    return

def visulation_export_cls(output_array):
    print("Start exporting class")
    rows,cols = output_array.shape
    aux = empty((rows,cols, 3))

    red_ch = copy(output_array)
    green_ch = copy(output_array)
    blue_ch = copy(output_array)

    # Code each class
    # Class 0: Unknown 0 Orange
    red_ch[red_ch == 0] = 255
    green_ch[green_ch == 0] = 128
    blue_ch[blue_ch == 0] = 0

    # Class 1: Terrain yellow
    red_ch[red_ch == 1] = 255
    green_ch[green_ch == 1] = 200
    blue_ch[blue_ch == 1] = 0

    # Class 2: Man made Objects RED
    red_ch[red_ch == 2] = 255
    green_ch[green_ch == 2] = 0
    blue_ch[blue_ch == 2] = 0

    # Class 3: Forest dark green
    red_ch[red_ch == 3] = 0
    green_ch[green_ch == 3] = 140
    blue_ch[blue_ch == 3] = 0

    # Class 4: Water? Blue light
    red_ch[red_ch == 4] = 0
    green_ch[green_ch == 4] = 128
    blue_ch[blue_ch == 4] = 255

    # Class 5: Roads
    red_ch[red_ch == 5] = 150
    green_ch[green_ch == 5] = 150
    blue_ch[blue_ch == 5] = 150

    # Class 6: Nothing Yet? Brown
    red_ch[red_ch == 6] = 102
    green_ch[green_ch == 6] = 51
    blue_ch[blue_ch == 6] = 0

    #Append to array
    aux[:, :, 0] = red_ch
    aux[:, :, 1] = green_ch
    aux[:, :, 2] = blue_ch

    print("Exporting class done!")

    return aux






