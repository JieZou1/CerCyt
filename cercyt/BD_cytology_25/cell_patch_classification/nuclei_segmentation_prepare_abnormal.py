import glob
import os
import random
import skimage

from cercyt.BD_cytology_25.cell_patch_classification.shared import DataInfo, FILE_DATA_INFO, DIR_tile_patches, \
    DIR_cLEAN2, DIR_nuclei_segmentation, DIR_G_Tom_Patch_abnormal
from cercyt.shared.NDPI_Slide import NDPI_Slide

patch_size = 1024
dir_to_save = os.path.join(DIR_nuclei_segmentation, 'abnormal')

patch_folder = DIR_G_Tom_Patch_abnormal
files = [
    '12XS00147_45572_38495_161_95.tif',
    '12XS00153_53437_32752_108_98.tif',
    '12XS00301_35755_27563_91_103.tif',
    '12XS13248_28279_32049_222_252.tif',
    '12XS21804_27841_7681_165_152.tif',
    '17XS00037_29407_45600_225_209.tif'
]

for file in files:
    tif_path = os.path.join(patch_folder, file)
    print(tif_path)

    folder, tif_file = os.path.split(tif_path)
    ndpi_id, x, y, w, h = tif_file[:-4].split('_')  # Find patch location
    x, y, w, h = int(x), int(y), int(w), int(h)

    # calculate (x,y) for 1024x1024 patches
    c_x, c_y = x + w / 2, y + h / 2
    x, y = int(c_x - patch_size / 2), int(c_y - patch_size / 2)

    # Load NDPI slide
    ndpi_path = os.path.join(DIR_cLEAN2, ndpi_id + '.ndpi')
    ndpi_slide = NDPI_Slide(ndpi_path)

    # Load image and save results
    image = ndpi_slide.read_region((x, y), 0, (patch_size, patch_size))
    cell_path = os.path.join(dir_to_save, '{0}_{1}_{2}_{3}_{4}.bmp'.format(ndpi_id, x, y, patch_size, patch_size))
    skimage.io.imsave(cell_path, image)

