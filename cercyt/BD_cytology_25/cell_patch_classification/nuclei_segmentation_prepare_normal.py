import glob
import os
import random
import skimage

from cercyt.BD_cytology_25.cell_patch_classification.shared import DataInfo, FILE_DATA_INFO, DIR_tile_patches, \
    DIR_cLEAN2, DIR_nuclei_segmentation
from cercyt.shared.NDPI_Slide import NDPI_Slide

data_info = DataInfo(FILE_DATA_INFO)
patch_size = 1024

# Collect some patches from normal slides
slides = data_info.get_normal_cell_patch_classification_train_slide_ids() + \
         data_info.get_normal_cell_patch_classification_test_slide_ids()
dir_to_save = os.path.join(DIR_nuclei_segmentation, 'normal')

for slide_id in slides:
    patch_folder = os.path.join(DIR_tile_patches, slide_id)
    tif_paths = glob.glob(patch_folder + '/*.tif')

    tif_paths = random.sample(tif_paths, 20)

    # Load over images
    for tif_path in tif_paths:
        print(tif_path)

        folder, tif_file = os.path.split(tif_path)
        ndpi_id, x, y, w, h = tif_file[:-4].split('_')  # Find patch location
        x, y, w, h = int(x), int(y), int(w), int(h)

        # Load NDPI slide
        ndpi_path = os.path.join(DIR_cLEAN2, ndpi_id + '.ndpi')
        ndpi_slide = NDPI_Slide(ndpi_path)

        # Load image and save results
        image = ndpi_slide.read_region((x, y), 0, (patch_size, patch_size))
        cell_path = os.path.join(dir_to_save, '{0}_{1}_{2}_{3}_{4}.bmp'.format(ndpi_id, x, y, patch_size, patch_size))
        skimage.io.imsave(cell_path, image)

