
# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

# Read dataset
import os
import glob
import cv2
import numpy as np
import skimage
import random

from mrcnn import visualize
from mrcnn import model as modellib

from cercyt.nucleus_mrcnn.nucleus_mrcnn import NucleusInferenceConfig

from cercyt.BD_cytology_25.cell_patch_classification.shared import \
    PatchDataset, \
    DIR_cLEAN2, DIR_cell_patch_normal, \
    CELL_SIZE, DataInfo, FILE_DATA_INFO, DIR_tile_patches

from cercyt.shared.NDPI_Slide import NDPI_Slide


data_info = DataInfo(FILE_DATA_INFO)
slides = data_info.get_normal_cell_patch_classification_train_slide_ids() + \
         data_info.get_normal_cell_patch_classification_test_slide_ids()

model_path = r'Y:\Users\Jie\CerCyt\nucleus_mrcnn\models\mask_rcnn_nucleus_0040-16_kaggle.h5'
patch_size = 1024

config = NucleusInferenceConfig()
config.display()
model = modellib.MaskRCNN(mode="inference", config=config, model_dir='')
model.load_weights(model_path, by_name=True)

# Load over slides
for slide_id in slides:
    patch_folder = os.path.join(DIR_tile_patches, slide_id)
    tif_paths = glob.glob(patch_folder + '/*.tif')

    tif_paths = random.sample(tif_paths, 500)

    # Load over images
    for tif_path in tif_paths:
        """Run segmentation on an image."""
        print(tif_path)

        folder, tif_file = os.path.split(tif_path)
        ndpi_id, x, y, w, h = tif_file[:-4].split('_')  # Find patch location
        x, y, w, h = int(x), int(y), int(w), int(h)

        # Load NDPI slide
        ndpi_path = os.path.join(DIR_cLEAN2, ndpi_id + '.ndpi')
        ndpi_slide = NDPI_Slide(ndpi_path)

        # Load image and run detection
        image = ndpi_slide.read_region((x, y), 0, (patch_size, patch_size))
        height, width, depth = image.shape
        image = cv2.resize(image, (int(width / 2), int(height / 2)))
        detection = model.detect([image])
        if len(detection) == 0:
            continue
        results = detection[0]

        # save segmentation for visualization
        # vis_path = tif_path.replace('.tif', '.jpg')
        # visualize.display_instances(
        #     image, results['rois'], results['masks'], results['class_ids'],
        #     'nucleus', results['scores'],
        #     show_bbox=False, show_mask=False,
        #     title="Predictions")
        # plt.savefig(vis_path)

        # save individual cell patches
        masks = results['masks']
        scores = results['scores']
        if len(scores) == 0:
            continue
        height, width, depth = masks.shape
        # for i in range(depth):
        for i in range(min(depth, 3)):
            if scores[i] < 0.9:
                continue
            mask = masks[:, :, i].astype('uint8') * 255
            # cv2.imshow('mask', mask)  # display for checking
            # cv2.waitKey()

            M = cv2.moments(mask)
            c_x, c_y = M["m10"] / M["m00"], M["m01"] / M["m00"]
            # convert back to level 0 coordinate
            c_x, c_y = x + c_x * 2, y + c_y * 2
            loc_x, loc_y = int(c_x - CELL_SIZE / 2 + 0.5), int(c_y - CELL_SIZE / 2 + 0.5)
            cell_patch = ndpi_slide.read_region((loc_x, loc_y), 0, (CELL_SIZE, CELL_SIZE))

            cell_path = os.path.join(DIR_cell_patch_normal,
                                     '{0}_{1}_{2}_{3}_{4}.tif'.format(ndpi_id, loc_x, loc_y, CELL_SIZE, CELL_SIZE))
            skimage.io.imsave(cell_path, cell_patch)



