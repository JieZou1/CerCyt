
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

from mrcnn import visualize
from mrcnn import model as modellib

from cercyt.nucleus_mrcnn.nucleus_mrcnn import NucleusInferenceConfig

from cercyt.BD_cytology_25.cell_patch_classification.shared import \
    PatchDataset, \
    DIR_G_Tom_Patch_abnormal, DIR_cLEAN2, DIR_annotated, intersection, DIR_cell_patch_abnormal, DIR_cell_patch_normal, \
    CELL_SIZE
from cercyt.shared.NDPI_Slide import NDPI_Slide


patch_folder = DIR_G_Tom_Patch_abnormal
model_path = r'Y:\Users\Jie\CerCyt\nucleus_mrcnn\models\mask_rcnn_nucleus_0040-16_kaggle.h5'
patch_size = 1024

config = NucleusInferenceConfig()
config.display()
model = modellib.MaskRCNN(mode="inference", config=config, model_dir='')
model.load_weights(model_path, by_name=True)

# Load over images
tif_paths = glob.glob(patch_folder + '/*.tif')
for tif_path in tif_paths:
    """Run segmentation on an image."""
    print(tif_path)

    folder, tif_file = os.path.split(tif_path)
    ndpi_id, x, y, w, h = tif_file[:-4].split('_')  # Find patch location
    x, y, w, h = int(x), int(y), int(w), int(h)

    # Load NDPI slide
    ndpi_path = os.path.join(DIR_cLEAN2, ndpi_id+'.ndpi')
    ndpi_slide = NDPI_Slide(ndpi_path)

    # Load NDPA annotation
    ndpa_annotation_path = os.path.join(DIR_annotated, ndpi_id + '.ndpi.ndpa')
    polygons, rects, titles = ndpi_slide.read_ndpa_annotation(ndpa_annotation_path)
    # Pick the one annotation corresponding to this patch
    for i, rect in enumerate(rects):
        if len(intersection(rect, (x, y, w, h))) is not 0:
            polygon = polygons[i]
            break

    # the top left corner of the patch to run mrcnn, we work at level 1 resolution
    x_0, y_0 = float(x) + float(w) / 2.0, float(y) + float(h) / 2.0
    location = int(x_0 - patch_size / 2 + 0.5), int(y_0 - patch_size / 2 + 0.5)

    polygon = polygon - location
    polygon_mask = np.zeros((patch_size, patch_size), np.uint8)
    polygon_mask = cv2.fillPoly(polygon_mask, [polygon], 255)
    height, width = polygon_mask.shape
    polygon_mask = cv2.resize(polygon_mask, (int(width/2), int(height/2)))

    # Load image and run detection
    image = ndpi_slide.read_region(location, 0, (patch_size, patch_size))
    height, width, depth = image.shape
    image = cv2.resize(image, (int(width/2), int(height/2)))
    detection = model.detect([image])
    if len(detection) == 0:
        continue
    results = detection[0]

    # save segmentation for visualization
    vis_path = tif_path.replace('.tif', '.jpg')
    visualize.display_instances(
        image, results['rois'], results['masks'], results['class_ids'],
        'nucleus', results['scores'],
        show_bbox=False, show_mask=False,
        title="Predictions")
    plt.savefig(vis_path)

    # save individual cell patches
    masks = results['masks']
    height, width, depth = masks.shape
    for i in range(depth):
        mask = masks[:, :, i].astype('uint8') * 255
        # cv2.imshow('mask', mask)  # display for checking
        # cv2.waitKey()

        # match to annotation (polygon) mask to get the cell patch
        intersect = cv2.bitwise_and(polygon_mask, mask)
        intersect = cv2.findNonZero(intersect)
        if intersect is not None:
            M = cv2.moments(mask)
            c_x, c_y = M["m10"] / M["m00"], M["m01"] / M["m00"]
            # convert back to level 0 coordinate
            c_x, c_y = location[0] + c_x*2, location[1] + c_y*2
            loc_x, loc_y = int(c_x - CELL_SIZE/2 + 0.5), int(c_y - CELL_SIZE/2 + 0.5)
            cell_patch = ndpi_slide.read_region((loc_x, loc_y), 0, (CELL_SIZE, CELL_SIZE))

            cell_path = os.path.join(DIR_cell_patch_abnormal, '{0}_{1}_{2}_{3}_{4}.tif'.format(ndpi_id, loc_x, loc_y, CELL_SIZE, CELL_SIZE))
            skimage.io.imsave(cell_path, cell_patch)
