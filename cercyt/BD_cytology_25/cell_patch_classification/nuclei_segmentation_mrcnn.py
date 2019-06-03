# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import glob
import os
import random
import cv2
import skimage

from mrcnn import model as modellib, visualize

from cercyt.nucleus_mrcnn.nucleus_mrcnn import NucleusInferenceConfig
from cercyt.BD_cytology_25.cell_patch_classification.shared import DIR_nuclei_segmentation, DIR_cLEAN2, NDPI_Slide

model_path = r'Y:\Users\Jie\CerCyt\nucleus_mrcnn\models\mask_rcnn_nucleus_0040-16_kaggle.h5'

config = NucleusInferenceConfig()
config.display()
model = modellib.MaskRCNN(mode="inference", config=config, model_dir='')
model.load_weights(model_path, by_name=True)

image_folder = os.path.join(DIR_nuclei_segmentation, 'normal')
image_paths = glob.glob(image_folder + '/*.bmp')

for image_path in image_paths:
    folder, tif_file = os.path.split(image_path)
    ndpi_id, x, y, w, h = tif_file[:-4].split('_')  # Find patch location
    x, y, w, h = int(x), int(y), int(w), int(h)

    # Load NDPI slide
    ndpi_path = os.path.join(DIR_cLEAN2, ndpi_id + '.ndpi')
    ndpi_slide = NDPI_Slide(ndpi_path)

    # Load image and run detection
    image = ndpi_slide.read_region((x, y), 0, (h, w))
    height, width, depth = image.shape
    image = cv2.resize(image, (int(width / 2), int(height / 2)))
    detection = model.detect([image])
    if len(detection) == 0:
        continue

    results = detection[0]

    # save segmentation for visualization
    vis_path = image_path.replace('.bmp', '.jpg')
    visualize.display_instances(
        image, results['rois'], results['masks'], results['class_ids'],
        'nucleus', results['scores'],
        show_bbox=False, show_mask=False,
        title="Predictions")
    plt.savefig(vis_path)



