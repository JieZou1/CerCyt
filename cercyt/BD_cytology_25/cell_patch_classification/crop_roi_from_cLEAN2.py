import glob
import os
import skimage

from cercyt.BD_cytology_25.cell_patch_classification.shared import \
    DIR_cLEAN2, \
    DIR_roi_level0, \
    NDPI_Slide, \
    DIR_patch_level0

#
# Crop level 0 roi and save as TIF images
# Totally 6 slides have no blue ink ROI, we ignore them
#

for coord_path in glob.glob(DIR_roi_level0 + '/*.txt'):

    # Read coord info
    coord_file = open(coord_path)
    lines = coord_file.readlines()
    coord_file.close()

    # Read NDPI slide
    folder, file = os.path.split(coord_path)
    ndpi_id = file[:-4]
    ndpi_path = os.path.join(DIR_cLEAN2, ndpi_id+'.ndpi')
    ndpi_slide = NDPI_Slide(ndpi_path)

    # Crop the ROIs
    for line in lines:
        words = line.strip().split(',')
        x, y, w, h = int(words[1]), int(words[2]), int(words[3]), int(words[4])
        roi_image = ndpi_slide.read_region((x, y), 0, (w, h))
        roi_file = '{0}_{1}_{2}_{3}_{4}.tif'.format(ndpi_id, x, y, w, h)
        roi_path = os.path.join(DIR_patch_level0, roi_file)
        skimage.io.imsave(roi_path, roi_image)
