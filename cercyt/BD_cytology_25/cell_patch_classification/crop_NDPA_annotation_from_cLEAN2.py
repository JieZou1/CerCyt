import os
import skimage
import numpy as np

from cercyt.BD_cytology_25.cell_patch_classification.shared import \
    DataInfo, NDPI_Slide, \
    FILE_DATA_INFO, DIR_annotated, DIR_ink, DIR_image_align_ink2clean, DIR_cLEAN2, DIR_G_Tom_Patch_normal, \
    DIR_G_Tom_Patch_malignancy, DIR_G_Tom_Patch_abnormal

data_info = DataInfo(FILE_DATA_INFO)
slide_ids = data_info.get_abnormal_slide_ids()

for ndpi_id in slide_ids:
    # read NDPA annotation
    ndpa_annotation_path = os.path.join(DIR_annotated, ndpi_id + '.ndpi.ndpa')
    if not os.path.exists(ndpa_annotation_path):
        print('Can not find ' + ndpa_annotation_path)

    ink_ndpi_slide_path = os.path.join(DIR_ink, ndpi_id+'.ndpi')
    ink_slide = NDPI_Slide(ink_ndpi_slide_path)
    polygons_ink, rects_ink, title_ink = ink_slide.read_ndpa_annotation(ndpa_annotation_path)
    # ink_patch = ink_slide.read_region((rects_ink[0][0], rects_ink[0][1]), 0, (rects_ink[0][2], rects_ink[0][3]))
    # skimage.io.imsave('ink_patch.bmp', ink_patch)

    # read ink2cLEARN align parameters
    align_path = os.path.join(DIR_image_align_ink2clean, ndpi_id+'.txt')
    align_para = np.loadtxt(align_path, delimiter=",")

    # covert to cLEAN2 image coord
    rects_cLEAN2 = []  # stores NDPA abnormal annotation in cLEARN2 coordinate
    x_offset = align_para[0, 2] * 4  # align parameter is at Level 2 resolution
    y_offset = align_para[1, 2] * 4  # We work on Level 0 resolution, so there is a 2^2 difference in scale
    for rect_ink in rects_ink:
        rect_cLEAN = (int(rect_ink[0] + x_offset), int(rect_ink[1] + y_offset), rect_ink[2], rect_ink[3])
        rects_cLEAN2.append(rect_cLEAN)

    clean_ndpi_slide_path = os.path.join(DIR_cLEAN2, ndpi_id + '.ndpi')
    clean_slide = NDPI_Slide(clean_ndpi_slide_path)

    # read cLEAN2 slide to crop for verification
    for i, rect_cLEAN in enumerate(rects_cLEAN2):
        clean_patch = clean_slide.read_region((rect_cLEAN[0], rect_cLEAN[1]), 0, (rect_cLEAN[2], rect_cLEAN[3]))
        print('{0}:{1}'.format(ndpi_id, title_ink[i]))
        if title_ink[i] == 'Nothing here' or \
                title_ink[i] == 'false positive' or \
                title_ink[i] == 'not sure what they are indicating here' or \
                title_ink[i] == 'False positive' or \
                title_ink[i] == 'Not ASCUS. Several cells on top of each other.' or \
                title_ink[i] == 'Inflammation- neutrophils (they\'re all over the place)':
                # title_ink[i] == 'They accidentally covered the cells in ink' or \
            clean_path = os.path.join(DIR_G_Tom_Patch_normal,
                                      '{0}_{1}_{2}_{3}_{4}.tif'.format(ndpi_id, rect_cLEAN[0], rect_cLEAN[1],
                                                                       rect_cLEAN[2], rect_cLEAN[3]))
        elif title_ink[i] == 'almost every red cell in this box is suspicious for malignancy':
            clean_path = os.path.join(DIR_G_Tom_Patch_malignancy,
                                      '{0}_{1}_{2}_{3}_{4}.tif'.format(ndpi_id, rect_cLEAN[0], rect_cLEAN[1],
                                                                       rect_cLEAN[2], rect_cLEAN[3]))
        else:
            clean_path = os.path.join(DIR_G_Tom_Patch_abnormal,
                                      '{0}_{1}_{2}_{3}_{4}.tif'.format(ndpi_id, rect_cLEAN[0], rect_cLEAN[1],
                                                                       rect_cLEAN[2], rect_cLEAN[3]))
        skimage.io.imsave(clean_path, clean_patch)
