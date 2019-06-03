import glob
import os
import numpy as np

from cercyt.BD_cytology_25.cell_patch_classification.shared import DIR_rois_matched_level3, DIR_roi_level0

#
# Convert level 3 roi coordinates to level 0 coordinates
#

for coord_path_3 in glob.glob(DIR_rois_matched_level3 + '/*.txt'):
    folder_3, file_3 = os.path.split(coord_path_3)
    coord_file_3 = open(coord_path_3)
    lines_3 = coord_file_3.readlines()
    coord_file_3.close()

    coord_path_0 = os.path.join(DIR_roi_level0, file_3)
    coord_file_0 = open(coord_path_0, 'w')
    for line_3 in lines_3:
        words_3 = line_3.strip().split(',')
        coord_3 = np.array([int(words_3[1]), int(words_3[2]), int(words_3[3]), int(words_3[4])])  # in level 3 resolution
        coord_0 = coord_3 * 8

        coord_str = '{0},{1},{2},{3},{4}\n'.format(words_3[0], coord_0[0], coord_0[1], coord_0[2], coord_0[3])
        coord_file_0.write(coord_str)

    coord_file_0.close()
