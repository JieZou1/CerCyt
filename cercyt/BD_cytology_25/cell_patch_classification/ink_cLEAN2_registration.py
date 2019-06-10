import glob
import os
import numpy as np

from cercyt.BD_cytology_25.cell_patch_classification.shared import \
    DIR_cLEAN2, DIR_ink, DIR_image_align_ink2clean, \
    AlignImages \
from cercyt.shared.NDPI_Slide import NDPI_Slide

clean_ndpi_paths = glob.glob(DIR_cLEAN2 + '/*.ndpi')

for clean_ndpi_path in clean_ndpi_paths:
    _, ndpi_file = os.path.split(clean_ndpi_path)
    ink_ndpi_path = os.path.join(DIR_ink, ndpi_file)

    clean_slide = NDPI_Slide(clean_ndpi_path)
    ink_slide = NDPI_Slide(ink_ndpi_path)

    clean_img_level_2 = clean_slide.read_image(2)
    ink_img_level_2 = ink_slide.read_image(2)

    h, aligned_ink_image = AlignImages.align(ink_img_level_2, clean_img_level_2)

    # save the alignment parameters
    ndpi_id = ndpi_file[:-5]
    align_path = os.path.join(DIR_image_align_ink2clean, ndpi_id+".txt")
    np.savetxt(align_path, np.asarray(h), delimiter=",")
