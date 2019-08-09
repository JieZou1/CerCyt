import os
import skimage

from cercyt.BD_cytology_25.cell_patch_classification.shared import DataInfo, FILE_DATA_INFO, DIR_cLEAN2
from cercyt.shared.NDPI_Slide import NDPI_Slide


def crop_patch():
    ndpi_path = os.path.join(DIR_cLEAN2, '12XS00147.ndpi')
    ndpi_slide = NDPI_Slide(ndpi_path)

    roi_image = ndpi_slide.read_region((30720, 28672), 0, (2048, 2048))
    skimage.io.imsave('patch.jpg', roi_image)


if __name__ == '__main__':
    crop_patch()
