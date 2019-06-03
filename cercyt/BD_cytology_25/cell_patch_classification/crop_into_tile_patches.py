import os
import skimage

from openslide import OpenSlide
import numpy as np

from cercyt.BD_cytology_25.cell_patch_classification.shared import \
    DIR_cLEAN2, DIR_tile_patches


def crop_patches(input_folder, output_folder, ndpi_file, level, patch_width, patch_height, overlap):
    ndpi_id = ndpi_file[:-5]
    ndpi_path = os.path.join(input_folder, ndpi_file)
    ndpi_slide = OpenSlide(ndpi_path)

    width = ndpi_slide.level_dimensions[level][0]
    height = ndpi_slide.level_dimensions[level][1]

    # We crop the (1024x1024) patches from NDPI image with overlapping of (128, 128)
    for x in range(0, width, patch_width - overlap):
        for y in range(0, height, patch_height - overlap):
            image_id = "{}_{}.{}.{}.{}".format(ndpi_file, x, y, patch_width, patch_height)
            image_path = os.path.join(output_folder, ndpi_id, image_id + ".tif")
            position = (int(x), int(y))
            image = ndpi_slide.read_region(position, level, (patch_width, patch_height))
            image = np.array(image)
            if image.shape[2] > 3:
                image = image[:, :, 0:3]  # convert from RGBA to RGB
            skimage.io.imsave(image_path, image)


def save_level_images(input_folder, output_folder, level):
    ndpi_id = ndpi_file[:-5]
    ndpi_path = os.path.join(input_folder, ndpi_file)
    ndpi_slide = OpenSlide(ndpi_path)

    width = ndpi_slide.level_dimensions[level][0]
    height = ndpi_slide.level_dimensions[level][1]

    image = ndpi_slide.read_region((0, 0), level, (width, height))
    image = np.array(image)
    if image.shape[2] > 3:
        image = image[:, :, 0:3]  # convert from RGBA to RGB

    image_path = os.path.join(output_folder, ndpi_id + '.tif')
    skimage.io.imsave(image_path, image)


if __name__ == '__main__':
    ndpi_list = next(os.walk(DIR_cLEAN2))[2]
    # ndpi_list = ndpi_list[:1]

    for ndpi_file in ndpi_list:
        if ndpi_file != '18XS00065.ndpi':
            continue
        crop_patches(DIR_cLEAN2, DIR_tile_patches, ndpi_file, 0, 1024, 1024, 128)
        # save_level_images(DIR_cLEAN2, DIR_NLM_CLEAN2_LEVEL3, 3)
        pass
