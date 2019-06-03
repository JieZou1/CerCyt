import os
import numpy as np
import skimage.io
from openslide import OpenSlide

from mrcnn import utils

from cercyt.config import DIR_NLM_DATA, DIR_NLM_PATCH


class NlmNucleusDataset(utils.Dataset):

    def load_nucleus_from_patches(self, dataset_dir):
        image_list = next(os.walk(dataset_dir))[2]
        # Add images
        for image_id in image_list:
            if image_id.endswith('.tif'):
                self.add_image(
                    "nucleus",
                    image_id=image_id,
                    path=os.path.join(dataset_dir, image_id))

    def load_nucleus_from_ndpi(self, dataset_dir, subset):
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("nucleus", 1, "nucleus")

        assert subset in ["cLEAN2", "ink"]
        dataset_path = os.path.join(dataset_dir, subset)

        ndpi_list = next(os.walk(dataset_path))[2]

        for ndpi_file in ndpi_list:
            ndpi_path = os.path.join(dataset_dir, subset, ndpi_file)
            ndpi_slide = OpenSlide(ndpi_path)

            width = ndpi_slide.level_dimensions[0][0]
            height = ndpi_slide.level_dimensions[0][1]

            # We crop the (1024x1024) patches from NDPI image with overlapping of (128, 128)
            for y in range(0, height, 1024-128):
                for x in range(0, width, 1024-128):
                    image_id = "{}.{}.{}.{}".format(ndpi_file, x, y, 1024, 1024)
                    image_path = os.path.join(DIR_NLM_PATCH, image_id+".tif")
                    self.add_image(
                        "nucleus",
                        image_id=image_id,
                        path=image_path
                    )

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nucleus":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)

