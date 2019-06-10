import numpy as np
from openslide import OpenSlide
import xml.etree.ElementTree as ET
import cv2

class NDPI_Slide:

    def __init__(self, ndpi_path):
        self.ndpi_slide = OpenSlide(ndpi_path)

    def read_image(self, level_select):
        """
        Read the whole image at the selected level
        :param level_select:
        :return:
        """
        level_count = self.ndpi_slide.level_count
        level_dims = self.ndpi_slide.level_dimensions

        # for level in range(level_count):
        #     print("level: {} \t dim: {}".format(level, level_dims[level]))

        im_low_res = self.ndpi_slide.read_region(location=(0, 0), level=level_select, size=level_dims[level_select])

        im_low_res = np.array(im_low_res)
        if im_low_res.shape[2] > 3:
            im_low_res = im_low_res[:, :, 0:3]

        return im_low_res

    def read_region(self, location, level, size):
        """
        Read image region
        :param location: (x, y)
        :param level:
        :param size: (w, h)
        :return:
        """
        image = self.ndpi_slide.read_region(location, level, size)
        image = np.array(image)
        if image.shape[2] > 3:
            image = image[:, :, 0:3]  # convert from RGBA to RGB
        return image

    def read_ndpa_annotation(self, ndpa_path):

        mmp_x = float(self.ndpi_slide.properties['openslide.mpp-x'])  # pixel size in um
        mmp_y = float(self.ndpi_slide.properties['openslide.mpp-y'])  # pixel size in um

        # Distance in X from the center of the entire slide (i.e., the macro image) to the center of the main image, in nm
        offset_x = float(self.ndpi_slide.properties['hamamatsu.XOffsetFromSlideCentre'])
        # Distance in Y from the center of the entire slide to the center of the main image, in nm
        offset_y = float(self.ndpi_slide.properties['hamamatsu.YOffsetFromSlideCentre'])
        level_0_dim = self.ndpi_slide.level_dimensions[0]
        level_0_width = level_0_dim[0]
        level_0_height = level_0_dim[1]

        # Steps to convert from NDPA coordinates to pixel coordinates
        # 0. The point coordinates are in physical units, relative to the center of the entire slide
        # 1. Subtract the [XY]OffsetFromSlideCentre ==> physical units, relative to the center of the main image
        # 2. Divide by (mpp-[xy] * 1000) ==> pixel coordinates, relative to the center of the main image
        # 3. Subtract (level-0 dimensions / 2) ==> level 0 pixel coordinates
        root = ET.parse(ndpa_path).getroot()
        polygons = []
        rects = []
        titles = []
        for ndpviewstate_node in root.findall('ndpviewstate'):
            titles.append(ndpviewstate_node.find('title').text)
            annotation_node = ndpviewstate_node.find('annotation')
            pointlist_node = annotation_node.find('pointlist')
            points = []
            for point_node in pointlist_node.findall('point'):
                x, y = float(point_node.find('x').text), float(point_node.find('y').text)  # physical position
                x, y = x - offset_x, y - offset_y  # physical position with respect to image center
                x, y = x / (mmp_x * 1000), y / (mmp_y * 1000)  # pixel position with respect to image center
                x, y = x + level_0_width / 2, y + level_0_height / 2  # pixel position with respect to image origin
                points.append((int(x), int(y)))

            points = np.array(points)
            rect = cv2.boundingRect(points)
            polygons.append(points)
            rects.append(rect)

        return polygons, rects, titles

