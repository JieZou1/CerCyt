import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET

from mrcnn.config import Config
from openslide import OpenSlide
from mrcnn import utils

DIR_DATASET_ROOT = r'Y:\Cytology\NLM_Data'
DIR_cLEAN2 = os.path.join(DIR_DATASET_ROOT, 'cLEAN2')
DIR_ink = os.path.join(DIR_DATASET_ROOT, 'ink')
DIR_annotated = os.path.join(DIR_DATASET_ROOT, 'annotated')
FILE_DATA_INFO = os.path.join(DIR_DATASET_ROOT, 'data_info.txt')

DIR_GENERATED_OUT_ROOT = r'Y:\Cytology\generated_out'        # The generated_out folder, contains many existing results
DIR_rois_matched_level3 = os.path.join(DIR_GENERATED_OUT_ROOT, 'rois_matched_level3')

DIR_BD_cytology_25 = r'Y:\Users\Jie\CerCyt\BD_cytology_25'
DIR_tile_patches = os.path.join(DIR_BD_cytology_25, 'tile_patches')             # The patches cropped from clean NDPI files
DIR_cell_patch_classification = os.path.join(DIR_BD_cytology_25, 'cell_patch_classification')
DIR_nuclei_segmentation = os.path.join(DIR_BD_cytology_25, 'nuclei_segmentation')

DIR_image_align_ink2clean = os.path.join(DIR_cell_patch_classification, 'image_align_ink2clean')
DIR_roi_level0 = os.path.join(DIR_cell_patch_classification, 'roi_level0')  # The blue ink roi on cLEAN2 slides
DIR_patch_level0 = os.path.join(DIR_cell_patch_classification, 'patch_level0')  # THe blue ink roi patches cropped from cLEAN2 slides
DIR_G_Tom_Patch_normal = os.path.join(DIR_cell_patch_classification, 'G_Tom_Patch_normal')    # the patches cropped from cLEAN2 slides and the G Tom annotation
DIR_G_Tom_Patch_abnormal = os.path.join(DIR_cell_patch_classification, 'G_Tom_Patch_abnormal')    # the patches cropped from cLEAN2 slides and the G Tom annotation
DIR_G_Tom_Patch_malignancy = os.path.join(DIR_cell_patch_classification, 'G_Tom_Patch_malignancy')    # the patches cropped from cLEAN2 slides and the G Tom annotation
DIR_cell_patch_abnormal = os.path.join(DIR_cell_patch_classification, 'cell_patch_abnormal')
DIR_cell_patch_normal = os.path.join(DIR_cell_patch_classification, 'cell_patch_normal')
# DIR_cell_patch_all = os.path.join(DIR_cell_patch_classification, 'cell_patch_all_random_partitioned')
DIR_cell_patch_all = os.path.join(DIR_cell_patch_classification, 'cell_patch_all_slide_partitioned')
DIR_roi_ndpa_level4 = os.path.join(DIR_cell_patch_classification, 'roi_ndpa_level4')  # The level 4 image with roi and ndpa annotation drawn

CELL_SIZE = 256

def union(rect_a, rect_b):
    """
    Union of 2 OpenCV rects
    :param rect_a:
    :param rect_b:
    :return:
    """
    x = min(rect_a[0], rect_b[0])
    y = min(rect_a[1], rect_b[1])
    w = max(rect_a[0] + rect_a[2], rect_b[0] + rect_b[2]) - x
    h = max(rect_a[1] + rect_a[3], rect_b[1] + rect_b[3]) - y
    return x, y, w, h


def intersection(rect_a, rect_b):
    """
    Intersection of 2 OpenCV rects
    :param rect_a:
    :param rect_b:
    :return:
    """
    x = max(rect_a[0], rect_b[0])
    y = max(rect_a[1], rect_b[1])
    w = min(rect_a[0] + rect_a[2], rect_b[0] + rect_b[2]) - x
    h = min(rect_a[1] + rect_a[3], rect_b[1] + rect_b[3]) - y
    if w < 0 or h < 0:
        return ()  # or (0,0,0,0) ?
    return x, y, w, h


class DataInfo:
    def __init__(self, data_info_path):
        data_info_file = open(data_info_path)
        data_info_lines = data_info_file.readlines()
        data_info_file.close()

        # Remove the first line
        data_info_lines = data_info_lines[1:]

        self.data_info = {}
        for data_info_line in data_info_lines:
            words = data_info_line.strip().split('\t')
            self.data_info[words[0]] = words[1]

    def get_abnormal_slide_ids(self):
        ids = [id for id in self.data_info.keys() if not self.data_info[id].startswith('NILM')]
        return ids

    def get_normal_slide_ids(self):
        ids = [id for id in self.data_info.keys() if self.data_info[id].startswith('NILM')]
        return ids

    def get_nilm_slide_ids(self):
        ids = [id for id in self.data_info.keys() if self.data_info[id] == 'NILM']
        return ids

    @staticmethod
    def get_normal_cell_patch_classification_train_slide_ids():
        ids = ['12XS05948', '12XS05976', '15XS00465', '17XS00645', '18XS00065']
        return ids

    @staticmethod
    def get_normal_cell_patch_classification_test_slide_ids():
        ids = ['17XS00128']
        return ids

    @staticmethod
    def get_abnormal_cell_patch_classification_train_slide_ids():
        ids = ['15XS00187', '17XS00037', '17XS00071',               # ASCUS
               '12XS13248', '12XS12129', '12XS12118', '12XS12121',  # LSIL
               '12XS00153', '12XS00301', '12XS00171', '12XS00692',  # HSIL
               '12XS28754',                                         # Adeno
               '12XS24488', '12XS24545'                             # SCC
               ]
        return ids

    @staticmethod
    def get_abnormal_cell_patch_classification_test_slide_ids():
        ids = ['15XS00195',  # ASCUS
               '12XS12147',  # LSIL
               '12XS00147',  # HSIL
               '12XS25358',  # Adeno
               '12XS21804'   # SCC
               ]
        return ids


class AlignImages:

    @staticmethod
    def align(image1, image2):
        """
        Align image1 to image 2
        :param image1:
        :param image2:
        :return: h -- the parameters to align image1 to image2
                im1Reg -- the aligned image of image1
        """

        MAX_FEATURES = 500
        GOOD_MATCH_PERCENT = 0.15

        if image1.shape[2] > 3:
            img1 = image1[:, :, 0:3]
        else:
            img1 = image1

        if image2.shape[2] > 3:
            img2 = image2[:, :, 0:3]
        else:
            img2 = image2

        # Convert images to grayscale
        im1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        im2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(MAX_FEATURES)
        keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

        # Match features.
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        # Draw top matches
        # imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
        # cv2.imwrite("matches.jpg", imMatches)

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        # Use homography
        height, width, channels = img2.shape
        im1Reg = cv2.warpPerspective(img1, h, (width, height))

        # im1Reg = cv2.cvtColor(im1Reg, cv2.COLOR_BGR2RGB)

        return h, im1Reg


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


class PatchDataset(utils.Dataset):

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
                    image_path = os.path.join(DIR_tile_patches, image_id+".tif")
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


