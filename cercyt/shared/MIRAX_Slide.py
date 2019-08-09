import skimage

import numpy as np
from openslide import OpenSlide


class MIRAX_Slide:

    def __init__(self, mirax_path):
        self.mirax_slide = OpenSlide(mirax_path)

    def read_region(self, location, level, size):
        """
        Read image region
        :param location: (x, y)
        :param level:
        :param size: (w, h)
        :return:
        """
        image = self.mirax_slide.read_region(location, level, size)
        image = np.array(image)
        if image.shape[2] > 3:
            image = image[:, :, 0:3]  # convert from RGBA to RGB
        return image

    def read_properties(self):

        mmp_x = float(self.mirax_slide.properties['openslide.mpp-x'])  # pixel size in um
        mmp_y = float(self.mirax_slide.properties['openslide.mpp-y'])  # pixel size in um

        # Distance in X from the center of the entire slide (i.e., the macro image) to the center of the main image, in nm
        offset_x = float(self.mirax_slide.properties['hamamatsu.XOffsetFromSlideCentre'])
        # Distance in Y from the center of the entire slide to the center of the main image, in nm
        offset_y = float(self.mirax_slide.properties['hamamatsu.YOffsetFromSlideCentre'])
        level_0_dim = self.mirax_slide.level_dimensions[0]
        level_0_width = level_0_dim[0]
        level_0_height = level_0_dim[1]



if __name__ == '__main__':
    mirax_path = r'Y:\Cytology\BD_MIRAX\3DHistech\slide-2019-03-19T20-55-53-R1-S25.mrxs'
    mirax_slide = MIRAX_Slide(mirax_path)
    x = int(mirax_slide.mirax_slide.level_dimensions[0][0]/2)
    y = int(mirax_slide.mirax_slide.level_dimensions[0][1]/2)
    image = mirax_slide.read_region((x, y), 0, (1024, 1024))
    skimage.io.imsave('temp.jpg', image)

    mirax_slide.read_properties()

