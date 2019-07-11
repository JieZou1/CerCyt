from openslide import OpenSlide


class MIRAX_Slide:

    def __init__(self, mirax_path):
        self.mirax_slide = OpenSlide(mirax_path)


if __name__ == '__main__':
    mirax_path = ''
    mirax_slide = MIRAX_Slide(mirax_path)

