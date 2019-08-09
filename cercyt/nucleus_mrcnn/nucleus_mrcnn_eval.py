import os

import numpy as np
from pycocotools.coco import COCO
import skimage.io
from pycocotools.cocoeval import COCOeval


def show_gt_annotation():
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt

    dir_gt = R'Y:\Users\Jie\CerCyt\BD_cytology_25\nuclei_segmentation\normal-5_coco'
    ann_file = os.path.join(dir_gt, 'annotations.json')
    coco = COCO(ann_file)

    catIds = coco.getCatIds(catNms=['nuclei'])
    imgIds = coco.getImgIds(catIds=catIds)

    for img_id in imgIds:
        img = coco.loadImgs(img_id)[0]
        I = skimage.io.imread('{}/{}'.format(dir_gt, img['file_name']))
        plt.clf()
        plt.imshow(I)
        plt.axis('off')
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        coco.showAnns(anns)
        plt.show()


def coco_eval():
    # initialize COCO ground truth api
    dir_gt = r'Y:\Users\Jie\CerCyt\BD_cytology_25\nuclei_segmentation\normal-abnormal_coco'
    ann_file = os.path.join(dir_gt, 'annotations.json')
    cocoGT = COCO(ann_file)

    # initialize COCO detections api
    result_dir = r'Y:\Users\Jie\CerCyt\BD_cytology_25\nuclei_segmentation\normal-abnormal_predicted'
    resFile = os.path.join(result_dir, 'results.json')
    cocoDt = cocoGT.loadRes(resFile)
    imgIds = cocoGT.getImgIds()

    # imgIds = sorted(cocoGT.getImgIds())
    # imgIds = imgIds[0:100]
    # imgId = imgIds[np.random.randint(100)]

    cocoEval = COCOeval(cocoGT, cocoDt, 'segm')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    # show_gt_annotation()
    coco_eval()
