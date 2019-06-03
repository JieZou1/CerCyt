import glob
import os
from random import shuffle
from shutil import copyfile

from cercyt.BD_cytology_25.cell_patch_classification.shared import DataInfo, DIR_cell_patch_normal, \
    DIR_cell_patch_abnormal, DIR_cell_patch_all


def load_data():
    train_normal_ids = []
    for slide_id in DataInfo.get_normal_cell_patch_classification_train_slide_ids():
        train_normal_ids += glob.glob(DIR_cell_patch_normal + "/{0}*.tif".format(slide_id))

    train_abnormal_ids = []
    for slide_id in DataInfo.get_abnormal_cell_patch_classification_train_slide_ids():
        train_abnormal_ids += glob.glob(DIR_cell_patch_abnormal + "/{0}*.tif".format(slide_id))

    validation_normal_ids = []
    for slide_id in DataInfo.get_normal_cell_patch_classification_test_slide_ids():
        validation_normal_ids += glob.glob(DIR_cell_patch_normal + "/{0}*.tif".format(slide_id))

    validation_abnormal_ids = []
    for slide_id in DataInfo.get_abnormal_cell_patch_classification_test_slide_ids():
        validation_abnormal_ids += glob.glob(DIR_cell_patch_abnormal + "/{0}*.tif".format(slide_id))

    return train_normal_ids, train_abnormal_ids, validation_normal_ids, validation_abnormal_ids


train_normal_ids, train_abnormal_ids, validation_normal_ids, validation_abnormal_ids = load_data()

# for id in train_normal_ids:
#     folder, file = os.path.split(id)
#     path = os.path.join(DIR_cell_patch_all, 'train', 'normal', file)
#     copyfile(id, path)
#
# for id in train_abnormal_ids:
#     folder, file = os.path.split(id)
#     path = os.path.join(DIR_cell_patch_all, 'train', 'abnormal', file)
#     copyfile(id, path)
#
# for id in validation_normal_ids:
#     folder, file = os.path.split(id)
#     path = os.path.join(DIR_cell_patch_all, 'valid', 'normal', file)
#     copyfile(id, path)
#
# for id in validation_abnormal_ids:
#     folder, file = os.path.split(id)
#     path = os.path.join(DIR_cell_patch_all, 'valid', 'abnormal', file)
#     copyfile(id, path)


# normal_ids = train_normal_ids + validation_normal_ids
# abnormal_ids = train_abnormal_ids + validation_abnormal_ids
# shuffle(normal_ids)
# shuffle(abnormal_ids)
#
# train_normal_ids = normal_ids[:5856]
# validation_normal_ids = normal_ids[5856:5856+700]
# train_abnormal_ids = abnormal_ids[:4226]
# validation_abnormal_ids = abnormal_ids[4226:]
#
# for id in train_normal_ids:
#     folder, file = os.path.split(id)
#     path = os.path.join(DIR_cell_patch_all, 'train', 'normal', file)
#     copyfile(id, path)
#
# for id in train_abnormal_ids:
#     folder, file = os.path.split(id)
#     path = os.path.join(DIR_cell_patch_all, 'train', 'abnormal', file)
#     copyfile(id, path)
#
# for id in validation_normal_ids:
#     folder, file = os.path.split(id)
#     path = os.path.join(DIR_cell_patch_all, 'valid', 'normal', file)
#     copyfile(id, path)
#
# for id in validation_abnormal_ids:
#     folder, file = os.path.split(id)
#     path = os.path.join(DIR_cell_patch_all, 'valid', 'abnormal', file)
#     copyfile(id, path)
