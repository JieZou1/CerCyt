import os
import numpy as np
import cv2
import glob

import keras
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

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

    # a dictionary holds 'train' and 'validation ids
    partition = {'train': train_normal_ids + train_abnormal_ids,
                 'validation': validation_normal_ids + validation_abnormal_ids}
    # a dictionary holds classification labels
    labels = {}
    for id in train_normal_ids:
        labels[id] = 0
    for id in train_abnormal_ids:
        labels[id] = 1
    for id in validation_normal_ids:
        labels[id] = 0
    for id in validation_abnormal_ids:
        labels[id] = 1

    return partition, labels


class DataGenerator(keras.utils.Sequence):
    """    Generates data for Keras    """

    def __init__(self, list_IDs, labels, batch_size, dim, n_channels, n_classes, shuffle=True):
        """
        Initialization
        :param list_IDs: The ids of this sample set
        :param labels: Labels of all samples
        :param batch_size:
        :param dim:
        :param n_channels:
        :param n_classes:
        :param shuffle:
        """
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        :return:
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index:
        :return:
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        '''
        Updates indexes after each epoch
        :return:
        '''
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples # X : (n_samples, *dim, n_channels)
        :param list_IDs_temp:
        :return:
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i, ] = cv2.resize(cv2.imread(ID), self.dim)

            # Store class
            y[i] = self.labels[ID]
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def load_image(self, image_id):
        return cv2.resize(cv2.imread(image_id), self.dim)

    def sample_size(self):
        return len(self.list_IDs)


def create_train_generator(dim, batch_size):
    data_gen = ImageDataGenerator()
    generator = data_gen.flow_from_directory(
        directory=os.path.join(DIR_cell_patch_all, 'train'),
        target_size=dim,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        # seed=42
    )
    return generator


def create_validation_generator(dim, batch_size):
    data_gen = ImageDataGenerator()
    generator = data_gen.flow_from_directory(
        directory=os.path.join(DIR_cell_patch_all, 'valid'),
        target_size=dim,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        # seed=42
    )
    return generator


def evaluate(generator, model):
    # get GT annotations
    gt_labels = [generator.labels[i] for i in generator.list_IDs]
    # get predictions
    pred_labels = []
    for list_id in generator.list_IDs:
        image = generator.load_image(list_id)
        result = model.predict_on_batch(np.expand_dims(image, axis=0))
        pred_labels.append(np.argmax(result, axis=1)[0])
    accu = accuracy_score(gt_labels, pred_labels)
    return accu


class Evaluate(keras.callbacks.Callback):
    def __init__(self, model, generator, tensorboard=None):
        self.generator = generator
        self.tensorboard = tensorboard
        # self.model = model

        super(Evaluate, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        print(logs)


def create_callback(model, validation_generator):
    callbacks = []

    evaluation = Evaluate(model, validation_generator)

    callbacks.append(evaluation)

    return callbacks

