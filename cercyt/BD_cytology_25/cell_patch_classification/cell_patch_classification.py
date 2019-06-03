import os

import cv2
import numpy as np
import glob
import keras
from keras import Model
from keras.applications import ResNet50
from keras.applications.densenet import densenet
import keras_resnet
import keras_resnet.models
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.engine.saving import load_model
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras_applications.resnext import ResNeXt50
from keras.applications.mobilenet_v2 import MobileNetV2
from sklearn.metrics import accuracy_score

from cercyt.BD_cytology_25.cell_patch_classification.shared import DataInfo, \
    DIR_cell_patch_abnormal, \
    DIR_cell_patch_normal, DIR_cell_patch_all


def create_generators(train_directory, valid_directory, target_size, batch_size):

    train_gen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True).flow_from_directory(
        directory=train_directory,
        target_size=target_size,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        # seed=42
    )

    valid_gen = ImageDataGenerator().flow_from_directory(
        directory=valid_directory,
        target_size=target_size,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        # seed=42
    )
    return train_gen, valid_gen


def create_model(model_name, include_top, input_shape):
    if model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=include_top, input_shape=input_shape)
    elif model_name == "MobileNetV2":
        base_model = MobileNetV2(weights='imagenet', include_top=include_top, input_shape=input_shape)
    else:
        return None, None

    x = base_model.output
    # x = Flatten()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    return base_model, model


def create_callback(model, validation_generator):
    callbacks = []

    # save model
    model_checkpoint = ModelCheckpoint(
        filepath='weights.{epoch:03d}-{val_loss:.5f}.h5',
        save_best_only=True,
        save_weights_only=False,
                    )
    callbacks.append(model_checkpoint)

    # tensor board
    tensor_board = TensorBoard(
        log_dir='./log',
        update_freq='epoch'
    )
    callbacks.append(tensor_board)

    return callbacks


def train():
    WIDTH, HEIGHT = 224, 224
    BATCH_SIZE = 32

    train_gen, valid_gen = create_generators(
        train_directory=os.path.join(DIR_cell_patch_all, 'train'),
        valid_directory=os.path.join(DIR_cell_patch_all, 'valid'),
        target_size=(HEIGHT, WIDTH),
        batch_size=BATCH_SIZE
    )

    base_model, model = create_model(
        model_name='ResNet50',
        # model_name='MobileNetV2',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all base_model layers
    # for layer in base_model.layers:
    #     layer.trainable = False
    model.compile(optimizer=keras.optimizers.adam(lr=0.000001), loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=keras.optimizers.rmsprop(), loss='categorical_crossentropy')
    # model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    model.fit_generator(
        generator=train_gen,
        steps_per_epoch=train_gen.n//BATCH_SIZE,
        validation_data=valid_gen,
        validation_steps=valid_gen.n//BATCH_SIZE,
        # validation_data=train_gen,
        # validation_steps=train_gen.n//BATCH_SIZE,
        epochs=100,
        callbacks=create_callback(model, valid_gen)
    )


def predict():
    model = load_model('.hdf5')


if __name__ == '__main__':
    train()
