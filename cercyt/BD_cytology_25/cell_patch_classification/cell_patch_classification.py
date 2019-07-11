import os
import time

import numpy as np
import keras
from keras import Model
from keras.applications import ResNet50, NASNetLarge, NASNetMobile, InceptionResNetV2, VGG16, VGG19
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.engine.saving import load_model
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras_applications.resnet import ResNet101
from keras_applications.resnet_common import ResNet101V2, ResNeXt101, ResNeXt50, ResNet50V2
from keras.applications.mobilenet_v2 import MobileNetV2
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, precision_score, recall_score, \
    roc_auc_score, roc_curve, f1_score
import matplotlib.pyplot as plt

from cercyt.BD_cytology_25.cell_patch_classification.shared import DataInfo, \
    DIR_cell_patch_abnormal, \
    DIR_cell_patch_normal, DIR_cell_patch_all, DIR_cell_patch_classification


def get_model_config(model_name):
    if model_name == 'DenseNet121':
        width, height = 224, 224
        batch_size = 32
        epoches = 20
    elif model_name == 'DenseNet169':
        width, height = 224, 224
        batch_size = 32
        epoches = 40
    elif model_name == 'DenseNet201':
        width, height = 224, 224
        batch_size = 16
        epoches = 80
    elif model_name == 'InceptionResNetV2':
        width, height = 299, 299
        batch_size = 16
        epoches = 40
    elif model_name == 'MobileNetV2':
        width, height = 224, 224
        batch_size = 32
        epoches = 20
    elif model_name == 'NASNetLarge':
        width, height = 331, 331
        batch_size = 8
        epoches = 80
    elif model_name == 'NASNetMobile':
        width, height = 224, 224
        batch_size = 32
        epoches = 80
    elif model_name == 'ResNet50':
        width, height = 224, 224
        batch_size = 32
        epoches = 20
    elif model_name == 'ResNet101':
        width, height = 224, 224
        batch_size = 32
        epoches = 20
    elif model_name == 'ResNet50V2':
        width, height = 224, 224
        batch_size = 32
        epoches = 40
    elif model_name == 'ResNet101V2':
        width, height = 224, 224
        batch_size = 32
        epoches = 40
    elif model_name == 'ResNeXt50':
        width, height = 224, 224
        batch_size = 16
        epoches = 80
    elif model_name == 'ResNeXt101':
        width, height = 224, 224
        batch_size = 16
        epoches = 80
    elif model_name == 'VGG16':
        width, height = 224, 224
        batch_size = 32
        epoches = 40
    elif model_name == 'VGG19':
        width, height = 224, 224
        batch_size = 32
        epoches = 40

    return width, height, batch_size, epoches


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
    elif model_name == 'ResNet101':
        base_model = ResNet101(weights='imagenet', include_top=include_top, input_shape=input_shape, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
    elif model_name == 'ResNet50V2':
        base_model = ResNet50V2(weights='imagenet', include_top=include_top, input_shape=input_shape, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
    elif model_name == 'ResNet101V2':
        base_model = ResNet101V2(weights='imagenet', include_top=include_top, input_shape=input_shape, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
    elif model_name == 'ResNeXt50':
        base_model = ResNeXt50(weights='imagenet', include_top=include_top, input_shape=input_shape, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
    elif model_name == 'ResNeXt101':
        base_model = ResNeXt101(weights='imagenet', include_top=include_top, input_shape=input_shape, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
    elif model_name == "MobileNetV2":
        base_model = MobileNetV2(weights='imagenet', include_top=include_top, input_shape=input_shape)
    elif model_name == "NASNetLarge":
        base_model = NASNetLarge(weights='imagenet', include_top=include_top, input_shape=input_shape)
    elif model_name == "NASNetMobile":
        base_model = NASNetMobile(weights='imagenet', include_top=include_top, input_shape=input_shape)
    elif model_name == "DenseNet121":
        base_model = DenseNet121(weights='imagenet', include_top=include_top, input_shape=input_shape)
    elif model_name == "DenseNet169":
        base_model = DenseNet169(weights='imagenet', include_top=include_top, input_shape=input_shape)
    elif model_name == "DenseNet201":
        base_model = DenseNet201(weights='imagenet', include_top=include_top, input_shape=input_shape)
    elif model_name == "InceptionResNetV2":
        base_model = InceptionResNetV2(weights='imagenet', include_top=include_top, input_shape=input_shape)
    elif model_name == "VGG16":
        base_model = VGG16(weights='imagenet', include_top=include_top, input_shape=input_shape)
    elif model_name == "VGG19":
        base_model = VGG19(weights='imagenet', include_top=include_top, input_shape=input_shape)
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

    # model_name = 'DenseNet121'
    # model_name = 'DenseNet169'
    # model_name = 'DenseNet201'
    # model_name = 'InceptionResNetV2'
    # model_name = 'MobileNetV2'
    # model_name = 'NASNetLarge'
    # model_name = 'NASNetMobile'
    # model_name = 'ResNet50'
    # model_name = 'ResNet101'
    # model_name = 'ResNet50V2'
    # model_name = 'ResNet101V2'
    # model_name = 'ResNeXt50'
    # model_name = 'ResNeXt101'
    # model_name = 'VGG16'
    model_name = 'VGG19'
    width, height, batch_size, epochs = get_model_config(model_name)

    train_gen, valid_gen = create_generators(
        train_directory=os.path.join(DIR_cell_patch_all, 'train'),
        valid_directory=os.path.join(DIR_cell_patch_all, 'valid'),
        target_size=(height, width),
        batch_size=batch_size
    )

    base_model, model = create_model(
        model_name=model_name,
        input_shape=(height, width, 3),
        include_top=False
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
        steps_per_epoch=train_gen.n//batch_size,
        validation_data=valid_gen,
        validation_steps=valid_gen.n//batch_size,
        # validation_data=train_gen,
        # validation_steps=train_gen.n//batch_size,
        epochs=epochs,
        callbacks=create_callback(model, valid_gen)
    )


def create_test_generators(test_directory, target_size, batch_size):

    test_gen = ImageDataGenerator().flow_from_directory(
        directory=test_directory,
        target_size=target_size,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
    )
    return test_gen


def predict():

    exp_dir = r'Z:\Users\jie\projects\CerCyt\cercyt\BD_cytology_25\cell_patch_classification\Exp'

    model_names = [
        'DenseNet121', 'DenseNet169', 'DenseNet201',
        'InceptionResNetV2', 'MobileNetV2',
        'NASNetLarge', 'NASNetMobile',
        'ResNet50', 'ResNet101', 'ResNet50V2', 'ResNet101V2', 'ResNeXt50', 'ResNeXt101',
        'VGG16', 'VGG19'
        ]
    model_files = dict()
    model_files['DenseNet121'] = os.path.join(exp_dir, 'Slides-DenseNet121-Lr0.000001-Dropout0.5', 'weights.012-0.31330.h5')
    model_files['DenseNet169'] = os.path.join(exp_dir, 'Slides-DenseNet169-Lr0.000001-Dropout0.5', 'weights.010-0.31900.h5')
    model_files['DenseNet201'] = os.path.join(exp_dir, 'Slides-DenseNet201-Lr0.000001-Dropout0.5', 'weights.011-0.28737.h5')
    model_files['InceptionResNetV2'] = os.path.join(exp_dir, 'Slides-InceptionResNetV2-Lr0.000001-Dropout0.5', 'weights.006-0.34031.h5')
    model_files['MobileNetV2'] = os.path.join(exp_dir, 'Slides-MoblieNetV2-Lr0.000001-Dropout0.5', 'weights.015-0.39489.h5')
    model_files['NASNetLarge'] = os.path.join(exp_dir, 'Slides-NASNetLarge-Lr0.000001-Dropout0.5', 'weights.010-0.33526.h5')
    model_files['NASNetMobile'] = os.path.join(exp_dir, 'Slides-NASNetMobile-Lr0.000001-Dropout0.5', 'weights.025-0.47007.h5')
    model_files['ResNet50'] = os.path.join(exp_dir, 'Slides-ResNet50-Lr0.000001-Dropout0.5', 'weights.009-0.32628.h5')
    model_files['ResNet50V2'] = os.path.join(exp_dir, 'Slides-ResNet50V2-Lr0.000001-Dropout0.5', 'weights.025-0.21299.h5')
    model_files['ResNet101'] = os.path.join(exp_dir, 'Slides-ResNet101-Lr0.000001-Dropout0.5', 'weights.007-0.34665.h5')
    model_files['ResNet101V2'] = os.path.join(exp_dir, 'Slides-ResNet101V2-Lr0.000001-Dropout0.5', 'weights.014-0.26708.h5')
    model_files['ResNeXt50'] = os.path.join(exp_dir, 'Slides-ResNeXt50-Lr0.000001-Dropout0.5', 'weights.003-0.33346.h5')
    model_files['ResNeXt101'] = os.path.join(exp_dir, 'Slides-ResNeXt101-Lr0.000001-Dropout0.5', 'weights.004-0.32565.h5')
    model_files['VGG16'] = os.path.join(exp_dir, 'Slides-VGG16-Lr0.000001-Dropout0.5', 'weights.005-0.26584.h5')
    model_files['VGG19'] = os.path.join(exp_dir, 'Slides-VGG19-Lr0.000001-Dropout0.5', 'weights.005-0.27325.h5')

    for model_name in model_names:
        print('Model is: {}'.format(model_name))
        width, height, batch_size, epochs = get_model_config(model_name)
        model_file = model_files[model_name]

        test_folder = os.path.join(DIR_cell_patch_all, 'valid')
        test_gen = create_test_generators(test_folder, (height, width), 1)

        # load model
        model = load_model(model_file, compile=False)
        model.summary()

        time1 = time.time()
        # predict prob. 'abnormal':0, 'normal':1
        probabilities = model.predict_generator(test_gen, len(test_gen.filenames))
        time2 = time.time()

        average_time = (time2 - time1)/len(test_gen.filenames)

        # Switch the colume to make 'abnormal':1, 'normal':0
        probabilities[:, 0], probabilities[:, 1] = probabilities[:, 1], probabilities[:, 0].copy()

        y_true = test_gen.labels == 0  # Also reverse the ground-truth label to make 'abnormal':1, 'normal':0
        y_pred = np.argmax(probabilities, axis=1)

        # metrics
        accu = accuracy_score(y_true, y_pred)
        confusion_mx = confusion_matrix(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        fscore = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        auc = roc_auc_score(y_true, probabilities[:, 1])

        metrics_file = open('metrics_{}.txt'.format(model_name), 'w+')
        np.savetxt(metrics_file, [average_time, accu, precision, recall, fscore, mcc, auc])
        np.savetxt(metrics_file, confusion_mx)
        metrics_file.close()

        # roc curve
        fpr, tpr, thresholds = roc_curve(y_true, probabilities[:, 1])
        curve_roc = np.array([fpr, tpr])
        roc_file = open('roc_{}.txt'.format(model_name), 'w+')
        np.savetxt(roc_file, curve_roc)
        roc_file.close()


def draw_roc():

    model_names = [
        'DenseNet121', 'DenseNet169', 'DenseNet201',
        'InceptionResNetV2', 'MobileNetV2',
        'NASNetLarge', 'NASNetMobile',
        'ResNet50', 'ResNet101', 'ResNet50V2', 'ResNet101V2', 'ResNeXt50', 'ResNeXt101',
        'VGG16', 'VGG19'
                   ]
    fprs = dict()
    tprs = dict()
    aucs = dict()
    for model_name in model_names:
        roc_file = open('roc_{}.txt'.format(model_name), 'r')
        [fpr, tpr] = np.loadtxt(roc_file)
        roc_file.close()

        metrics_file = open('metrics_{}.txt'.format(model_name), 'r')
        average_time, accu, precision, recall, fscore, mcc, auc = np.loadtxt(metrics_file, max_rows=7)
        metrics_file.close()

        fprs[model_name] = fpr
        tprs[model_name] = tpr
        aucs[model_name] = auc

    plt.figure()
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.grid(True)
    plt.title('ROC curves')
    colors = ['maroon', 'orangered', 'saddlebrown', 'darkorange', 'olive',
              'darkgreen', 'lime', 'lightseagreen', 'deepskyblue', 'darkblue',
              'aqua',  'cornflowerblue', 'darkviolet', 'darkmagenta', 'crimson']
    for i, model_name in enumerate(model_names):
        auc = aucs[model_name]
        fpr = fprs[model_name]
        tpr = tprs[model_name]
        if model_name == 'InceptionResNetV2':
            model_name = 'InceptionV4'
        elif model_name == 'NASNetLarge':
            model_name = 'NASNet'
        plt.plot(fpr, tpr, color=colors[i], label='{0}: AUC={1:0.2f}'.format(model_name, auc))
    plt.legend(loc="lower right")
    plt.savefig('rocs.eps', format='eps')


if __name__ == '__main__':
    # train()
    # predict()
    draw_roc()
