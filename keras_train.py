from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras import backend as kb
from keras.models import load_model
from keras.preprocessing import image
# metrics
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
# for file management
import util.file_util as file_util
import math
import os


def save_model(model, file_name: str = None):
    model_directory = 'keras_model/'
    weights_directory = 'keras_weights/'
    extension = '.h5'

    if file_name is None:
        file_name = file_util.date_string_now()

    model.save_weights(weights_directory + file_name + extension)
    model.save(model_directory + file_name + extension)


def _get_model(model):
    if type(model) == str:
        return load_model(model)
    elif type(model) == Sequential:
        return model


def _calc_steps_per_epoch(samples: int, batch_size: int):
    return math.ceil(samples / batch_size)


def filters_dropout_compensation(min_active_filters: int, dropout: float):
    """
    Calculates the minimum number of active filters, adjusted for dropout. For example, if 50 active filters are
    required and a dropout value of .2 is used, the value 63 will be returned. (When the 63 filters are set, 20% will
    be turned off due to dropout).
    :param min_active_filters: The minimum number of active filters
    :param dropout: A dropout value
    :return: Returns a greater filter value to ensure the minimum number of filters are always active.
    """
    if dropout != 1:
        return int(math.ceil(min_active_filters / (1 - dropout)))
    else:
        return 0


def class_label(labels_directory: str, class_index: int):
    return os.listdir(labels_directory)[class_index]


def get_predictions(model, img_directory: str, train_directory: str, width: int, height: int):
    # if model is file path string, load keras model from file path
    print('directory:', img_directory)
    model = _get_model(model)
    predictions = []

    # run predictions on model
    images = []
    files = file_util.get_files(img_directory)

    for file in files:
        img = image.load_img(os.path.join(img_directory, file), target_size=(width, height))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        images.append(img)

    images = np.vstack(images)
    # predictions = model.predict(images)
    class_indices = model.predict_classes(images)
    i = 0
    for class_index in class_indices:
        result = class_label(train_directory, class_index)
        predictions.append(result)
        print(files[i] + ': predicted ' + result.upper() + ' (' + str(class_index) + ')')
        i += 1

    return predictions


def model_metrics(model, test_directory: str, img_width: int, img_height: int):
    model = _get_model(model)

    test_generator = ImageDataGenerator()
    test_data_generator = test_generator.flow_from_directory(
        test_directory,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode=None,
        shuffle=False)
    # test_steps_per_epoch = np.math.ceil(test_data_generator.samples / test_data_generator.batch_size)
    # test_steps_per_epoch = np.math.ceil(test_data_generator.samples / test_data_generator.batch_size)
    test_steps_per_epoch = _calc_steps_per_epoch(test_data_generator.samples, test_data_generator.batch_size)

    predictions = model.predict_generator(test_data_generator, steps=test_steps_per_epoch)
    # Get most likely class
    predicted_classes = np.argmax(predictions, axis=1)

    true_classes = test_data_generator.classes
    class_labels = list(test_data_generator.class_indices.keys())

    report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report)

    return report


def show_plot(history, file_name: str = None):
    accuracy_directory = 'graphs/accuracy/'
    loss_directory = 'graphs/loss/'
    extension = '.png'

    fig_size = [8, 6]
    line_width = 2.0
    font_size = 16
    legend_font_size = 18

    # Plot the Loss Curves
    plt.figure(figsize=fig_size)
    plt.plot(history.history['loss'], 'r', linewidth=line_width)
    plt.plot(history.history['val_loss'], 'b', linewidth=line_width)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=legend_font_size)
    plt.xlabel('Epochs ', fontsize=font_size)
    plt.ylabel('Loss', fontsize=font_size)
    plt.title('Loss Curves', fontsize=font_size)

    if file_name is not None:
        plt.savefig(loss_directory + 'loss-' + file_name + extension)
    plt.show()

    # Plot the Accuracy Curves
    plt.figure(figsize=fig_size)
    plt.plot(history.history['acc'], 'r', linewidth=line_width)
    plt.plot(history.history['val_acc'], 'b', linewidth=line_width)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=legend_font_size)
    plt.xlabel('Epochs ', fontsize=font_size)
    plt.ylabel('Accuracy', fontsize=font_size)
    plt.title('Accuracy Curves', fontsize=font_size)

    if file_name is not None:
        plt.savefig(accuracy_directory + 'accuracy-' + file_name + extension)
    plt.show()


def train(train_directory: str, validate_directory: str, img_width: int, img_height: int, save: bool = True,
          show: bool = True, use_pre_processing: bool = False, pre_processing_directory: str = None):
    # check preprocessing save directory is to be used
    if pre_processing_directory is not None:
        file_util.empty_directory(pre_processing_directory)

    rescale = 1. / 255
    shear_range = None
    zoom_range = None
    horizontal_flip = False
    fill_mode = None

    if use_pre_processing:
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode = 'nearest',

    num_classes = file_util.count_subdirectories(train_directory)
    class_mode = 'categorical'

    batch_size = 32
    epochs = 20

    channels = 3
    kernel_size = (3, 3)
    pool_size = (2, 2)

    dropout = .4

    conv_2d_layers = 4
    conv_2d_filters = 32

    # ensure minimum number of active filters (adjust filters for dropout)
    # conv_2d_filters = filters_dropout_compensation(conv_2d_filters, dropout)
    print('conv_2d_filters:', conv_2d_filters)

    dense_filters = 512

    # ensure minimum number of active filters (adjust filters for dropout)
    # dense_filters = filters_dropout_compensation(dense_filters, dropout)
    print('dense_filters:', dense_filters, end='\n\n')

    # set input shape
    if kb.image_data_format() == 'channels_first':
        input_shape = (channels, img_width, img_height)
    else:
        input_shape = (img_width, img_height, channels)

    model = Sequential()

    # input layer
    model.add(Conv2D(conv_2d_filters, kernel_size, activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=pool_size))

    # add conv2d layers
    for i in range(conv_2d_layers):
        model.add(Conv2D(conv_2d_filters, kernel_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        # conv_2d_filters *= 2

    # add flatten and dense layers
    model.add(Flatten())
    model.add(Dense(dense_filters, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=rescale,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        fill_mode=fill_mode,
    )

    validate_datagen = ImageDataGenerator(rescale=rescale)

    train_generator = train_datagen.flow_from_directory(
        train_directory,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=class_mode,
        save_to_dir=pre_processing_directory)

    validation_generator = validate_datagen.flow_from_directory(
        validate_directory,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=class_mode)

    steps_per_epoch = _calc_steps_per_epoch(train_generator.samples, batch_size)
    validation_steps = _calc_steps_per_epoch(validation_generator.samples, batch_size)

    # train
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps)

    # file naming strings
    now_string = file_util.date_string_now()

    info_string = str(img_width) + 'x' + str(img_height) + '_' + str(epochs) + '-e_' + str(
        conv_2d_layers) + '-c2dl_' + str(conv_2d_filters) + '-f_' + str(dropout) + '-d'

    file_string = now_string + '_' + info_string

    # show data
    if show:
        show_plot(history, file_string)

    # save data
    if save:
        save_model(model, file_string)

    return model


def run():
    train_directory = 'datasets/pokemon/train'
    validate_directory = 'datasets/pokemon/validate'
    use_pre_processing = True
    pre_processing_directory = 'img_preprocessing'
    # pre_processing_directory = None

    img_width = 100
    img_height = img_width

    model = train(train_directory, validate_directory, img_width, img_height,
                  use_pre_processing=use_pre_processing, pre_processing_directory=pre_processing_directory)
    get_predictions(model, 'examples', train_directory, img_width, img_height)
    model_metrics(model, validate_directory, img_width, img_height)


run()
# print(get_predictions('C:/Users\colom\PycharmProjects\pokemon-repo\keras_model/2018-12-04 08-20-28.372753_100x100_'
#                       + '20-epochs_4-inner_layers_32-filters_0.4-dropout.h5', 'examples', 'datasets/pokemon/train',
#                       100, 100))
