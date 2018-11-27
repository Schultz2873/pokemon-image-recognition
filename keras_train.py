from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
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


def show_predictions(model, images_path: str, width, height):
    # if model is file path string, load keras model from file path
    model = _get_model(model)

    # run predictions on model
    images = []
    files = file_util.get_files(images_path)
    print(files)

    for file in files:
        img = image.load_img(images_path + '/' + file, target_size=(width, height))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        images.append(img)

    images = np.vstack(images)
    predictions = model.predict(images)
    classes = model.predict_classes(images)
    print('predictions:\n', predictions)
    print('prediction indices:\n', classes)
    for class_value in classes:
        print(str(class_value) + ': ', end='')
        if class_value == 0:
            print('bulbasaur')
        elif class_value == 1:
            print('charmander')
        elif class_value == 2:
            print('mewtwo')
        elif class_value == 3:
            print('pikachu')
        elif class_value == 4:
            print('squirtle')


def model_metrics(model, test_directory: str, img_width: int, img_height: int):
    model = _get_model(model)

    test_generator = ImageDataGenerator()
    test_data_generator = test_generator.flow_from_directory(
        test_directory,
        target_size=(img_width, img_height),
        batch_size=16,
        shuffle=False)
    # test_steps_per_epoch = np.math.ceil(test_data_generator.samples / test_data_generator.batch_size)
    test_steps_per_epoch = np.math.ceil(test_data_generator.samples / test_data_generator.batch_size)

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
    line_width = 3.0
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


def train(epochs, img_width, img_height, save: bool = True, show: bool = True):
    training_directory = 'datasets/pokemon/train'
    validation_directory = 'datasets/pokemon/validate'

    num_classes = file_util.count_subdirectories(training_directory)
    class_mode = 'categorical'

    channels = 3
    kernel_size = (3, 3)

    batch_size = 16

    inner_layers = 2
    inner_layer_filters = 32
    dropout = .25

    if kb.image_data_format() == 'channels_first':
        input_shape = (channels, img_width, img_height)
    else:
        input_shape = (img_width, img_height, channels)

    pool_size = (2, 2)

    model = Sequential()
    # model.add(Conv2D(64, kernel_size, activation='relu', input_shape=input_shape))
    # model.add(Dropout(dropout))
    # model.add(MaxPooling2D(pool_size=pool_size))

    filters = inner_layer_filters
    for i in range(inner_layers):
        print(filters)
        # first layer
        if i == 0:
            model.add(Conv2D(filters, kernel_size, activation='relu', input_shape=input_shape))
        # other layers
        else:
            model.add(Conv2D(filters, kernel_size, activation='relu'))

        # model.add(Dropout(dropout))
        model.add(Conv2D(filters, kernel_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=pool_size))

        # filters *= 2

    model.add(Flatten())
    # model.add(Dropout(dropout))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=90,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.15,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest')

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        training_directory,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=class_mode)

    validation_generator = test_datagen.flow_from_directory(
        validation_directory,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=class_mode)

    steps_per_epoch = math.ceil(train_generator.samples / batch_size)
    validation_steps = math.ceil(validation_generator.samples / batch_size)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps)

    # file naming strings
    now_string = file_util.date_string_now()

    info_string = str(img_width) + 'x' + str(img_height) + '_' + str(epochs) + '-epochs_' + str(
        inner_layers) + '-inner_layers_' + str(inner_layer_filters) + '-filters_' + str(batch_size) + '-batch_size'

    file_string = now_string + '_' + info_string

    # show data
    if show:
        show_plot(history, file_string)

    # save data
    if save:
        save_model(model, file_string)

    return model


def run():
    epochs = 20
    img_width = 100
    img_height = img_width

    model = train(epochs, img_width, img_height)
    show_predictions(model, 'examples', img_width, img_height)
    model_metrics(model, 'datasets/pokemon/validate', img_width, img_height)


def evaluate(model, test_directory, img_width, img_height):
    if type(model) == str:
        model = load_model(model)

    model_metrics(model, test_directory, img_width, img_height)


run()
# evaluate('keras_model/2018-11-19 22-19-16.839323_100x100_30-epochs_3-inner_layers_50-filters_32-batch_size.h5',
#          'datasets/pokemon/validate', 100, 100)
