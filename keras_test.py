from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as kb
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# for file management
import util.file_util as file_util

# for naming generated files
import datetime


def save_model(model, name: str = None):
    model_directory = 'keras_model/'
    weights_directory = 'keras_weights/'
    now_string = str(datetime.datetime.now())
    now_string = now_string.replace(':', '-')
    if name is None:
        model.save_weights(weights_directory + 'weights-' + now_string + '.h5')
        model.save(model_directory + 'model-' + now_string + '.h5')
    else:
        model.save_weights(weights_directory + name + '.h5')
        model.save(model_directory + name + '.h5')


def show_predictions(model_path: str, images_path: str, width, height):
    model: Sequential = load_model(model_path)
    model.summary()
    images = []
    files = file_util.get_files(images_path)
    print(files)

    for file in files:
        img = image.load_img(images_path + '/' + file, target_size=(width, height))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        images.append(img)

    images = np.vstack(images)
    print('predictions:\n', model.predict(images))
    print('prediction indices:\n', model.predict_classes(images))


def show_plot(history):
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
    plt.show()

    # Plot the Accuracy Curves
    plt.figure(figsize=fig_size)
    plt.plot(history.history['acc'], 'r', linewidth=line_width)
    plt.plot(history.history['val_acc'], 'b', linewidth=line_width)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=legend_font_size)
    plt.xlabel('Epochs ', fontsize=font_size)
    plt.ylabel('Accuracy', fontsize=font_size)
    plt.title('Accuracy Curves', fontsize=font_size)
    plt.show()


def test():
    training_directory = 'datasets/pokemon/train'
    validation_directory = 'datasets/pokemon/validate'

    num_classes = file_util.count_subdirectories(training_directory)
    class_mode = 'categorical'

    img_width, img_height = 50, 50
    channels = 3

    steps_per_epoch = 2000
    epochs = 15
    validation_steps = 800
    batch_size = 16

    if kb.image_data_format() == 'channels_first':
        input_shape = (channels, img_width, img_height)
    else:
        input_shape = (img_width, img_height, channels)

    model = Sequential()
    # add layers
    model.add(Conv2D(64, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # the ml_model so far outputs 3D feature maps (height, width, features)

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    # model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  # optimizer='adam',
                  metrics=['accuracy'])

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # this is a generator that will read pictures found in
    # sub-folders of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
        training_directory,  # this is the target directory
        target_size=(img_width, img_height),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode=class_mode)  # since we use binary_crossentropy loss, we need binary labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        validation_directory,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=class_mode)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps // batch_size)

    print(history.history.keys())
    show_plot(history)

    # model.save_weights(weights_directory + '/first_try.h5',
    #                    overwrite=True)  # always save your weights after training or during training

    # save the model's .h5 file
    save_model(model)


# test()
show_predictions('keras_model/model-2018-11-08 01-07-24.778151.h5', 'examples', 50, 50)
