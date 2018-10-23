from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers import Dense

#initialize the CNN
classifier = Sequential()

#Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

#Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#Flattening
classifier.add(Flatten())

#Full Connection
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid'))

#Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Part 2 - Fitting the CNN to the image
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('datasets/pikachu', target_size=(64, 64), batch_size=32, class_mode='binary')

test_set= test_datagen.flow_from_directory('datasets/pikachu', target_size=(64, 64), batch_size=32, class_mode='binary')

#Training the network
from IPython.display import display
from PIL import Image

classifier.fit_generator(training_set, steps_per_epoch=500, epochs=1, validation_data=test_set, validation_steps=800)

#Testing
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('025Pikachu_Dream_2.png', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] >= 0.5:
    prediction = 'pikachu'
else:
    prediction = 'other'
print("answer ", prediction)



