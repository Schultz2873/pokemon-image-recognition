from PIL import Image
import os

# directory_path = 'datasets/original/images/pokemon-images'
directory_path = 'datasets/original/images/pokemon-a'

img_width = 256
img_height = 256


def resize_images(directory, width, height):
    files = os.listdir(directory)

    for i in range(0, len(files)):
        image_path = directory + '/' + files[i]
        image = Image.open(image_path)
        new_image = image.resize((width, height))
        new_image.save(image_path)


resize_images(directory_path, img_width, img_height)
