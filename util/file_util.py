import os
from os.path import isfile, join
from shutil import copyfile
from PIL import Image
import random

# for naming generated files
import datetime


def get_files(directory_path):
    return [f for f in os.listdir(directory_path) if isfile(join(directory_path, f))]


def num_files(directory):
    total = 0
    for root, dirs, files in os.walk(directory):
        total += len(files)
    return total


def split_directory(source_path: str, destination_path1: str, destination_path2: str, percentage: float,
                    is_random: bool = True, is_copy: bool = True):
    """
    Splits a target directory into two separate directories with the amount of files determined by a split percentage
    value.
    :param source_path: The target directory to be split.
    :param destination_path1: The destination directory. The percentage of files specified will be placed here.
    :param destination_path2: The second destination directory. Remainder files (1 - percentage) will be placed here.
    :param percentage: The amount of files to be placed in the directory determined by destination_path1
    :param is_random: If true, files will be randomly chosen, else chosen in order as appearing in directory.
    :param is_copy: If true, files will be copied, else files will be moved.
    :return:
    """
    if 0 <= percentage <= 1:

        if not os.path.exists(destination_path1):
            os.mkdir(destination_path1)

        if not os.path.exists(destination_path2):
            os.mkdir(destination_path2)

        # get file names from source path
        file_names = [f for f in os.listdir(source_path) if isfile(join(source_path, f))]
        destination1_file_names = []

        print(len(file_names))

        # iterate through files, popping selected files from file_names into destination1_file_names
        # file_names becomes remainder file list
        i = 0
        iterations = int(len(file_names) * percentage)
        while i < iterations:

            # if random, select random file, else select current index file
            if is_random:
                index = random.randint(0, len(file_names) - 1)
            else:
                index = 0

            # pop file from file_names into destination1_file_names
            destination1_file_names.append(file_names.pop(index))
            i += 1

        print(len(destination1_file_names))
        print(len(file_names))

        # copy or move files in destination1_file_names to destination_path1
        for i in range(len(destination1_file_names)):
            _copy_or_move(source_path, destination_path1, destination1_file_names, i, is_copy)

        # copy or move remainder files to destination_path2
        for i in range(len(file_names)):
            _copy_or_move(source_path, destination_path2, file_names, i, is_copy)


def _copy_or_move(source_path, destination_path, file_names, index, is_copy):
    source = source_path + '/' + file_names[index]
    destination = destination_path + '/' + file_names[index]

    if is_copy:
        copyfile(source, destination)
    else:
        os.rename(source, destination)


def count_subdirectories(directory):
    count = 0
    directory_list = os.listdir(directory)
    for name in directory_list:
        if os.path.isdir(directory + '/' + name):
            count += 1
    return count


def date_string_now():
    now_string = str(datetime.datetime.now())
    now_string = now_string.replace(':', '-')

    return now_string


def resize_images(directory, width, height):
    files = os.listdir(directory)

    for i in range(0, len(files)):
        image_path = directory + '/' + files[i]
        image = Image.open(image_path)
        new_image = image.resize((width, height))
        new_image.save(image_path)


def change_image_type(image_path: str, extension: str, overwrite: bool = True):
    if os.path.isfile(image_path):

        img = Image.open(image_path)

        # only change format if extension is different from file's extension
        if img.format != extension:
            period_index = image_path.index('.')

            # if has transparency, convert to RGB
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # save image
            img.save(image_path[:period_index + 1] + extension)

            # if overwrite enabled, delete old file
            if overwrite:
                os.remove(image_path)

            img.close()
            return True
        img.close()

    else:
        print(image_path + ' is not a file')
    return False


def directory_change_image_type(directory: str, extension: str, overwrite: bool = True):
    for directory_name, subdirectory_list, file_list in os.walk(directory):
        for file_name in file_list:
            change_image_type(directory_name + '/' + file_name, extension, overwrite)

# name = 'squirtle'
# split_directory('C:/Users/colom/PycharmProjects/pokemon-repo/poke_dataset/' + name,
#                 'C:/Users/colom/PycharmProjects/pokemon-repo/datasets/pokemon/train/' + name,
#                 'C:/Users/colom/PycharmProjects/pokemon-repo/datasets/pokemon/validate/' + name,
#                 .7)

# directory_change_image_type('C:/Users/colom/PycharmProjects/pokemon-repo/datasets', 'jpg')
