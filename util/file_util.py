import os
from os.path import isfile, join
from shutil import copyfile
from PIL import Image
import random

# for naming generated files
import datetime


def get_files(directory_path):
    return [f for f in os.listdir(directory_path) if isfile(join(directory_path, f))]


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

# split_directory('C:/Users/colom/PycharmProjects/pokemon-repo/poke_dataset/squirtle',
#                 'C:/Users/colom/PycharmProjects/pokemon-repo/datasets/pokemon/train/squirtle',
#                 'C:/Users/colom/PycharmProjects/pokemon-repo/datasets/pokemon/validate/squirtle',
#                 .7)
