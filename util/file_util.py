import os
from os.path import isfile, join
import shutil
from shutil import copyfile
from PIL import Image
import random

# for naming generated files
import datetime


def empty_directory(directory):
    for file in os.listdir(directory):
        os.remove(os.path.join(directory, file))


def get_files_directory(directory_path):
    return [f for f in os.listdir(directory_path) if isfile(join(directory_path, f))]


def get_files_walk(directory):
    files = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))
    return files


def num_files(directory):
    total = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        total += len(filenames)
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

        print('splitting ' + source_path + ' with ' + str(percentage) + ' split value')

        # create directory if not exists
        if not os.path.exists(destination_path1):
            os.mkdir(destination_path1)

        # create directory if not exists
        if not os.path.exists(destination_path2):
            os.mkdir(destination_path2)

        # get file names from source path
        file_names = [f for f in os.listdir(source_path) if isfile(join(source_path, f))]
        destination1_file_names = []

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
        if os.path.isdir(os.path.join(directory, name)):
            count += 1
    return count


def date_string_now():
    now_string = str(datetime.datetime.now())
    now_string = now_string.replace(':', '-')

    return now_string


def resize_image(image_path, width, height):
    if isfile(image_path):
        image = Image.open(image_path)
        new_image = image.resize((width, height))
        new_image.save(image_path)


def resize_images(directory, width, height):
    file_paths = os.listdir(directory)

    for file_path in file_paths:
        resize_image(file_path, width, height)


def change_image_type(image_path: str, extension: str = None, overwrite: bool = True, handle_palette: bool = True):
    if os.path.isfile(image_path):
        # get file root and extension
        img_root, img_extension = os.path.splitext(image_path)

        if extension is not None:
            extension = '.' + extension
        else:
            extension = img_extension

        # if img_extension != extension:
        print('modifying file:', image_path)
        img = Image.open(image_path)
        print(img)

        is_mode_converted = False

        new_image_path = img_root + extension

        # if has transparency, convert to RGB
        if handle_palette and img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
            is_mode_converted = True

        if img_extension != extension or is_mode_converted:

            img.save(new_image_path)
            img.close()

            if overwrite:
                os.remove(image_path)

            return True
        img.close()

    return False


def directory_change_image_type(directory: str, extension: str, overwrite: bool = True):
    for directory_name, subdirectory_list, file_list in os.walk(directory):
        for file_name in file_list:
            change_image_type(directory_name + '/' + file_name, extension, overwrite)
    print('\nimage types changed')


def build_dataset(core_dataset_directory: str, base_directory: str, dataset_name: str, split_percentage: float,
                  image_extension: str = None, class_list: list = None, overwrite_existing: bool = False):
    # if core dataset path is a directory
    if os.path.isdir(core_dataset_directory):

        dataset_name_directory = os.path.join(base_directory, dataset_name)

        # if overwrite enabled, overwrite existing directory
        if overwrite_existing and os.path.isdir(dataset_name_directory):
            # delete directory
            shutil.rmtree(dataset_name_directory)

        # if dataset directory does not exist
        if not os.path.exists(dataset_name_directory):

            train_directory = os.path.join(dataset_name_directory, 'train')
            validate_directory = os.path.join(dataset_name_directory, 'validate')

            # create directories
            os.mkdir(dataset_name_directory)
            os.mkdir(train_directory)
            os.mkdir(validate_directory)

            classes = os.listdir(core_dataset_directory)

            # iterate over class directories in core dataset
            for class_name in classes:

                # if classlist exists and match found use class or use directory contents only
                if (class_list is not None and class_name in class_list) or class_list is None:
                    dataset_class_name_directory = os.path.join(core_dataset_directory, class_name)
                    class_train_directory = os.path.join(train_directory, class_name)
                    class_validate_directory = os.path.join(validate_directory, class_name)

                    # split contents of core dataset class folder into train and validate class folders
                    split_directory(dataset_class_name_directory, class_train_directory, class_validate_directory,
                                    split_percentage)

            if image_extension is not None:
                directory_change_image_type(base_directory, image_extension)


def create_train_validate():
    core_dataset_directory = 'C:/Users/colom/PycharmProjects/pokemon-repo/poke_dataset'
    base_directory = 'C:/Users/colom/PycharmProjects/pokemon-repo/datasets'
    dataset_name = 'pokemon'
    split_percentage = .8
    image_extension = 'jpg'
    class_list = ['bulbasaur', 'charmander', 'pikachu', 'squirtle']
    # class_list = None
    overwrite_existing = True

    build_dataset(core_dataset_directory, base_directory, dataset_name, split_percentage, image_extension,
                  class_list=class_list, overwrite_existing=overwrite_existing)


# create_train_validate()
