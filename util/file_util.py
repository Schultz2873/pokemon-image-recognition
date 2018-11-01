import os
from os.path import isfile, join
from shutil import copyfile
import random


def move_random_files(source_path: str, destination_path: str, percentage):
    if 0 <= percentage <= 1:
        file_names = [f for f in os.listdir(source_path) if isfile(join(source_path, f))]

        for i in range(int(len(file_names) * percentage)):
            random_index = random.randint(0, len(file_names))
            file_name = file_names[random_index]
            os.rename(source_path + '/' + file_name, destination_path + '/' + file_name)
            file_names.pop(random_index)


def split_directory(source_path: str, destination_path1: str, destination_path2: str, percentage: float,
                    is_random: bool = True, is_copy: bool = True):
    if 0 <= percentage <= 1:
        # get file names from source path
        file_names = [f for f in os.listdir(source_path) if isfile(join(source_path, f))]
        destination1_file_names = []

        # iterate through files, popping selected files from file_names into destination1_file_names
        # file_names becomes remainder file list
        i = 0
        while i < len(file_names):

            # if random, select random file, else select current index file
            if is_random:
                index = random.randint(0, len(file_names) - 1)
            else:
                index = i

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
