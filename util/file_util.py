import os
from os.path import isfile, join
import random


def move_random_files(source_path: str, destination_path: str, percentage):
    if 0 <= percentage <= 1:
        file_names = [f for f in os.listdir(source_path) if isfile(join(source_path, f))]
        print(len(file_names))

        for i in range(int(len(file_names) * percentage)):
            random_index = random.randint(0, len(file_names))
            file_name = file_names[random_index]
            os.rename(source_path + '/' + file_name, destination_path + '/' + file_name)
            file_names.pop(random_index)

        print(len(file_names))


def clean_images():
    pass
