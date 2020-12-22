import os
import random
from shutil import copyfile

path_data = "./dataset"


def split_data(data_dir, size_data_train):
    if not (isinstance(data_dir, str)):
        raise AttributeError('data_dir must be a string')

    if not os.path.exists(data_dir):
        raise OSError("data_dir doesn't exist")

    if not (isinstance(size_data_train, float)):
        raise AttributeError('size_data_train must be a float')

    # Set up empty folder structure if not exists

    if not os.path.exists('training'):
        os.makedirs('training')
    if not os.path.exists('test'):
        os.makedirs('test')

    # Get the subdirectories in the main image folder
    subdirs = [subdir for subdir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, subdir))]

    for subdir in subdirs:
        subdir_fullpath = os.path.join(data_dir, subdir)
        if len(os.listdir(subdir_fullpath)) == 0:
            print(subdir_fullpath + ' is empty')
            break

        train_subdir = os.path.join('training', subdir)
        validation_subdir = os.path.join('test', subdir)

        # Create subdirectories in train and validation folders
        if not os.path.exists(train_subdir):
            os.makedirs(train_subdir)

        if not os.path.exists(validation_subdir):
            os.makedirs(validation_subdir)

        train_counter = 0
        validation_counter = 0

        # Randomly assign an image to train or validation folder
        for filename in os.listdir(subdir_fullpath):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                fileparts = filename.split('.')

                if random.uniform(0, 1) <= size_data_train:
                    copyfile(os.path.join(subdir_fullpath, filename),
                             os.path.join(train_subdir, str(train_counter) + '.' + fileparts[1]))
                    train_counter += 1
                else:
                    copyfile(os.path.join(subdir_fullpath, filename),
                             os.path.join(validation_subdir, str(validation_counter) + '.' + fileparts[1]))
                    validation_counter += 1

        print('Copied ' + str(train_counter) + ' images to data/train/' + subdir)
        print('Copied ' + str(validation_counter) + ' images to data/validation/' + subdir)


split_data(path_data, 0.5)
