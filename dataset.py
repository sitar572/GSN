import os
import random
import shutil
from PIL import Image, ImageOps
from distutils.dir_util import copy_tree


class Params:
    # Define truncated dataset params
    number_of_classes = 200
    trn_dataset_size = 200

    # Define full CASIA dataset paths
    main_path = "F:/CASIA_data/raw"
    name_trn_P1 = "HWDB1.1trn_gnt_P1"
    name_trn_P2 = "HWDB1.1trn_gnt_P2"
    name_trn_preprocessed_P1 = "HWDB1.1trn_gnt_preprocessed_P1"
    name_trn_preprocessed_P2 = "HWDB1.1trn_gnt_preprocessed_P2"
    # name_trn_preprocessed_P1 = "HWDB1.1trn_gnt_P1"
    # name_trn_preprocessed_P2 = "HWDB1.1trn_gnt_P2"
    name_tst = "HWDB1.1tst_gnt"
    name_preprocessed_tst = "HWDB1.1tst_gnt_preprocessed"
    # name_preprocessed_tst = "HWDB1.1tst_gnt"
    name_trc = "HWDB1.1trc_gnt_" + str(number_of_classes)

    # Join paths
    path_trn_P1 = os.path.join(main_path, name_trn_P1)
    path_trn_P2 = os.path.join(main_path, name_trn_P2)
    path_trn_preprocessed_P1 = os.path.join(
        main_path, name_trn_preprocessed_P1)
    path_trn_preprocessed_P2 = os.path.join(
        main_path, name_trn_preprocessed_P2)
    path_tst = os.path.join(main_path, name_tst)
    path_tst_preprocessed = os.path.join(
        main_path, name_preprocessed_tst)
    path_trc = os.path.join(main_path, name_trc)

    # !!! WARNING !!!
    # Directory "path_trc" will be removed,
    # if it already exists while running the app.


def process_image(path_img_src, path_img_dst):
    img = Image.open(path_img_src)
    img = ImageOps.grayscale(img)
    img = ImageOps.equalize(img)
    path_img_dst = path_img_dst.replace(".jpg", ".bmp")
    img.save(path_img_dst)


def process_batch_images(path_src, path_dst, batch_name):
    dirs = os.listdir(path_src)
    for index, char in enumerate(dirs):
        print('Processing character: ' + char + ", batch " + batch_name + " progress: " +
              str(int(index / len(dirs) * 100)) + "%")

        path_char = os.path.join(path_src, char)
        path_preprocessed_char = os.path.join(path_dst, char)
        os.mkdir(path_preprocessed_char)
        char_dir = os.listdir(path_char)
        for img in char_dir:
            process_image(os.path.join(path_char, img),
                          os.path.join(path_preprocessed_char, img))


def make_preprocessed_dataset():
    # Create folder to store processed dataset
    dirs = os.listdir(Params.main_path)

    # Process P1
    if Params.name_trn_preprocessed_P1 not in dirs:
        os.mkdir(Params.path_trn_preprocessed_P1)
        process_batch_images(
            Params.path_trn_P1, Params.path_trn_preprocessed_P1, Params.name_trn_P1)

    # Process P2
    if Params.name_trn_preprocessed_P2 not in dirs:
        os.mkdir(Params.path_trn_preprocessed_P2)
        process_batch_images(
            Params.path_trn_P2, Params.path_trn_preprocessed_P2, Params.name_trn_P2)

    # Process tst
    if Params.name_preprocessed_tst not in dirs:
        os.mkdir(Params.path_tst_preprocessed)
        process_batch_images(
            Params.path_tst, Params.path_tst_preprocessed, Params.name_tst)


def copy_class_imgs(path_class_imgs, path_trn_char, path_val_char, img_prefix):
    img_list = os.listdir(path_class_imgs)
    random_img_numbers = random.sample(
        range(len(img_list)), int(Params.trn_dataset_size / 2))

    # Copy images to new folders
    for img_number, img in enumerate(img_list):
        if img_number in random_img_numbers:
            shutil.copyfile(os.path.join(path_class_imgs, img),
                            os.path.join(path_trn_char, img_prefix + img))
        else:
            shutil.copyfile(os.path.join(path_class_imgs, img),
                            os.path.join(path_val_char, img_prefix + img))


def make_subsets():
    # Create folder to store truncated dataset
    dirs = os.listdir(Params.main_path)
    if Params.name_trc in dirs:
        shutil.rmtree(Params.path_trc)
    os.mkdir(Params.path_trc)

    # Choose random classes
    dirs = os.listdir(Params.path_trn_preprocessed_P1)
    random_class_numbers = random.sample(
        range(len(dirs)), Params.number_of_classes)

    # Choose random chars for train and validation sets,
    # then copy them to the truncated directory.

    # Create folders for subsets
    path_trn_subset = os.path.join(Params.path_trc, "train")
    path_val_subset = os.path.join(Params.path_trc, "validate")
    path_tst_subset = os.path.join(Params.path_trc, "test")

    os.mkdir(path_trn_subset)
    os.mkdir(path_val_subset)
    os.mkdir(path_tst_subset)

    for index, class_number in enumerate(random_class_numbers):
        print('Choosen character ' + str(index + 1) + ": " +
              dirs[class_number] + ", progress: " +
              str(int(index / len(random_class_numbers) * 100)) + "%")

        # Create folder for class
        path_trn_char = os.path.join(path_trn_subset, dirs[class_number])
        path_val_char = os.path.join(path_val_subset, dirs[class_number])
        path_tst_char = os.path.join(path_tst_subset, dirs[class_number])

        os.mkdir(path_trn_char)
        os.mkdir(path_val_char)
        os.mkdir(path_tst_char)

        # Copy test set
        path_tst_class_to_copy = os.path.join(Params.path_tst_preprocessed,
                                              dirs[class_number])
        copy_tree(path_tst_class_to_copy, path_tst_char)

        # Choose and copy characters within class from P1
        path_class_imgs = os.path.join(
            Params.path_trn_preprocessed_P1, dirs[class_number])
        copy_class_imgs(path_class_imgs=path_class_imgs,
                        path_trn_char=path_trn_char,
                        path_val_char=path_val_char,
                        img_prefix="P1_")

        # Choose and copy characters within class from P2
        path_class_imgs = os.path.join(
            Params.path_trn_preprocessed_P2, dirs[class_number])
        copy_class_imgs(path_class_imgs=path_class_imgs,
                        path_trn_char=path_trn_char,
                        path_val_char=path_val_char,
                        img_prefix="P2_")


def main():
    make_preprocessed_dataset()
    make_subsets()


if __name__ == '__main__':
    main()
