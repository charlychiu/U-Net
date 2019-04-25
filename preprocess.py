from os import listdir
from os.path import join, isfile
from PIL import Image, ImageSequence
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = './dataset/'


def load_from_multi_page_tiff(file_path):
    """
    Handle multiPage TIFF file, and convert to numpy array
    :param file_path:
    :return: np_array
    """
    image_array_list = []
    image_set = Image.open(DATA_PATH + file_path)
    for idx, page_image in enumerate(ImageSequence.Iterator(image_set)):
        image_array_list.append(np.array(page_image))
    return np.stack(image_array_list)


def load_from_single_page_tiff(file_path):
    return np.array(Image.open(file_path))


def image_preview(np_array):
    img = Image.fromarray(np_array)
    img.show()


def image_preview_with_scale(np_array):
    plt.imshow(np_array, cmap="gray")
    plt.clim(np_array.min(), np_array.max())
    plt.colorbar()
    plt.show()


def check_image_shape(np_array):
    if isinstance(np_array, (np.ndarray, np.generic)):
        print("Loaded data shape: {}".format(np_array.shape))
    else:
        print('Input type error!')


def dataset_one():
    X_image_array = load_from_multi_page_tiff(file_path='ISBI2012/train-volume.tif')
    check_image_shape(X_image_array)
    Y_image_array = load_from_multi_page_tiff(file_path='ISBI2012/train-labels.tif')
    check_image_shape(Y_image_array)
    return X_image_array, Y_image_array


def seek_file(data_path):
    return [join(DATA_PATH + data_path, f) for f in listdir(DATA_PATH + data_path) if
            isfile(join(DATA_PATH + data_path, f))]


def dataset_two():
    get_data_list = seek_file('CellTracking/PhC-C2DH-U373/01')
    input_data = []
    for data in get_data_list:
        input_data.append(load_from_single_page_tiff(data))
    print(len(input_data))
    get_data_groud_truth_list = seek_file('CellTracking/PhC-C2DH-U373/01_GT')
    ground_truth_data = []
    for data_gt in get_data_groud_truth_list:
        ground_truth_data.append(load_from_single_page_tiff(data_gt))
    print(len(ground_truth_data))
    # image_preview_with_scale(ground_truth_data[1])


if __name__ == '__main__':
    # X, Y = dataset_one()
    dataset_two()
