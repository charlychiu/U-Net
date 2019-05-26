from os import listdir
from os.path import join, isfile
from PIL import Image, ImageSequence
import matplotlib.pyplot as plt
import numpy as np
from utils import *

DATA_PATH = './dataset/'


def load_from_multi_page_tiff(path_to_image):
    """
    Handle multiPage TIFF file, and convert to numpy array
    :param path_to_image:
    :return: np_array
    """
    image_np_array_list = []
    image_with_multi_page = Image.open(path_to_image)
    for idx, page_image in enumerate(ImageSequence.Iterator(image_with_multi_page)):
        image_np_array_list.append(np.array(page_image))
    return np.stack(image_np_array_list)


def load_from_single_page_tiff(path_to_image):
    """
    Convert image to array
    :param path_to_image:
    :return: np_array
    """
    return np.array(Image.open(path_to_image))


def image_preview(np_array):
    """
    Preview image from np_array
    :param np_array:
    """
    img = Image.fromarray(np_array)
    img.show()


def image_preview_fit_its_scale(np_array):
    """
    Preview image with corresponding scale
    :param np_array:
    """
    plt.imshow(np_array, cmap="gray")
    plt.clim(np_array.min(), np_array.max())
    plt.colorbar()
    plt.show()


def get_image_shape(np_array):
    if isinstance(np_array, (np.ndarray, np.generic)):
        print("Loaded data shape: {}".format(np_array.shape))
    else:
        print('Input type error!')


def get_ISBI_2012_dataset():
    """
    Loading from ISBI dataset and convert to two image array (raw_data & ground_truth)
    :return: np_array, np_array
    """
    x_image_array = load_from_multi_page_tiff(path_to_image=DATA_PATH + 'ISBI2012/train-volume.tif')
    y_image_array = load_from_multi_page_tiff(path_to_image=DATA_PATH + 'ISBI2012/train-labels.tif')
    get_image_shape(x_image_array)
    return x_image_array, y_image_array


def seek_file_in_folder(folder_path):
    return [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f))]


def get_DIC_C2DH_HeLa():
    """
    get DIC C2DH HeLa dataset where had ground truth data only
    :return: x_np_array, y_np_array
    """
    folder_path_list = ['CellTracking/PhC-C2DH-U373/01_GT', 'CellTracking/PhC-C2DH-U373/02_GT']
    x, y = parsing_cell_tracking_data(folder_path_list)
    get_image_shape(x)
    return x, y


def get_PhC_C2DH_U373():
    """
    get PhC C2DH U373 dataset where had ground truth data only
    :return: x_np_array, y_np_array
    """
    folder_path_list = ['CellTracking/DIC-C2DH-HeLa/01_GT', 'CellTracking/DIC-C2DH-HeLa/02_GT']
    x, y = parsing_cell_tracking_data(folder_path_list)
    get_image_shape(x)
    return x, y


def parsing_cell_tracking_data(ground_truth_path):
    """
    For parsing cell tracking data folder structure only
    :param ground_truth_path:
    :return: x_np_array, y_np_array
    """
    x_image_array = []
    y_image_array = []
    for folder in ground_truth_path:
        image_name_list = seek_file_in_folder(DATA_PATH + folder)
        seg_image_dict = dict()
        for image_name in image_name_list:
            seg_image_dict[image_name[-7:]] = load_from_single_page_tiff(image_name)
        image_name_list_1 = seek_file_in_folder(DATA_PATH + folder[:-3])
        x_image = []
        y_image = []
        for img_name, img_array in seg_image_dict.items():
            matching = [s for s in image_name_list_1 if img_name in s]
            x_image.append(load_from_single_page_tiff(matching[0]))
            y_image.append(img_array)
        x_image_array += x_image
        y_image_array += y_image
    return np.stack(x_image_array), np.stack(y_image_array)


def overlap_tile_processing(img_array, expend_px_width, expend_px_height):
    """
    Following U-Net paper 'Overlap-tile strategy' processing image
    :param img_array: input image array
    :param expend_px_width: per edge expend width ex. 512*512 => 512*(512+(92*2))
    :param expend_px_height: per edge expend height ex. 512*512 => (512+(92*2))*512
    :return: processed image array
    """
    import cv2

    def flip_horizontally(np_array):
        return cv2.flip(np_array, 1)

    def flip_vertically(np_array):
        return cv2.flip(np_array, 0)

    original_height = img_array.shape[0]
    original_width = img_array.shape[1]

    # Expand width first
    # left:
    left_result = flip_horizontally(img_array[0:0 + original_height, 0:0 + expend_px_width])
    # right:
    right_result = flip_horizontally(
        img_array[0:0 + original_height, original_width - expend_px_width: original_width])

    result_img = cv2.hconcat([left_result, img_array])
    result_img = cv2.hconcat([result_img, right_result])

    result_img_height = result_img.shape[0]
    result_img_width = result_img.shape[1]

    # Expand height
    top_result = flip_vertically(result_img[0:0 + expend_px_height, 0:0 + result_img_width])
    bottom_result = flip_vertically(
        result_img[result_img_height - expend_px_height: result_img_height, 0:0 + result_img_width])

    result_img = cv2.vconcat([top_result, result_img])
    result_img = cv2.vconcat([result_img, bottom_result])

    return result_img


if __name__ == '__main__':
    X, Y = get_ISBI_2012_dataset()
    X1, Y1 = get_DIC_C2DH_HeLa()
    X2, Y2 = get_PhC_C2DH_U373()
    image_preview_fit_its_scale(X[0])
    # image_preview_fit_its_scale(Y[0])
    image_preview_fit_its_scale(overlap_tile_processing(X[0], 92, 92))
    # import cv2
    # cv2.imwrite('test.png', X[0])
