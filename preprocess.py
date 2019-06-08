from os import listdir
from os.path import join, isfile
from PIL import Image, ImageSequence
import matplotlib.pyplot as plt
import numpy as np
from utils import *
import cv2
from keras.preprocessing.image import ImageDataGenerator

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

def convert_to_datagen_format(py_list):
    nparray = np.array(py_list)
    return nparray.reshape(nparray.shape + (1,))

def data_generator(x, y, batch_size, epoch):
    '''
    Set same seed for image_datagen & mask_datagen to ensure the transformation for image and mask is the same
    '''
    seed = 1
    data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')
    
    datagen = ImageDataGenerator(**data_gen_args)

    datagen_generator = datagen.flow(x, y, batch_size=batch_size, seed=seed)

    tmp_x = list()
    tmp_y = list()
    i = 0
    for batch_x, batch_y in datagen_generator:
        tmp_x += list(batch_x)
        tmp_y += list(batch_y)
        i += 1
        if i >= epoch:
            return np.array(tmp_x), np.array(tmp_y)

if __name__ == '__main__':
    # In large scale data, do not save it as picture, cost large storage
    X, Y = get_ISBI_2012_dataset() # 30 pictures of microscope 

    # First: Over-tile strategy
    X_1 = list()
    Y_1 = list()
    for idx, picture in enumerate(X):
        X_1.append(overlap_tile_processing(picture, 92, 92))

    for idx, picture in enumerate(Y):
        Y_1.append(overlap_tile_processing(picture, 92, 92))

    X_1 = convert_to_datagen_format(X_1)
    Y_1 = convert_to_datagen_format(Y_1)
    assert(X_1.shape == (30, 696, 696, 1)), "data loading error"
    assert(Y_1.shape == (30, 696, 696, 1)), "data loading error"

    # Second: Data Generator
    
    X_2, Y_2 = data_generator(X_1, Y_1, 10, 10) ## total picture 10*10= 100
    print(X_2.shape)
    print(Y_2.shape)
    # image_preview(X_2[0][:,:,0])
    # image_preview(Y_2[0][:,:,0])

    # X1, Y1 = get_DIC_C2DH_HeLa()
    # save_npy_array_to_picture(X1, './data/data2/x/')
    # save_npy_array_to_picture(Y1, './data/data2/y/')
    # X2, Y2 = get_PhC_C2DH_U373()
    # save_npy_array_to_picture(X2, './data/data3/x/')
    # save_npy_array_to_picture(Y2, './data/data3/y/')

    # image_preview_fit_its_scale(X[0])
    # image_preview_fit_its_scale(overlap_tile_processing(X[0], 92, 92))

    # from PIL import Image
    #
    # im = Image.fromarray(X[0])
    # im.save("original.jpeg")
    # im = Image.fromarray(overlap_tile_processing(X[0], 92, 92))
    # im.save("augmentation.jpeg")


    # Step 1. Overlap-tile for all image(include ground-truth images)
    # processed_X = list()
    # for per_img in X:
    #     processed_X.append(overlap_tile_processing(per_img, 92, 92))
    # # processed_X = np.array(processed_X)
    #
    # processed_Y = list()
    # for per_img in Y:
    #     processed_Y.append(overlap_tile_processing(per_img, 92, 92))
    # # processed_Y = np.array(processed_Y)


    # print(processed_X.shape)
    # print(processed_Y.shape)
    # save_variable_to_pickle(np.array(processed_X), 'processed_x')
    # save_variable_to_pickle(np.array(processed_Y), 'processed_y')
    # a, b = keras_data_augmentation(processed_X, processed_Y)
    # print(a.shape)
    # print(b.shape)
