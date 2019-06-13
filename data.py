from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import cv2
import elastic_transform as ET


No_img = 50         # number of augmentated image per set = No_img + 1

class DataAug():
    """
    This class generate data by using ImageDataGenerator in Keras
    It also generates a separated file for each picture
    Loading data from 'train_path' and 'label_path' with separated tif
    """

    def __init__(self,
                 train_path="data1/train",
                 label_path="data1/label",
                 merge_path="data1/merge",
                 aug_merge_path="data1/aug_merge",
                 aug_train_path="data1/aug_train",
                 aug_label_path="data1/aug_label",
                 img_type="tif"):

        # get all picutre under path with specific file extension
        self.train_imgs = glob.glob(train_path + "/*." + img_type)
        self.label_imgs = glob.glob(label_path + "/*." + img_type)

        self.train_path = train_path
        self.label_path = label_path
        self.merge_path = merge_path
        self.img_type = img_type
        self.aug_merge_path = aug_merge_path
        self.aug_train_path = aug_train_path
        self.aug_label_path = aug_label_path
        self.slices = len(self.train_imgs)

        # image data generator parameter
        self.datagen = ImageDataGenerator(
            rotation_range=0.2,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.08,
            zoom_range=0.08,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest')

        self.create_dir(self.merge_path)
        self.create_dir(self.aug_merge_path)
        self.create_dir(self.aug_train_path)
        self.create_dir(self.aug_label_path)
    
    def create_dir(self, path):
        if not os.path.lexists(path):
                os.mkdir(path)

    def augmentation(self):

        if len(self.train_imgs) != len(self.label_imgs) or len(self.train_imgs) == 0 or len(self.label_imgs) == 0:
            print("trains can't match labels")
            return 0

        print("len of trains: ", len(self.train_imgs))

        for num_of_picture in range(len(self.train_imgs)):
            img_t = load_img(self.train_path + "/" + str(num_of_picture) + "." + self.img_type)
            img_l = load_img(self.label_path + "/" + str(num_of_picture) + "." + self.img_type)
            x_t = img_to_array(img_t)
            x_l = img_to_array(img_l)

            # Merge image
            x_t[:, :, 2] = x_l[:, :, 0]     # last channel of x_t is label --> x_t is called merged img

            img_tmp = array_to_img(x_t)
            img_tmp.save(self.merge_path + "/" + str(num_of_picture) + "." + self.img_type)
            img = x_t
            img = img.reshape((1,) + img.shape)

            savedir = self.aug_merge_path + "/" + str(num_of_picture)
            self.create_dir(savedir)
            
            print("Doing augmentation at picture: ", str(num_of_picture))
            self.do_augmentation(img, savedir, str(num_of_picture), 1, self.img_type)
        
        self.split_merge_image()

    def do_augmentation(self,
                        img,
                        save_to_dir,
                        save_prefix,
                        batch_size=1,
                        save_format='tif',
                        imgnum=No_img):
        """
        Do Augmentation of one image
        """

        counter4No_img = 0
        for _ in self.datagen.flow(
                img,
                batch_size=batch_size,
                save_to_dir=save_to_dir,
                save_prefix=save_prefix,
                save_format=save_format):

            counter4No_img += 1
            if counter4No_img > imgnum:
                break
    
    def split_merge_image(self):
        """
		Split merged image apart
		"""

        for num_of_picture in range(self.slices):        # 30 images
            path = self.aug_merge_path + "/" + str(num_of_picture)
            train_imgs = glob.glob(path + "/*." + self.img_type)    # add subfolder 0 --> 29

            savedir = self.aug_train_path + "/" + str(num_of_picture)
            self.create_dir(savedir)

            savedir = self.aug_label_path + "/" + str(num_of_picture)
            self.create_dir(savedir)

            print("len of split: ", len(train_imgs))

            for imgname in train_imgs:

                ## For windows usage --> \\
                ## For unix usage --> /
                midname = imgname[imgname.rindex("\\") + 1:imgname.rindex(
                    "." + self.img_type)]
                img = cv2.imread(imgname)
                img_train = img[:, :, 2]  #cv2 read image rgb->bgr
                img_label = img[:, :, 0]
            
                cv2.imwrite(self.aug_train_path + "/" + str(num_of_picture) + "/" + midname + "_train"
                            + "." + self.img_type, img_train)
                cv2.imwrite(self.aug_label_path + "/" + str(num_of_picture) + "/" + midname + "_label"
                            + "." + self.img_type, img_label)


class DataProcess():
    def __init__(self,
                 out_rows=512,
                 out_cols=512,
                 data_path="data1/aug_train",
                 label_path="data1/aug_label",
                 test_path="data1/test",
                 npy_path="data1/dataset",
                 img_type="tif",
                 img_No_train=0,
                 img_No_val=0,
                 extra_padding=184):

        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path
        self.label_path = label_path
        self.img_type = img_type
        self.test_path = test_path
        self.npy_path = npy_path
        self.img_No_train = (No_img + 1) * 25
        self.img_No_val = (No_img + 1) * 30 - self.img_No_train
        self.extra_padding = extra_padding

    def input_filled_mirroring(self, x, e = 92):      # Overlap-tile strategy
        w, h = np.shape(x)[0], np.shape(x)[1]
        y = np.zeros((h + e * 2, w + e * 2))
        y[e:h + e, e:w + e] = x
        y[e:e + h, 0:e] = np.flip(y[e:e + h, e:2 * e], 1)  # flip vertically
        y[e:e + h, e + w:2 * e + w] = np.flip(y[e:e + h, w:e + w], 1)  # flip vertically
        y[0:e, 0:2 * e + w] = np.flip(y[e:2 * e, 0:2 * e + w], 0)  # flip horizontally
        y[e + h:2 * e + h, 0:2 * e + w] = np.flip(y[h:e + h, 0:2 * e + w], 0)  # flip horizontally
        return y

    def create_training_data(self):
        extra = self.extra_padding
        print('-' * 30)
        print('Creating training images...')
        print('-' * 30)

        ET_params = np.array([[2, 0.08, 0.08], [2, 0.05, 0.05], [3, 0.07, 0.09], [3, 0.12, 0.07]]) * self.out_cols
        len_scaled = len(ET_params) + 1

        imgdatas = np.ndarray(
            (self.img_No_train*len_scaled, 1, self.out_rows+extra, self.out_cols+extra), dtype=np.uint8)
        imglabels = np.ndarray(
            (self.img_No_train*len_scaled, 1, self.out_rows, self.out_cols), dtype=np.uint8)

        imgdatas_val = np.ndarray(
            (self.img_No_val*len_scaled, 1, self.out_rows + extra, self.out_cols + extra), dtype=np.uint8)
        imglabels_val = np.ndarray(
            (self.img_No_val*len_scaled, 1, self.out_rows, self.out_cols), dtype=np.uint8)

        index = 0
        import time
        start = time.time()

        for num_of_picture in range(30):
            train_foldername = self.data_path + "/" + str(num_of_picture)
            label_foldername = self.label_path + "/" + str(num_of_picture)
            imgs = glob.glob(train_foldername + "/*." + self.img_type)

            for imgname in imgs:
                # print "imgname: ", imgname
                midname = imgname[imgname.rindex("\\") + 1:]
                img_name_only = midname[0:midname.rindex("_")]

                train_img_path = train_foldername + "/" + img_name_only + "_train." + self.img_type
                label_img_path = label_foldername + "/" + img_name_only + "_label." + self.img_type
                img = load_img(train_img_path, grayscale=True)
                label = load_img(label_img_path, grayscale=True)

                img = np.array(img)         # size of 512x512
                label = np.array(label)     # size of 512x512

                #  Doing elastic transform 
                im_merge = np.concatenate((img[..., None], label[..., None]), axis=2)
            
                for k in range(len(ET_params) + 1):
                    if k > 0:   # index 0 is for the original image
                        im_merge_t = ET.elastic_transform(im_merge, ET_params[k-1,0], ET_params[k-1,1],ET_params[k-1,2])
                        # Split image and mask
                        img = im_merge_t[..., 0]
                        label = im_merge_t[..., 1]

                    # original code for only 1 image augmentation
                    img = self.input_filled_mirroring(img)
                    img = np.expand_dims(img,0)
                    label = np.expand_dims(label,0)
                    if index < self.img_No_train*len_scaled:
                        imglabels[index] = label
                        imgdatas[index] = img
                    else:
                        imglabels_val[index-self.img_No_train*len_scaled] = label # save validation data
                        imgdatas_val[index-self.img_No_train*len_scaled] = img
                    index += 1
                    # print("index: ", index)
                    if (index + 1) % 10 == 0: print("Processed: %d/%d...Time passed: %.5f mins" % (index + 1,
                            self.img_No_train*len_scaled + self.img_No_train*len_scaled, (time.time() - start)/60.0))

        print('loading done')
        print('Start Saving processing....')
        np.save(self.npy_path + '/imgs_train.npy', imgdatas)
        np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
        np.save(self.npy_path + '/imgs_val.npy', imgdatas_val)
        np.save(self.npy_path + '/imgs_mask_val.npy', imglabels_val)
        print('Saving to .npy files done.')


def image_preview(data):
    plt.imshow(data)
    plt.colorbar()

def image_save(data, filename):
    plt.imshow(data)
    plt.colorbar()
    plt.savefig(filename)

def augmentation(trains, labels):
    """
    Doing data augmetation using keras.preprocessing.image.ImageDataGenerator
    """

    assert(len(trains) == len(labels)), "trains can not match labels"

    print('Total trains: ', len(trains))

    datagen = ImageDataGenerator(
            rotation_range=0.2,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.08,
            zoom_range=0.08,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest')

    seed = 1
    batch_size = 1
    epoch = 50
    datagen_generator = datagen.flow(trains, labels, batch_size=batch_size, seed=seed)

    tmp_x = list()
    tmp_y = list()
    i = 0
    for batch_x, batch_y in datagen_generator:
        tmp_x += list(batch_x)
        tmp_y += list(batch_y)
        i += 1
        if i >= epoch:
            return np.array(tmp_x), np.array(tmp_y)

def split_image2_4patch(x, y):
    processing_set = zip(x, y)
    tmp_x = list()
    tmp_y = list()
    for (image, mask) in processing_set:
        tmp_x.append(image[0:572, 0:572])
        tmp_y.append(mask[0:388, 0:388])
        tmp_x.append(image[0:572, 124:696])
        tmp_y.append(mask[0:388, 124:512])
        tmp_x.append(image[124:696, 0:572])
        tmp_y.append(mask[124:512, 0:388])
        tmp_x.append(image[124:696, 124:696])
        tmp_y.append(mask[124:512, 124:512])
    return np.array(tmp_x), np.array(tmp_y)

if __name__ == '__main__':

    # aug = DataAug()
    # aug.augmentation()

    # dp = DataProcess()
    # dp.create_training_data()

    trains = np.load('data1/dataset/imgs_val.npy')
    labels = np.load('data1/dataset/imgs_mask_val.npy')
    print('Trains shape: ', trains.shape)
    print('Labels shape: ', labels.shape)
    # image_save(trains[10][0,:,:], 'train.png')
    # image_save(labels[10][0,:,:], 'label.png')

    trains = trains[:,0,:,:]
    trains = trains.reshape(trains.shape + (1,))
    labels = labels[:,0,:,:]
    labels = labels.reshape(labels.shape + (1,))
    print('Trains shape: ', trains.shape)
    print('Labels shape: ', labels.shape)

    X, Y = split_image2_4patch(trains, labels)
    print('X shape: ', X.shape) # 572 x 572
    print('Y shape: ', Y.shape) # 388 x 388

    np.save('data1/dataset/imgs_val2.npy', X)
    np.save('data1/dataset/imgs_mask_val2.npy', Y)
