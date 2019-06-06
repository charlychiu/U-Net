from keras.models import Model
from keras.layers import Input, Cropping2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

class Unet():
    def __init__(self):
        print("Initial U-Net model...")
        self.model = self.initial_model()

    def initial_model(self):
        concat_axis = 3

        inputs = Input((572, 572, 1))
        conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='valid', name='conv1_1')(inputs)
        conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='valid', name='conv1_2')(conv1_1)
        pool1 = MaxPooling2D(pool_size=(2, 2), name='maxpooling_1')(conv1_2)

        conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='valid', name='conv2_1')(pool1)
        conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='valid', name='conv2_2')(conv2_1)
        pool2 = MaxPooling2D(pool_size=(2, 2), name='maxpooling_2')(conv2_2)

        conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='valid', name='conv3_1')(pool2)
        conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='valid', name='conv3_2')(conv3_1)
        pool3 = MaxPooling2D(pool_size=(2, 2), name='maxpooling_3')(conv3_2)

        conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='valid', name='conv4_1')(pool3)
        conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='valid', name='conv4_2')(conv4_1)
        pool4 = MaxPooling2D(pool_size=(2, 2), name='maxpooling_4')(conv4_2)

        conv5_1 = Conv2D(1024, (3, 3), activation='relu', padding='valid', name='conv5_1')(pool4)
        conv5_2 = Conv2D(1024, (3, 3), activation='relu', padding='valid', name='conv5_2')(conv5_1)

        upsampling1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='valid', name='upsampling1')(conv5_2)
        crop_conv4_2 = Cropping2D(cropping=((4, 4), (4, 4)), name='cropped_conv4_2')(conv4_2)
        up6 = concatenate([upsampling1, crop_conv4_2], axis=concat_axis, name='skip_connection1')
        conv6_1 = Conv2D(512, (3, 3), activation='relu', padding='valid', name='conv6_1')(up6)
        conv6_2 = Conv2D(512, (3, 3), activation='relu', padding='valid', name='conv6_2')(conv6_1)

        upsampling2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='valid', name='upsampling2')(conv6_2)
        crop_conv3_2 = Cropping2D(cropping=((16, 16), (16, 16)), name='cropped_conv3_2')(conv3_2)
        up7 = concatenate([upsampling2, crop_conv3_2], axis=concat_axis, name='skip_connection2')
        conv7_1 = Conv2D(256, (3, 3), activation='relu', padding='valid', name='conv7_1')(up7)
        conv7_2 = Conv2D(256, (3, 3), activation='relu', padding='valid', name='conv7_2')(conv7_1)

        upsampling3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='valid', name='upsampling3')(conv7_2)
        crop_conv2_2 = Cropping2D(cropping=((40, 40), (40, 40)), name='cropped_conv2_2')(conv2_2)
        up8 = concatenate([upsampling3, crop_conv2_2], axis=concat_axis, name='skip_connection3')
        conv8_1 = Conv2D(128, (3, 3), activation='relu', padding='valid', name='conv8_1')(up8)
        conv8_2 = Conv2D(128, (3, 3), activation='relu', padding='valid', name='conv8_2')(conv8_1)

        upsampling4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='valid', name='upsampling4')(conv8_2)
        crop_conv1_2 = Cropping2D(cropping=((88, 88), (88, 88)), name='cropped_conv1_2')(conv1_2)
        up9 = concatenate([upsampling4, crop_conv1_2], axis=concat_axis, name='skip_connection4')
        conv9_1 = Conv2D(64, (3, 3), activation='relu', padding='valid', name='conv9_1')(up9)
        conv9_2 = Conv2D(64, (3, 3), activation='relu', padding='valid', name='conv9_2')(conv9_1)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid', name='conv10')(conv9_2)

        model = Model(inputs=[inputs], outputs=[conv10])
        return model

    def model_summary(self):
        self.model.summary()

    def get_model(self):
        return self.model
