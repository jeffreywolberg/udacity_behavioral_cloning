# flip half input images and steering angles np.fliplr(image), -steering_angle
# use l and r camera images by pretending they are in center, and adding/subtracting correction.
# if l image, add correction, if r image, subtract correction

# start with 160x320x3 image into network by reading using cv2 (W, H), output steering angle
# clip top 50 pix, and bottom 20 pix using
# model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
# normalize pixels using Lambda(lambda x: (x / 255.0) - 0.5)


import os
import csv
import cv2
import numpy as np
import sklearn
from enum import Enum

from keras.optimizers import Adam
from keras.models import load_model, save_model
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Conv2D, Dense, Flatten, MaxPool2D, Dropout
from sklearn.model_selection import train_test_split

shuffle = sklearn.utils.shuffle
ceil = np.ceil
join = os.path.join

class ImagePos(Enum):
    center=0
    left=1
    right=2

class BehavioralCloning(object):
    def __init__(self):

        self.batch_size = 32
        self.crop_up, self.crop_down = 50, 20
        self.orig_dims = (160, 320, 3)
        self.model_name = 'my_model_5.h5'
    def get_train_val_data(self):
        samples = []
        file_names = [
            # r'.\driving_logs\driving_log_train.csv',
            r'.\driving_logs\driving_log_train2.csv']
        #file_name = '/home/data/driving_log.csv'
        # file_name = './driving_log_train.csv'
        for i, file_name in enumerate(file_names):
            with open(file_name) as csvfile:
                reader = csv.reader(csvfile)
                for line in reader:
                    samples.append([line, i])

        train_samples, validation_samples = train_test_split(samples, test_size=0.2)
        return train_samples, validation_samples


    def generator(self, samples, batch_size=32):
        def img_name(dir_name, img_details, imagePos: ImagePos):
            if dir_name.startswith("/opt/carnd_p3/data/"):
                return join(dir_name, img_details[imagePos.value].split("/")[-1])
            else:
                return join(dir_name, img_details[imagePos.value].split("\\")[-1])

        num_samples = len(samples)
        correction = .15
        # dir_name = r'/home/data/IMG/'
        dir_names = [
        # r'.\IMG_folders\IMG_train',
                     r'.\IMG_folders\IMG_train2']

        while 1:  # Loop forever so the generator never terminates
            shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]
                images = []
                angles = []
                for batch_sample in batch_samples:
                    dir_index = batch_sample[1]
                    dir_name = dir_names[dir_index]
                    img_details = batch_sample[0]

                    center_name = img_name(dir_name, img_details, ImagePos.center)
                    center_image = cv2.imread(center_name)

                    if center_image is None:
                        print("Image doesn't exist")
                        continue

                    center_angle = float(img_details[3])
                    center_flipped_image = np.fliplr(center_image)
                    center_flipped_angle = -center_angle

                    left_name = img_name(dir_name, img_details, ImagePos.left)
                    left_image = cv2.imread(left_name)
                    left_angle = float(img_details[3]) + correction

                    right_name = img_name(dir_name, img_details, ImagePos.right)
                    right_image = cv2.imread(right_name)
                    right_angle = float(img_details[3]) - correction

                    images.extend([center_image, center_flipped_image, left_image, right_image])
                    angles.extend([center_angle, center_flipped_angle, left_angle, right_angle])

                X_train = np.array(images)
                y_train = np.array(angles)
                # yield shuffle(X_train, y_train)
                yield X_train, y_train


    def create_model(self):
        model = Sequential()
        model.add(Cropping2D(cropping=((self.crop_up, self.crop_down), (0, 0)), input_shape=self.orig_dims))
        # 90, 360, 3
        dims_1 = (self.orig_dims[0] - self.crop_up - self.crop_down, self.orig_dims[1], self.orig_dims[2])
        print(dims_1)
        model.add(Lambda(lambda x: (x / 255.0) - 0.5))
        filters, kernel_size, stride = 24, 5, (1, 1)
        model.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=stride,
                         padding='valid', activation='relu', input_shape=dims_1))
        model.add(MaxPool2D((2,2)))
        filters, kernel_size, stride = 36, 5, (1, 1)
        model.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=stride,
                         padding='valid', activation='relu'))
        model.add(MaxPool2D((2,2)))
        filters, kernel_size, stride = 48, 3, (1, 1)
        model.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=stride,
                         padding='valid', activation='relu'))
        model.add(MaxPool2D((1,2)))
        filters, kernel_size, stride = 64, 3, (1, 1)
        model.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=stride,
                         padding='valid', activation='relu'))
        model.add(MaxPool2D((2,2)))
        filters, kernel_size, stride = 64, 3, (1, 1)
        model.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=stride,
                         padding='valid', activation='relu'))
        model.add(Dropout(.3))
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Dense(50))
        model.add(Dense(10))
        model.add(Dense(1))

        return model

    def load_my_model(self):
        #         return load_model(f".{os.sep}{self.model_name}")
        num = int(self.model_name.split('_')[-1].split('.h5')[0]) - 1
        return load_model(join(os.getcwd(), "my_model_{}.h5".format(num)))

    def train_model(self):
        train_samples, validation_samples = self.get_train_val_data()
        # compile and train the model using the generator function
        train_generator = self.generator(train_samples, batch_size=self.batch_size)
        validation_generator = self.generator(validation_samples, batch_size=self.batch_size)

        # model = self.load_my_model()
        model = self.create_model()
        optimizer = Adam(lr=.0005)
        # print(model.summary())
        model.compile(loss='mse', optimizer=optimizer)
        model.fit_generator(train_generator, steps_per_epoch=ceil(len(train_samples)/self.batch_size),
                    validation_data=validation_generator,
                    validation_steps=ceil(len(validation_samples)/self.batch_size),
                    epochs=2, verbose=1)
        # model.save(fr'.{os.sep}{self.model_name}')
        save_model(model, join(os.getcwd(), self.model_name))

if __name__ == '__main__':
    inst = BehavioralCloning()
    inst.train_model()


