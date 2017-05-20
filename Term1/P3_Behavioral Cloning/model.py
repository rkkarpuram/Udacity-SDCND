import os
import csv
import cv2
import numpy as np
from sklearn.utils import shuffle

dir = r'./my_data2'

samples = []
with open(os.path.join(dir, 'driving_log.csv')) as csvfile:
    lines = csv.reader(csvfile)
    for line in lines:
        samples.append(line)

from sklearn.cross_validation import train_test_split
shuffle(samples)

# test and validation samples split with 80:20 ratio.
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        shuffle(samples)
        
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                local_path = dir + '/IMG/'
                # for each image path from log, obtain the file name as token
                centre_img_token = batch_sample[0].split('\\')[-1]
                left_img_token = batch_sample[1].split('\\')[-1]
                right_img_token = batch_sample[2].split('\\')[-1]
                
                # build pah names for each of left, centre and right images
                centre_img_path = os.path.join(local_path, centre_img_token)
                left_img_path = os.path.join(local_path, left_img_token)
                right_img_path = os.path.join(local_path, right_img_token)
                
                centre_image = cv2.imread(centre_img_path)
                left_image = cv2.imread(left_img_path)
                right_image = cv2.imread(right_img_path)
                
                # Append each image into the images array.
                # For the centre image the streeging angle is unchanged from log
                # For the left and right images the stteering angle is adjusted
                # by adding and subtracting  0.2 (radians) respectively.
                # All three images are then flipped vertically and added to the samples
                # and so are their steering angles reversed
                centre_angle = float(batch_sample[3])
                images.append(centre_image)
                angles.append(centre_angle)
                images.append(left_image)
                angles.append(centre_angle + 0.2)
                images.append(right_image)
                angles.append(centre_angle - 0.2)
                images.append(cv2.flip(centre_image, 1))
                angles.append(-centre_angle)
                images.append(cv2.flip(left_image, 1))
                angles.append(-(centre_angle + 0.2))
                images.append(cv2.flip(right_image, 1))
                angles.append(-(centre_angle-0.2))
                
            X_train = np.asarray(images)
            y_train = np.asarray(angles)
            
            yield shuffle(X_train, y_train)

batch_size = 16
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

def cnn_model():
    model = Sequential()
    # Crop the top 50 and bottom 20 pixels as that part of image is not relevant for model
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    # Normalize the image values to range -0.5 to 0.5
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    # First three layers of CNN with relu non-linear activations
    # followed by MaxPool layers    
    model.add(Convolution2D(24, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(36, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(48, 3, 3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D())
    # Flaten all neurons to form a fully connected layer followed by
    # three FC layers and then the output
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dense(1))
    model.summary()
    
    model.compile(optimizer='adam', loss='mse')
    
    return model

model = cnn_model()

# Use fit_generator to train the model with batch sizes.
# Both train and validation number of samples are adjusted for
# number of transformations and augmentations for each image in generator.
model.fit_generator(train_generator, samples_per_epoch=len(train_samples) * 6,
            validation_data=validation_generator, nb_val_samples=len(validation_samples)*6,
            nb_epoch = 5)

model.save('model.h5')
