# Tensorflow
from email.policy import default
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses, metrics

# Contains Global Default Variables: (difficulty, save_folder)
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from global_params import default_params

# 0.1_1000
class CNN:
    
    def __init__(self):
        # Parameters
        self.input_shape = (3, 3000, 4)
        self.num_classes = 2

        # Model
        self.model = models.Sequential()

        # Convolution 1 (scans once horizontally and not vertically)
        # Input Shape = self.input_shape -> (32 channels), (3 height and 4 width kernel), (vertical stride length of 3)
        self.model.add(layers.Conv2D(96, (3, 4), strides=(3, 1), activation=None, input_shape=self.input_shape, data_format='channels_first'))
        self.model.add(layers.Activation('relu'))
        self.model.add(layers.BatchNormalization())
        assert self.model.output_shape == (None, 96, 1000, 1)
        # Pool
        self.model.add(layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), data_format='channels_first'))
        assert self.model.output_shape == (None, 96, 500, 1)
        # Dropout
        self.model.add(layers.Dropout(0.5)) # was 0.8
        assert self.model.output_shape == (None, 96, 500, 1)

        # Padding
        self.model.add(layers.ZeroPadding2D(((0, 1), (0, 0)), data_format='channels_first'))
        assert self.model.output_shape == (None, 96, 501, 1)

        # Convolution 2
        self.model.add(layers.Conv2D(64, (3, 1), strides=(3, 1), activation=None, data_format='channels_first'))
        self.model.add(layers.Activation('relu'))
        self.model.add(layers.BatchNormalization())
        assert self.model.output_shape == (None, 64, 167, 4)
        # Padding
        self.model.add(layers.ZeroPadding2D(((0, 1), (0, 0)), data_format='channels_first'))
        assert self.model.output_shape == (None, 64, 168, 1)
        # Pool
        self.model.add(layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), data_format='channels_first'))
        assert self.model.output_shape == (None, 64, 84, 1)
        # Dropout
        self.model.add(layers.Dropout(0.5)) # was 0.7
        assert self.model.output_shape == (None, 64, 84, 1)

        self.model.add(layers.Flatten())
        assert self.model.output_shape == (None, 64 * 84 * 1)

        self.model.add(layers.Dense(2048, activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.6)) # was 0.6
        assert self.model.output_shape == (None, 2048)

        self.model.add(layers.Dense(1024, activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.6)) # was 0.6
        assert self.model.output_shape == (None, 1024)

        self.model.add(layers.Dense(512, activation='relu'))
        self.model.add(layers.BatchNormalization())
        assert self.model.output_shape == (None, 512)

        self.model.add(layers.Dense(self.num_classes, activation='softmax'))
        assert self.model.output_shape == (None, self.num_classes)

        self.loss = losses.BinaryCrossentropy()
        self.optimizer = optimizers.Adam(learning_rate=0.001)

        self.model.compile(
            loss = self.loss,
            optimizer = self.optimizer,
            metrics = ['accuracy'],
        )

    def __str__(self):
        self.model.summary()
        return ''



# 0.5_200
# class CNN:
    
#     def __init__(self):
#         # Parameters
#         self.input_shape = (3, 600, 4)
#         self.num_classes = 2

#         # Model
#         self.model = models.Sequential()

#         # Convolution 1 (scans once horizontally and not vertically)
#         # Input Shape = self.input_shape -> (32 channels), (3 height and 4 width kernel), (vertical stride length of 3)
#         self.model.add(layers.Conv2D(96, (3, 4), strides=(3, 1), activation=None, input_shape=self.input_shape, data_format='channels_first'))
#         self.model.add(layers.Activation('relu'))
#         self.model.add(layers.BatchNormalization())
#         self.model.add(layers.Dropout(0.5)) # was 0.8
#         assert self.model.output_shape == (None, 96, 200, 1)

#         self.model.add(layers.ZeroPadding2D(((0, 1), (0, 0)), data_format='channels_first'))
#         assert self.model.output_shape == (None, 96, 201, 1)

#         self.model.add(layers.Conv2D(64, (3, 1), strides=(3, 1), activation=None, data_format='channels_first'))
#         self.model.add(layers.Activation('relu'))
#         self.model.add(layers.BatchNormalization())
#         self.model.add(layers.Dropout(0.5)) # was 0.7
#         assert self.model.output_shape == (None, 64, 67, 1)

#         self.model.add(layers.Flatten())
#         assert self.model.output_shape == (None, 64 * 67 * 1)

#         self.model.add(layers.Dense(2048, activation='relu'))
#         self.model.add(layers.BatchNormalization())
#         self.model.add(layers.Dropout(0.6)) # was 0.6
#         assert self.model.output_shape == (None, 2048)

#         self.model.add(layers.Dense(1024, activation='relu'))
#         self.model.add(layers.BatchNormalization())
#         self.model.add(layers.Dropout(0.6)) # was 0.6
#         assert self.model.output_shape == (None, 1024)

#         self.model.add(layers.Dense(512, activation='relu'))
#         self.model.add(layers.BatchNormalization())
#         assert self.model.output_shape == (None, 512)

#         self.model.add(layers.Dense(self.num_classes, activation='softmax'))
#         assert self.model.output_shape == (None, self.num_classes)

#         self.loss = losses.BinaryCrossentropy()
#         self.optimizer = optimizers.Adam(learning_rate=0.001)

#         self.model.compile(
#             loss = self.loss,
#             optimizer = self.optimizer,
#             metrics = ['accuracy'],
#         )

#     def __str__(self):
#         self.model.summary()
#         return ''

# # 0.2_100
# class CNN:
    
#     def __init__(self):
#         # Parameters
#         self.input_shape = (3, 300, 4)
#         self.num_classes = 2

#         # Model
#         self.model = models.Sequential()

#         # Convolution 1 (scans once horizontally and not vertically)
#         # Input Shape = self.input_shape -> (32 channels), (3 height and 4 width kernel), (vertical stride length of 3)
#         self.model.add(layers.Conv2D(96, (3, 4), strides=(3, 1), activation=None, input_shape=self.input_shape, data_format='channels_first'))
#         self.model.add(layers.Activation('relu'))
#         self.model.add(layers.BatchNormalization())
#         self.model.add(layers.Dropout(0.5)) # was 0.8
#         assert self.model.output_shape == (None, 96, 100, 1)

#         self.model.add(layers.ZeroPadding2D(((0, 2), (0, 0)), data_format='channels_first'))
#         assert self.model.output_shape == (None, 96, 102, 1)

#         self.model.add(layers.Conv2D(64, (3, 1), strides=(3, 1), activation=None, data_format='channels_first'))
#         self.model.add(layers.Activation('relu'))
#         self.model.add(layers.BatchNormalization())
#         self.model.add(layers.Dropout(0.5)) # was 0.7
#         assert self.model.output_shape == (None, 64, 34, 1)

#         self.model.add(layers.Flatten())
#         assert self.model.output_shape == (None, 64 * 34 * 1)

#         self.model.add(layers.Dense(2048, activation='relu'))
#         self.model.add(layers.BatchNormalization())
#         self.model.add(layers.Dropout(0.6)) # was 0.6
#         assert self.model.output_shape == (None, 2048)

#         self.model.add(layers.Dense(1024, activation='relu'))
#         self.model.add(layers.BatchNormalization())
#         self.model.add(layers.Dropout(0.6)) # was 0.6
#         assert self.model.output_shape == (None, 1024)

#         self.model.add(layers.Dense(512, activation='relu'))
#         self.model.add(layers.BatchNormalization())
#         assert self.model.output_shape == (None, 512)

#         self.model.add(layers.Dense(self.num_classes, activation='softmax'))
#         assert self.model.output_shape == (None, self.num_classes)

#         self.loss = losses.BinaryCrossentropy()
#         self.optimizer = optimizers.Adam(learning_rate=0.001)

#         self.model.compile(
#             loss = self.loss,
#             optimizer = self.optimizer,
#             metrics = ['accuracy'],
#         )

#     def __str__(self):
#         self.model.summary()
#         return ''




# 0.5_45 -> 75% accuracy roughly
# class CNN:
    
#     def __init__(self):
#         # Parameters
#         self.input_shape = (3, 135, 4)
#         self.num_classes = 2

#         # Model
#         self.model = models.Sequential()

#         # Convolution 1 (scans once horizontally and not vertically)
#         # Input Shape = self.input_shape -> (32 channels), (3 height and 4 width kernel), (vertical stride length of 3)
#         self.model.add(layers.Conv2D(32, (3, 4), strides=(3, 1), activation=None, input_shape=self.input_shape, data_format='channels_first'))
#         self.model.add(layers.Activation('relu'))
#         self.model.add(layers.BatchNormalization())
#         self.model.add(layers.Dropout(0.8)) # was 0.7
#         assert self.model.output_shape == (None, 32, 45, 1)

#         self.model.add(layers.Conv2D(64, (3, 1), strides=(3, 1), activation=None, data_format='channels_first'))
#         self.model.add(layers.Activation('relu'))
#         self.model.add(layers.BatchNormalization())
#         self.model.add(layers.Dropout(0.7)) # was 0.6
#         assert self.model.output_shape == (None, 64, 15, 1)

#         self.model.add(layers.Flatten())
#         assert self.model.output_shape == (None, 64 * 15 * 1)

#         self.model.add(layers.Dense(512, activation='relu'))
#         self.model.add(layers.BatchNormalization())
#         assert self.model.output_shape == (None, 512)

#         self.model.add(layers.Dense(64, activation='relu'))
#         self.model.add(layers.BatchNormalization())
#         assert self.model.output_shape == (None, 64)

#         self.model.add(layers.Dense(self.num_classes, activation='softmax'))
#         assert self.model.output_shape == (None, self.num_classes)

#         self.loss = losses.BinaryCrossentropy()
#         self.optimizer = optimizers.Adam(learning_rate=0.0005)

#         self.model.compile(
#             loss = self.loss,
#             optimizer = self.optimizer,
#             metrics = ['accuracy'],
#         )




# 0.05_500 -> like 66% usually
# class CNN:
    
#     def __init__(self):
#         # Parameters
#         self.input_shape = (3, 1500, 4)
#         self.num_classes = 2

#         # Model
#         self.model = models.Sequential()

#         # Convolution 1 (scans once horizontally and not vertically)
#         # Input Shape = self.input_shape -> (32 channels), (3 height and 4 width kernel), (vertical stride length of 3)
#         self.model.add(layers.Conv2D(32, (3, 4), strides=(3, 1), activation=None, input_shape=self.input_shape, data_format='channels_first'))
#         self.model.add(layers.Activation('relu'))
#         self.model.add(layers.BatchNormalization())
#         self.model.add(layers.Dropout(0.7)) # was 0.7
#         assert self.model.output_shape == (None, 32, 500, 1)

#         self.model.add(layers.ZeroPadding2D(((0, 1), (0, 0)), data_format='channels_first'))
#         assert self.model.output_shape == (None, 32, 501, 1)

#         self.model.add(layers.Conv2D(64, (3, 1), strides=(3, 1), activation=None, data_format='channels_first'))
#         self.model.add(layers.Activation('relu'))
#         self.model.add(layers.BatchNormalization())
#         assert self.model.output_shape == (None, 64, 167, 1)
#         self.model.add(layers.ZeroPadding2D(((0, 1), (0, 0)), data_format='channels_first'))
#         assert self.model.output_shape == (None, 64, 168, 1)
#         self.model.add(layers.MaxPool2D((2, 1), data_format='channels_first'))
#         assert self.model.output_shape == (None, 64, 84, 1)
#         self.model.add(layers.Dropout(0.6)) # was 0.6
        

#         self.model.add(layers.Flatten())
#         assert self.model.output_shape == (None, 64 * 84 * 1)

#         self.model.add(layers.Dense(2048, activation='relu'))
#         self.model.add(layers.BatchNormalization())
#         self.model.add(layers.Dropout(0.5)) # was 0.5
#         assert self.model.output_shape == (None, 2048)

#         self.model.add(layers.Dense(512, activation='relu'))
#         self.model.add(layers.BatchNormalization())
#         assert self.model.output_shape == (None, 512)

#         self.model.add(layers.Dense(self.num_classes, activation='softmax'))
#         assert self.model.output_shape == (None, self.num_classes)

#         self.loss = losses.BinaryCrossentropy()
#         self.optimizer = optimizers.Adam(learning_rate=0.0005)

#         self.model.compile(
#             loss = self.loss,
#             optimizer = self.optimizer,
#             metrics = ['accuracy'],
#         )



# 0.1_100 -> stabalized very much at 70% val accuracy

# class CNN:
    
#     def __init__(self):
#         # Parameters
#         self.input_shape = (3, 300, 4)
#         self.num_classes = 2

#         # Model
#         self.model = models.Sequential()

#         # Convolution 1 (scans once horizontally and not vertically)
#         # Input Shape = self.input_shape -> (32 channels), (3 height and 4 width kernel), (vertical stride length of 3)
#         self.model.add(layers.Conv2D(32, (3, 4), strides=(3, 1), activation=None, input_shape=self.input_shape, data_format='channels_first'))
#         self.model.add(layers.Activation('relu'))
#         self.model.add(layers.BatchNormalization())
#         self.model.add(layers.Dropout(0.8)) # was 0.7
#         assert self.model.output_shape == (None, 32, 100, 1)

#         self.model.add(layers.ZeroPadding2D(((0, 2), (0, 0)), data_format='channels_first'))
#         assert self.model.output_shape == (None, 32, 102, 1)

#         self.model.add(layers.Conv2D(64, (3, 1), strides=(3, 1), activation=None, data_format='channels_first'))
#         self.model.add(layers.Activation('relu'))
#         self.model.add(layers.BatchNormalization())
#         self.model.add(layers.Dropout(0.7)) # was 0.6
#         assert self.model.output_shape == (None, 64, 34, 1)

#         self.model.add(layers.Flatten())
#         assert self.model.output_shape == (None, 64 * 34 * 1)

#         self.model.add(layers.Dense(2048, activation='relu'))
#         self.model.add(layers.BatchNormalization())
#         self.model.add(layers.Dropout(0.6)) # was 0.5
#         assert self.model.output_shape == (None, 2048)

#         self.model.add(layers.Dense(512, activation='relu'))
#         self.model.add(layers.BatchNormalization())
#         assert self.model.output_shape == (None, 512)

#         self.model.add(layers.Dense(self.num_classes, activation='softmax'))
#         assert self.model.output_shape == (None, self.num_classes)

#         self.loss = losses.BinaryCrossentropy()
#         self.optimizer = optimizers.Adam(learning_rate=0.0005)

#         self.model.compile(
#             loss = self.loss,
#             optimizer = self.optimizer,
#             metrics = ['accuracy'],
#         )