# Contains Global Default Variables: (difficulty, save_folder)
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from global_params import default_params
# Glob
import glob
import pickle
# Numpy
import numpy as np
# OpenCv
import cv2
# Matplotlib
import matplotlib.pyplot as plt
# CNN
from cnn import CNN
# Tensorflow
from tensorflow.keras import models


def load_and_train():
    # Load Data
    with open(f"{default_params['data_folder']}/good_data_train.pkl", "rb") as f:
        good_data_train = pickle.load(f)
    with open(f"{default_params['data_folder']}/good_data_test.pkl", "rb") as f:
        good_data_test = pickle.load(f)
    with open(f"{default_params['data_folder']}/bad_data_train.pkl", "rb") as f:
        bad_data_train = pickle.load(f)
    with open(f"{default_params['data_folder']}/bad_data_test.pkl", "rb") as f:
        bad_data_test = pickle.load(f)

    # Combine Data
    good_data_song_independent_train = []
    for song in good_data_train:
        for frame in song:
            good_data_song_independent_train.append(frame)
    good_data_song_independent_train = np.array(good_data_song_independent_train)
    
    bad_data_song_independent_train = []
    for song in bad_data_train:
        for frame in song:
            bad_data_song_independent_train.append(frame)
    bad_data_song_independent_train = np.array(bad_data_song_independent_train)

    good_data_song_independent_test = []
    for song in good_data_test:
        for frame in song:
            good_data_song_independent_test.append(frame)
    good_data_song_independent_test = np.array(good_data_song_independent_test)

    bad_data_song_independent_test = []
    for song in bad_data_test:
        for frame in song:
            bad_data_song_independent_test.append(frame)
    bad_data_song_independent_test = np.array(bad_data_song_independent_test)

    # Labels One Hot (Training)
    good_labels_train = [[0, 1] for _ in range(len(good_data_song_independent_train))]
    good_labels_train = np.array(good_labels_train)
    bad_labels_train = [[1, 0] for _ in range(len(bad_data_song_independent_train))]
    bad_labels_train = np.array(bad_labels_train)

    # Labels One Hot (Testing)
    good_labels_test = [[0, 1] for _ in range(len(good_data_song_independent_test))]
    good_labels_test = np.array(good_labels_test)
    bad_labels_test = [[1, 0] for _ in range(len(bad_data_song_independent_test))]
    bad_labels_test = np.array(bad_labels_test)

    # Print Shapes
    print(); print()
    print(f"good_data_song_independent_train: {good_data_song_independent_train.shape} || good_labels_train: {good_labels_train.shape}")
    print(f"bad_data_song_independent_train: {bad_data_song_independent_train.shape} || bad_labels_train: {bad_labels_train.shape}")
    print(f"good_data_song_independent_test: {good_data_song_independent_test.shape} || good_labels_test: {good_labels_test.shape}")
    print(f"bad_data_song_independent_test: {bad_data_song_independent_test.shape} || bad_labels_test: {bad_labels_test.shape}")

    # Combine Train Data
    train_dataset = np.concatenate((good_data_song_independent_train, bad_data_song_independent_train), axis = 0)
    train_labels = np.concatenate((good_labels_train, bad_labels_train), axis = 0)

    # Combine Test Data
    test_dataset = np.concatenate((good_data_song_independent_test, bad_data_song_independent_test), axis = 0)
    test_labels = np.concatenate((good_labels_test, bad_labels_test), axis = 0)

    # Print Shapes
    print(); print()
    print("Combined Train Data:")
    print(f"train_dataset: {train_dataset.shape} || train_labels: {train_labels.shape}")
    print("Combined Test Data:")
    print(f"test_dataset: {test_dataset.shape} || test_labels: {test_labels.shape}")
    print(); print()


    # Hyperparameters
    batch_size = 64
    epochs = 1000

    # Model Declaration
    cnn = CNN()

    # Train the model
    train(model = cnn.model, x = train_dataset, y = train_labels, val_x = test_dataset, val_y = test_labels, batch_size = batch_size, epochs = epochs)

    # Model Name
    model_name = f"model_{default_params['time_increment']}_{default_params['time_steps_per_image']}_{epochs}_epoch.h5"

    # Save Model
    save_model(cnn.model, model_name)

    # Load Model
    model = load_model(model_name)
    print(model.summary())


def train(model, x, y, val_x, val_y, batch_size, epochs, verbose = 1):
    
    # Train the model
    model.fit(
        x = x,
        y = y,
        batch_size = batch_size,
        epochs = epochs,
        verbose = verbose, 
        validation_data = (val_x, val_y),
        shuffle = True,
    )

def save_model(model, model_name):
    # Save Model
    model.save(f"{default_params['model_save_folder']}/{model_name}")

def load_model(model_name):
    # Load Model
    return models.load_model(f"{default_params['model_save_folder']}/{model_name}")

if __name__ == "__main__":
    load_and_train()