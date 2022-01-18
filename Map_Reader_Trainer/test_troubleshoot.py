import glob
import numpy as np
import cv2
import json
import pickle
# Tensorflow
from tensorflow.keras import models
from reader import Reader

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from global_params import default_params
from Map_Reader_Trainer.main import print_group_pretty


def main():
    model = load_model('model_0.5_200_1000_epoch.h5')
    # Load Song
    song_name = "202db (Sakayume - Joetastic, Xhera & nolan121405)"
    song_folder_path = f"Map_Reader_Trainer/test_troubleshoot/{song_name}"
    reader = Reader(song_folder_path, default_params['difficulty'])
    data = reader.get_song_data()
    print(data.shape)

    # Predict Song Classification using model
    prediction = model.predict(data)
    print(prediction)

    positive = 0; negative = 0
    for value_arr in prediction:
        positive += value_arr[1]
        negative += value_arr[0]
        print(np.argmax(value_arr))
    print(f"Percent Good: {positive/(positive + negative)}")


def load_model(model_name):
    # Load Model
    return models.load_model(f"{default_params['model_save_folder']}/{model_name}")


if __name__ == "__main__":
    # main()
    model = load_model("model_0.1_1000_1000_epoch.h5")
    # Load Data
    data = pickle.load(open("best_per_generation_0.pkl", "rb"))[-1]
    print(data.shape)
    print_group_pretty(data[0])