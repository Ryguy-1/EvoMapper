from reader import Reader
# Contains Global Default Variables: (difficulty, save_folder)
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from global_params import default_params
# Opencv
import cv2
# Glob
import glob
# Pickle
import pickle

import numpy as np
import shutil

# Stores Good Songs and Data List will be Populated during Aggregation
good_songs = glob.glob(f"{default_params['training_songs_folder']}/Good_Songs/*")
good_songs = [song.split("\\")[-1] for song in good_songs]
good_data = []

# Stores Bad Songs and Data List will be Populated during Aggregation
bad_songs = glob.glob(f"{default_params['training_songs_folder']}/Bad_Songs/*")
bad_songs = [song.split("\\")[-1] for song in bad_songs]
bad_data = []


# Aggregates Songs and Saves Data to Lists for Pickleing
def aggregate_songs():
    # Keep Track of Total Songs Loaded
    total_songs = 0

    # Get Good Data
    for song_name in good_songs:
        try:
            reader = Reader(f"{default_params['training_songs_folder']}/Good_Songs/{song_name}", default_params['difficulty'])
            data = reader.get_song_data()
            good_data.append(data)
            total_songs += 1
            print(f'Good Song # {total_songs} / {len(good_songs)} Loaded = {total_songs / len(good_songs) * 100}%')
        except Exception:
            write_error_to_error_file(f"Error Loading {song_name}")
        
    total_songs = 0
    # Get Bad Data
    for song_name in bad_songs:
        try:
            reader = Reader(f"{default_params['training_songs_folder']}/Bad_Songs/{song_name}", default_params['difficulty'])
            data = reader.get_song_data()
            bad_data.append(data)
            total_songs += 1
            print(f'Bad Song # {total_songs} / {len(bad_songs)} Loaded = {total_songs / len(bad_songs) * 100}%')
        except Exception:
            write_error_to_error_file(f"Error Loading {song_name}")
            

    # Split Data into Train and Test
    good_train = np.array(good_data[:int(len(good_data)*default_params['train_test_split'])])
    good_test = np.array(good_data[int(len(good_data)*default_params['train_test_split']):])
    bad_train = np.array(bad_data[:int(len(bad_data)*default_params['train_test_split'])])
    bad_test = np.array(bad_data[int(len(bad_data)*default_params['train_test_split']):])

    # Save Data using Pickle
    print(f"Saving {total_songs} Songs")
    with open(f"{default_params['data_folder']}/good_data_test.pkl", "wb") as f:
        pickle.dump(good_test, f)
    with open(f"{default_params['data_folder']}/bad_data_test.pkl", "wb") as f:
        pickle.dump(bad_test, f)
    with open(f"{default_params['data_folder']}/good_data_train.pkl", "wb") as f:
        pickle.dump(good_train, f)
    with open(f"{default_params['data_folder']}/bad_data_train.pkl", "wb") as f:
        pickle.dump(bad_train, f)


    print(f"Gathered Data and Saved to {default_params['data_folder']} with train-test split of {default_params['train_test_split']}")


def write_error_to_error_file(error):
    with open(default_params['error_log_file_convert_songs'], 'a') as f:
        f.write(f"Error: {error}" + '\n')

def print_group_pretty(group):
    (channels, y, x) = group.shape
    on_off = group[0]
    red_blue_bomb = group[1]
    direction = group[2]

    frame_counter = 0
    for i in range(y):
        if i%3 == 0:
            print(f'Frame # {frame_counter}')
            frame_counter += 1
        print(f"On/Off: || {on_off[i]} || Red/Blue/Bomb: || {red_blue_bomb[i]} || Direction: || {direction[i]} ||")


if __name__ == "__main__":
    aggregate_songs()
