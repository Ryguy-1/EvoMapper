from typing import DefaultDict
from create_files import create_info_dot_dat
from create_files import create_difficulty_dot_dat
# For finding BPM
import librosa
# Numpy
import numpy as np
# Pickle
import pickle
# Tensorflow
from tensorflow.keras import models
# Global Params
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from global_params import default_params
from Map_Reader_Trainer.main import print_group_pretty
from Map_Reader_Trainer.reader import Reader

class Data_To_Mapping:

    def __init__(self, data, song_name, song_author_name, audio_file_location = None):
        self.data = data
        self.bpm = self.find_bpm(audio_file_location)
        self.song_name = song_name
        self.song_author_name = song_author_name
        self.audio_file_location = audio_file_location

        # Create Info Dat
        self.info_dat = create_info_dot_dat(self.song_name, self.song_author_name, self.bpm, difficulty = default_params['difficulty'], song_time_offset=0)
        # Initialize Difficulty Dat
        self.difficulty_dat = create_difficulty_dot_dat(self.song_name)
        # Add Notes
        self.add_notes()
        # Compile
        self.difficulty_dat.compile_and_save()
        self.info_dat.compile_and_save()

    # Get Approximate BPM of song
    def find_bpm(self, filename):
        y, sr = librosa.load(filename)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        return round(tempo)

    # Reverse data back into notes in the song
    def add_notes(self):
        # notes_objects = []

        # Loop through each frame
        for i in range(self.data.shape[0]):
            training_frame = self.data[i]
            (channels, y, x) = training_frame.shape
            on_off = training_frame[0]
            red_blue_bomb = training_frame[1]
            direction = training_frame[2]
            for j in range(0, y, 3):
                # Beat
                beat = ((i*1000) + (j/3)) * default_params['time_increment']
                # Convert to note and add to difficulty dat
                self.convert_frame_to_notes_and_add(beat, on_off[j:j+3], red_blue_bomb[j:j+3], direction[j:j+3])

    def convert_frame_to_notes_and_add(self, beat, on_off, red_blue_bomb, direction):
        # Initialize Note Object
        # notes = []
        # Loop through each note
        for i in range(len(on_off)):
            for j in range(len(on_off[i])):
                # If note is on
                if on_off[i][j] == 1:

                    # Line Index
                    line_index = j
                    if line_index not in [0, 1, 2, 3]:
                        raise Exception('Line Index not in [0, 1, 2, 3]')
                    # Line Layer
                    line_layer = abs(2-i)
                    if line_layer not in [0, 1, 2]:
                        raise Exception('Line Layer not in [0, 1, or 2]')

                    # Reverse Type Input
                    type_input = None
                    if red_blue_bomb[i][j] == 0:
                        type_input = 0
                    elif red_blue_bomb[i][j] == 1:
                        type_input = 1
                    elif red_blue_bomb[i][j] == 0.5:
                        type_input = 3
                    else:
                        raise Exception('Invalid Note Type')
                    # Reverse Direction Input
                    direction_input = 0
                    if direction[i][j] == 0:
                        direction_input = 0
                    elif direction[i][j] == 1/8:
                        direction_input = 1
                    elif direction[i][j] == 2/8:
                        direction_input = 2
                    elif direction[i][j] == 3/8:
                        direction_input = 3
                    elif direction[i][j] == 4/8:
                        direction_input = 4
                    elif direction[i][j] == 5/8:
                        direction_input = 5
                    elif direction[i][j] == 6/8:
                        direction_input = 6
                    elif direction[i][j] == 7/8:
                        direction_input = 7
                    elif direction[i][j] == 8/8:
                        direction_input = 8
                    else:
                        raise Exception('Invalid Direction')

                    self.difficulty_dat.add_note(time = beat, line_index = line_index, line_layer = line_layer, type = type_input, cut_direction = direction_input)

                    # # Add Note
                    # notes.append({
                    #     "_time": beat,
                    #     "_lineIndex": line_index,
                    #     "_lineLayer": line_layer,
                    #     "_type": type_input,
                    #     "_cutDirection": direction_input,
                    # })

        # return notes

if __name__ == "__main__":
    # Params
    song_name = "Outerspacee"
    author_name = "Your's truly"
    audio_file_location = f"{default_params['custom_songs_folder']}/{song_name}/song.ogg"
    # Load Data
    data = pickle.load(open("best_per_generation_1.pkl", "rb"))[-1]
    print(data.shape)
    # Create Mapping
    mapping = Data_To_Mapping(data, song_name, author_name, audio_file_location)