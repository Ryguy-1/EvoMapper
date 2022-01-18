import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from global_params import default_params
# Librosa
import librosa
# Song Creator
from create_files import create_info_dot_dat
from create_files import create_difficulty_dot_dat
# Random
import random

class CreateHardMapping:

    def __init__(self, song_name, song_author_name, audio_file_location = None):
        if audio_file_location is None:
            return

        # Audio File Location
        self.audio_file_location = audio_file_location

        # General Information
        self.song_name = song_name
        self.song_author_name = song_author_name



        # Read Audio
        (y, sr) = librosa.load(self.audio_file_location)
        # Get BPM
        self.bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
        self.bpm = int(round(self.bpm))
        print("BPM:", self.bpm)
        # Find Onsets
        self.onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        print(f"Number of Onsets: {len(self.onsets)}")

        # Create Info Dat
        self.info_dat = create_info_dot_dat(self.song_name, self.song_author_name, self.bpm, difficulty = "ExpertPlus", song_time_offset=0)
        # Initialize Difficulty Dat
        self.difficulty_dat = create_difficulty_dot_dat(self.song_name)
        # Add Notes
        self.add_notes()
        # Compile and Save
        self.info_dat.compile_and_save(save_folder = "Map_Creator/song_temp_folder")
        self.difficulty_dat.compile_and_save(save_folder = "Map_Creator/song_temp_folder")

    def add_notes(self):
        # Add Notes
        # (Previous beat time so no overlapping notes)
        previous_beat_time = 0
        for i in range(len(self.onsets)):
            # Convert onset time to beat time
            beat_time = self.onsets[i] * (self.bpm / 60)
            # Round Beat Time to time increment
            beat_time = round(beat_time / default_params['time_increment']) * default_params['time_increment']
            # Check if this onset is the same as the previous onset
            if beat_time == previous_beat_time:
                continue
            # Add Note
            self.difficulty_dat.add_note(time=beat_time, line_index=random.randint(0, 3), line_layer=random.randint(0, 2), type=generate_block_type_random(), cut_direction=random.randint(0, 8))
            self.difficulty_dat.add_note(time=beat_time, line_index=random.randint(0, 3), line_layer=random.randint(0, 2), type=generate_block_type_random(), cut_direction=random.randint(0, 8))
            # Update Previous Beat Time
            previous_beat_time = beat_time

def generate_block_type_random():
    # rand_num = random.randint(0, 2) -> For Bombs also
    rand_num = random.randint(0, 1)
    if rand_num == 0:
        return 0
    elif rand_num == 1:
        return 1
    # For Bombs (indices skip over 2) ->>>> No Bombs for now
    # elif rand_num == 2:
    #     return 3