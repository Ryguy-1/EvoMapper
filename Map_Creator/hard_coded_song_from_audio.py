import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Map_Reader_Trainer.reader import Reader
from global_params import default_params
# Librosa
import librosa
# Song Creator
from create_files import create_info_dot_dat
from create_files import create_difficulty_dot_dat
# NumPy
import numpy as np

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
        self.info_dat.compile_and_save()
        self.difficulty_dat.compile_and_save()

    def add_notes(self):
        # Add Notes
        for i in range(len(self.onsets)):
            # Convert onset time to beat time
            beat_time = self.onsets[i] * (self.bpm / 60)
            # Round Beat Time to time increment
            beat_time = round(beat_time / default_params['time_increment']) * default_params['time_increment']
            # Add Note
            self.difficulty_dat.add_note(time=beat_time, line_index=2, line_layer=0, type=1, cut_direction=i%2)
            self.difficulty_dat.add_note(time=beat_time, line_index=1, line_layer=0, type=0, cut_direction=i%2)


if __name__ == "__main__":
    mapping = CreateHardMapping(song_name = 'Sakay', song_author_name = 'rounded0.1', audio_file_location = default_params['custom_songs_folder'] + '/Sakay/song.ogg')