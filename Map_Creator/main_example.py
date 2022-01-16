from typing import DefaultDict
from create_files import create_info_dot_dat
from create_files import create_difficulty_dot_dat
# For finding BPM
import librosa
# Global Params
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from global_params import default_params


# find_bpm(f"{default_params['save_folder']}/song.ogg")

def main():
    info_dat = create_info_dot_dat("DO OR DIE TEST SONG", "Cool Test Name", 88, difficulty = "ExpertPlus", song_time_offset=0.125, shuffle_period=0.5)
    info_dat.compile_and_save()

    difficulty_dat = create_difficulty_dot_dat(song_name = "DO OR DIE TEST SONG", difficulty = "ExpertPlus")
    for i in range(60):
        difficulty_dat.add_note(time=i+1, line_index=2, line_layer=0, type=1, cut_direction=1)
    difficulty_dat.compile_and_save()

# Get Approximate BPM of song
def find_bpm(filename):
    y, sr = librosa.load(filename)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    return round(tempo)

if __name__ == "__main__":
    main()
    # print(find_bpm("Song_Folder/song.ogg"))