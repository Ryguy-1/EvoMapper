import librosa
import librosa.display
from numba.cuda import test
import cv2
import matplotlib.pyplot as plt
import numpy as np

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from global_params import default_params


test_name = "25f (Reality Check Through The Skull - rickput)"
song_local_name = "RCTTS.egg"
file_path = f'{default_params["custom_songs_folder"]}/{test_name}/{song_local_name}'

y, sr = librosa.load(file_path, offset = 10, duration=50)
oenv = librosa.onset.onset_strength(y=y, sr=sr)
times = librosa.times_like(oenv)

# Detect events without backtracking
onset_raw = librosa.onset.onset_detect(onset_envelope=oenv,
                                       backtrack=False)
onset_bt = librosa.onset.onset_backtrack(onset_raw, oenv)


S = np.abs(librosa.stft(y=y))
rms = librosa.feature.rms(S=S)
onset_bt_rms = librosa.onset.onset_backtrack(onset_raw, rms[0])

fig, ax = plt.subplots(nrows=3, sharex=True)
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                         y_axis='log', x_axis='time', ax=ax[0])
ax[0].label_outer()
ax[1].plot(times, oenv, label='Onset strength')
ax[1].vlines(librosa.frames_to_time(onset_raw), 0, oenv.max(), label='Raw onsets')
ax[1].vlines(librosa.frames_to_time(onset_bt), 0, oenv.max(), label='Backtracked', color='r')
ax[1].legend()
ax[1].label_outer()
ax[2].plot(times, rms[0], label='RMS')
ax[2].vlines(librosa.frames_to_time(onset_bt_rms), 0, rms.max(), label='Backtracked (RMS)', color='r')
ax[2].legend()
ax[2].label_outer()
plt.tight_layout()
plt.show()


