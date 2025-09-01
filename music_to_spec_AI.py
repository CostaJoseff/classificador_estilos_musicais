import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from logger import log
from all_functions import *
import os, h5py, librosa
from threading import Thread
import numpy as np

dir_base = "musicas"
dir_spec = "specs"
dir_h5 = "h5"

estilos = os.listdir(dir_base)

for estilo in estilos:
    print()
    musicas = os.listdir(os.path.join(dir_base, estilo))
    for i, musica in enumerate(musicas):
        log(estilo, f"{i+1}/{len(musicas)}", i+1, len(musicas), progress_char="|", void_char="_")
        music_name = musica.replace(".webm", "")
        file_path = os.path.join(dir_base, estilo, musica)
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        start_times = get_start_times(duration)
        threads = []
        for label, start_time in start_times.items():
            start_sample = int(start_time * sr)
            end_sample = start_sample + int(30 * sr)
            segment = y[start_sample:end_sample]

            thread = Thread(
                target=mel_spec_and_save_h5,
                args=[segment, sr, dir_base, estilo, dir_h5, music_name, label]
            )
            thread.start()
            threads.append(thread)
        
        thread: Thread
        for thread in threads:
            thread.join()