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
ignore = []

for estilo in estilos:
    if estilo in ignore:
        continue
    print()
    musicas = os.listdir(os.path.join(dir_base, estilo))
    threads = []
    for i, musica in enumerate(musicas):
        if ".mp3" not in musica and ".webm" not in musica:
            continue
        thread: Thread = Thread(
            target=dir_to_mel_and_save_h5,
            args=[estilo, musicas, musica, dir_base, dir_h5, i]
        )
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()
        