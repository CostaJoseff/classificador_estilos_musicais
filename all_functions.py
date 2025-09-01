import librosa, os, h5py, random
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import numpy as np

def search_and_download(resultados, pasta):
    musicas_por_artista = 5
    global stop_thread
    # input(pasta)
    baixadas = 0
    for video in resultados['result']:
        if stop_thread:
            return
        video_mins = video["duration"].split(":")
        if len(video_mins) >= 3:
            continue
        if int(video_mins[0]) > 10:
            continue
        if baixadas >= musicas_por_artista:
            break
        url = video['link']
        try:
            download_audio(url, pasta)
            baixadas += 1
            
        except Exception as e:
            print(f"Erro ao baixar: {e}")

def download_audio(url, pasta_destino):
    cmd = [
        'yt-dlp', '--no-check-certificate',
        '-x', '--audio-format', 'mp3',
        '-o', os.path.join(pasta_destino, '%(title)s.%(ext)s'),
        url
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def get_start_times(duration):
    start_times = {
        "inicio": 0,
        "meio": max(0, (duration / 2) - 15),
        "fim": max(0, duration - 30)
    }
    start_times["inicio_meio"] = (start_times["inicio"] + start_times["meio"]) / 2
    start_times["meio_fim"] = (start_times["meio"] + start_times["fim"]) / 2
    return start_times

def save_plot_spec(S_db, sr, music_name, dir_base, estilo, dir_spec, label):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(zoom(S_db, (.5, .5)), sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Espectrograma - {music_name}')
    plt.tight_layout()
    if not os.path.exists(os.path.join(dir_base, estilo, dir_spec)):
        os.makedirs(os.path.join(dir_base, estilo, dir_spec))
    plt.savefig(os.path.join(dir_base, estilo, dir_spec, f"spectrogram_{label}_{music_name}.png"))
    plt.close()

def mel_spec_and_save_h5(segment, sr, dir_base, estilo, dir_h5, music_name, label):
    S = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db + 80) / 80
    input_array = np.expand_dims(S_norm, axis=0)
    print(input_array.shape)
    return
    # input_array = np.expand_dims(input_array, axis=0)
    # input_array = zoom(input_array, (.5, .5, 1))
    if not os.path.exists(os.path.join(dir_base, estilo, dir_h5)):
        os.makedirs(os.path.join(dir_base, estilo, dir_h5))
    with h5py.File(os.path.join(dir_base, estilo, dir_h5, f"spectrogram_{estilo}_{random.randint(0, 10000)}_{label}_{music_name}.h5"), 'w') as f:
        f.create_dataset("data", data=input_array, compression="gzip")