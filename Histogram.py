import librosa
import numpy as np
import matplotlib.pyplot as plt

def preprocess_audio(audio_file):
    audio_data, sample_rate = librosa.load(audio_file)
    normalized_audio = librosa.util.normalize(audio_data)
    hop_length = 512
    n_fft = 2048
    spectrogram = np.abs(librosa.stft(normalized_audio, hop_length=hop_length, n_fft=n_fft))
    energy = np.sum(spectrogram, axis=0)
    return energy

def create_histogram(energy):
    histogram, bins = np.histogram(energy, bins=10, range=(0, np.max(energy)))
    plt.bar(bins[:-1], histogram, width=np.diff(bins))
    plt.xlabel("Enerji Aralığı")
    plt.ylabel("Frekans")
    plt.title("Enerji Histogramı")
    plt.show()

audio_file = "C:/Users/Cengiz/Desktop/Sesler/Actor_01/03-01-01-01-01-01-01.wav"
energy = preprocess_audio(audio_file)
create_histogram(energy)