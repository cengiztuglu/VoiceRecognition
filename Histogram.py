import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean


def preprocess_audio(audio_file):
    audio_data, sample_rate = librosa.load(audio_file)
    normalized_audio = librosa.util.normalize(audio_data)
    hop_length = 512
    n_fft = 2048
    spectrogram = np.abs(librosa.stft(normalized_audio, hop_length=hop_length, n_fft=n_fft))
    energy = np.sum(spectrogram, axis=0)
    return energy


def create_histogram(energy, person_name, output_directory, index):
    histogram, bins = np.histogram(energy, bins=10, range=(0, np.max(energy)))
    plt.bar(bins[:-1], histogram, width=np.diff(bins))
    plt.xlabel("Enerji Aralığı")
    plt.ylabel("Frekans")
    plt.title("Enerji Histogramı")
    output_path = os.path.join(output_directory, f"{person_name}_histogram_{index}.png")
    plt.savefig(output_path)
    plt.close()


data_directory = "C:/Users/Cengiz/Desktop/Sesler"  # Veri seti dizini
output_directory = "C:/Users/Cengiz/Desktop/histogram"  # Histogramların kaydedileceği ana dizin

person_names = os.listdir(data_directory)  # Tüm kişi isimlerini al

for person_name in person_names:
    person_directory = os.path.join(data_directory, person_name)
    output_person_directory = os.path.join(output_directory, person_name)

    if os.path.isdir(person_directory):
        os.makedirs(output_person_directory, exist_ok=True)  # Kişi için histogram klasörünü oluştur

        file_names = os.listdir(person_directory)
        file_names.sort()  # Dosya adlarını sırala

        for index, file_name in enumerate(file_names):
            file_path = os.path.join(person_directory, file_name)
            energy = preprocess_audio(file_path)
            create_histogram(energy, person_name, output_person_directory, index + 1)
