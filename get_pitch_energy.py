
import pyworld as pw
import torchaudio
import torch
import numpy as np
import os
from tqdm import tqdm

def get_pitch(wav_path, mel):
    """
    pyworld expectes wav in type double(float64)
    """
    audio, sample_rate = torchaudio.load(wav_path)
    audio = audio.to(torch.float64).numpy().sum(axis=0)
    f0, t = pw.dio(
        audio.astype(np.float64),
        sample_rate,
        frame_period=(audio.shape[0] / sample_rate * 1000) / mel.shape[0],
    )
    f0 = pw.stonemask(audio, f0, t, sample_rate)[:mel.shape[0]]

    return f0

def get_energy(mel):
    e = np.linalg.norm(mel, axis=-1)
    return e

def main():

    energy_path = os.path.join("data", "energy")
    pitch_path = os.path.join("data", "pitch")

    for _, _, files in os.walk("data/LJSpeech-1.1/wavs/"):
        for id, file_name in tqdm(enumerate(sorted(files))):
            wavpath = os.path.join("data/LJSpeech-1.1/wavs/", file_name)
            mel = np.load("mels/ljspeech-mel-%05d.npy" % (id+1))
            pitch = get_pitch(wavpath, mel) # [T, ] T = Number of frames
            energy = get_energy(mel)  # [T, ]

            np.save("{}/{}.npy".format(energy_path, id), energy, allow_pickle=False)
            np.save("{}/{}.npy".format(pitch_path, id), pitch, allow_pickle=False)

if __name__ == "__main__":
    main()
