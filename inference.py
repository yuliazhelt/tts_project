import numpy as np
import os
import torch
import text_tools
from tqdm import tqdm
import torchaudio

import hw_tts.model as module_arch
from hw_tts.configs.config import FastSpeechConfig, MelSpectrogramConfig, TrainConfig
from hw_tts.logger import WanDBWriter
from hw_tts.utils import get_WaveGlow

import waveglow

def load_audio(path):
    audio_tensor, sr = torchaudio.load(path)
    audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
    return audio_tensor, sr

def synthesis(model, text, train_config, alpha=1.0, beta=1.0, gamma=1.0):
    # text = np.array(phn)
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().to(train_config.device)
    src_pos = torch.from_numpy(src_pos).long().to(train_config.device)
    
    with torch.no_grad():
        mel = model.forward(sequence, src_pos, alpha=alpha, beta=beta, gamma=gamma)
    return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)


def get_data(train_config):
    tests = [ 
        "I am very happy to see you again!",
        "Durian model is a very good speech synthesis!",
        "When I was twenty, I fell in love with a girl.",
        "I remove attention module in decoder and use average pooling to implement predicting r frames at once",
        "You can not improve your past, but you can improve your future. Once time is wasted, life is wasted.",
        "Death comes to all, but great achievements raise a monument which shall endure until the sun grows old.",
        "Lesha and Ulad are my best friends but they do not respond"
    ]
    data_list = list(text_tools.text_to_sequence(test, train_config.text_cleaners) for test in tests)

    return data_list


def get_test_data(train_config):
    tests = [ 
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space"
    ]
    data_list = list(text_tools.text_to_sequence(test, train_config.text_cleaners) for test in tests)

    return data_list


def main():
    mel_config = MelSpectrogramConfig()
    train_config = TrainConfig()
    model_config = FastSpeechConfig()


    model = module_arch.FastSpeech2(model_config, mel_config)
    model = model.to(train_config.device)
    model.eval()

    checkpoint = torch.load('model_best.pth.tar')
    model.load_state_dict(checkpoint['model'])

    logger = WanDBWriter(train_config)
    data_list = get_test_data(train_config)
    WaveGlow = get_WaveGlow()
    WaveGlow = WaveGlow.cuda()
    speeds = [0.8, 1, 1.2]
    pitches = [0.8, 1, 1.2]
    energies = [0.8, 1, 1.2]
    for speed in speeds:
        for pitch in pitches:
            for energy in energies:
                for i, phn in tqdm(enumerate(data_list)):

                    mel, mel_cuda = synthesis(model, phn, train_config, alpha=speed, beta=pitch, gamma=energy)
                    
                    os.makedirs("results", exist_ok=True)
                    
                    waveglow.inference.inference(
                        mel_cuda, WaveGlow,
                        f"results/speed={speed}_pitch={pitch}_energy={energy}_{i}_waveglow.wav"
                    )

                    audio, sr = load_audio(f"results/speed={speed}_pitch={pitch}_energy={energy}_{i}_waveglow.wav")
                    # logger.add_audio(f"audio_{speed}_{pitch}_{energy}_{i}_waveglow", audio, sample_rate=sr)


if __name__ == "__main__":
    main()