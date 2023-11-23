import warnings
from tqdm import tqdm
import os

from functools import partial

import numpy as np
import torch.nn as nn
import torch
import waveglow

import hw_tts.loss as module_loss
import hw_tts.model as module_arch
import hw_tts.datasets as module_dataset
from hw_tts.collate_fn.collate import collate_fn_tensor, get_data_to_buffer
from hw_tts.configs.config import MelSpectrogramConfig, FastSpeechConfig, TrainConfig
from hw_tts.logger import WanDBWriter
from hw_tts.utils import get_WaveGlow

from inference import get_data, load_audio, synthesis

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR


warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main():
    
    mel_config = MelSpectrogramConfig()
    model_config = FastSpeechConfig()
    train_config = TrainConfig()

    buffer = get_data_to_buffer(train_config)

    dataset = module_dataset.BufferDataset(buffer)

    training_loader = DataLoader(
        dataset,
        batch_size=train_config.batch_expand_size * train_config.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn_tensor, batch_expand_size=train_config.batch_expand_size),
        drop_last=True,
        num_workers=4
    )

    # build model architecture, then print to console
    model = module_arch.FastSpeech2(model_config, mel_config)
    model = model.to(train_config.device)

    fastspeech_loss = module_loss.FastSpeech2Loss()
    current_step = 0

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9
    )

    print(len(training_loader))
    scheduler = OneCycleLR(optimizer, **{
        "steps_per_epoch": len(training_loader) * train_config.batch_expand_size,
        "epochs": train_config.epochs,
        "anneal_strategy": "cos",
        "max_lr": train_config.learning_rate,
        "pct_start": 0.1
    })


    tqdm_bar = tqdm(total=train_config.epochs * len(training_loader) * train_config.batch_expand_size - current_step)

    writer = WanDBWriter(train_config) 
    WaveGlow = get_WaveGlow()
    WaveGlow = WaveGlow.cuda()

    data_list = get_data(train_config)


    for epoch in range(1):
        model.train()
        for i, batches in tqdm(enumerate(training_loader)):
            # real batch start here
            for j, db in enumerate(batches):
                current_step += 1
                tqdm_bar.update(1)
                
                writer.set_step(current_step)

                # Get Data
                character = db["text"].long().to(train_config.device)
                mel_target = db["mel_target"].float().to(train_config.device)
                duration = db["duration"].int().to(train_config.device)
                pitch_target = db["pitch"].int().to(train_config.device)
                energy_target = db["energy"].int().to(train_config.device)


                mel_pos = db["mel_pos"].long().to(train_config.device)
                src_pos = db["src_pos"].long().to(train_config.device)
                max_mel_len = db["mel_max_len"]


                # Forward
                mel_output, duration_predictor_output, pitch_prediction, energy_prediction = model(character,
                                                            src_pos,
                                                            mel_pos=mel_pos,
                                                            mel_max_length=max_mel_len,
                                                            length_target=duration)

                # Calc Loss
                mel_loss, duration_loss, pitch_loss, energy_loss = fastspeech_loss(mel_output,
                                                        mel_target,
                                                        duration_predictor_output,
                                                        duration,
                                                        pitch_prediction, 
                                                        pitch_target, 
                                                        energy_prediction,
                                                        energy_target)
                total_loss = mel_loss + duration_loss + pitch_loss + energy_loss

                # writer
                t_l = total_loss.detach().cpu().numpy()
                m_l = mel_loss.detach().cpu().numpy()
                d_l = duration_loss.detach().cpu().numpy()

                writer.add_scalar("duration_loss", d_l)
                writer.add_scalar("mel_loss", m_l)
                writer.add_scalar("total_loss", t_l)

                # Backward
                total_loss.backward()

                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(
                    model.parameters(), train_config.grad_clip_thresh)
                
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                if current_step % train_config.save_step == 0:
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                    )}, os.path.join(train_config.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                    print("save model at step %d ..." % current_step)

        model.eval()

        for speed in [0.8, 1., 1.3]:
            for i, phn in tqdm(enumerate(data_list)):

                mel, mel_cuda = synthesis(model, phn, train_config, speed)
                
                os.makedirs("results", exist_ok=True)
                
                waveglow.inference.inference(
                    mel_cuda, WaveGlow,
                    
                    f"results/s={speed}_{i}_waveglow.wav"
                )
                audio, sr = load_audio(f"results/s={speed}_{i}_waveglow.wav")
                writer.add_audio(f"audio_{speed}_{i}_waveglow", audio, sample_rate=sr)


if __name__ == "__main__":
    main()
