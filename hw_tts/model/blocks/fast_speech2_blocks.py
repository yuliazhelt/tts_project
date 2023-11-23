import torch
import torch.nn as nn
from hw_tts.configs.config import FastSpeechConfig 
import numpy as np
import torch.nn.functional as F

def create_alignment(base_mat, duration_predictor_output):
    N, L = duration_predictor_output.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(duration_predictor_output[i][j]):
                base_mat[i][count+k][j] = 1
            count = count + duration_predictor_output[i][j]
    return base_mat

class Transpose(nn.Module):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        return x.transpose(self.dim_1, self.dim_2)


class VariancePredictor(nn.Module):
    """ Variance (duration/pitch/energy) Predictor """

    def __init__(self, model_config: FastSpeechConfig):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config.encoder_dim
        self.filter_size = model_config.duration_predictor_filter_size
        self.kernel = model_config.duration_predictor_kernel_size
        self.conv_output_size = model_config.duration_predictor_filter_size
        self.dropout = model_config.dropout

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)
            
        out = self.linear_layer(encoder_output)
        out = self.relu(out)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out

class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self, model_config):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)


        output = alignment @ x


        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        duration_predictor_output = self.duration_predictor(x)
        if target is not None:
            lr_output = self.LR(x, target, mel_max_length)
            return lr_output, duration_predictor_output
        else:
            duration_predictor_output = ((duration_predictor_output + 0.5) * alpha).int()
            lr_output = self.LR(x, duration_predictor_output)
            mel_pos = torch.stack([torch.Tensor([i+1 for i in range(lr_output.size(1))])]).long().to(x.device)

            return lr_output, mel_pos
        

class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self, model_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator(model_config)
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)

        self.pitch_bins = nn.Parameter(
            torch.exp(
                torch.linspace(np.log(model_config.f0_min), np.log(model_config.f0_max), model_config.n_bins - 1)
            )
        )
        self.energy_bins = nn.Parameter(
            torch.linspace(model_config.energy_min, model_config.energy_max, model_config.n_bins - 1)
        )
        self.pitch_embedding = nn.Embedding(model_config.n_bins, model_config.encoder_dim)
        self.energy_embedding = nn.Embedding(model_config.n_bins, model_config.encoder_dim)

    def get_pitch_block(self, x, target=None, control=1.0):
        prediction = self.pitch_predictor(x)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = (torch.exp(prediction) - 1) * control
            embedding = self.pitch_embedding(
                torch.bucketize(torch.log1p(prediction), self.pitch_bins)
            )
        return prediction, embedding

    def get_energy_block(self, x, target=None, control=1.0):
        prediction = self.energy_predictor(x)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = (torch.exp(prediction) - 1) * control
            embedding = self.energy_embedding(
                torch.bucketize(torch.log1p(prediction), self.energy_bins)
            )
        return prediction, embedding

    def forward(
        self,
        x,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
        mel_mask=None,
        duration_target=None,
        pitch_target=None,
        energy_target=None,
        mel_max_length=None
    ):
        
        if self.training:
            x, duration_predictor_output = self.length_regulator(x, target=duration_target, alpha=alpha)

            pitch_prediction, pitch_embedding = self.get_pitch_block(
                x, pitch_target, control=beta
            )
            energy_prediction, energy_embedding = self.get_energy_block(
                x, energy_target, control=gamma
            )


            x = x + pitch_embedding + energy_embedding

            return x, duration_predictor_output, pitch_prediction, energy_prediction
        else:
            x, mel_pos = self.length_regulator(x, alpha)
            _, pitch_embedding = self.get_pitch_block(x, control=beta)
            _, energy_embedding = self.get_energy_block(x, control=gamma)
            return x, mel_pos

