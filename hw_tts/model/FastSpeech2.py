import torch
import torch.nn as nn
from hw_tts.model.blocks import Encoder, Decoder, VarianceAdaptor


def get_mask_from_lengths(lengths, max_len=None):
    if max_len == None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, 1, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask

class FastSpeech2(nn.Module):
    """ FastSpeech """

    def __init__(self, model_config, mel_config):
        super(FastSpeech2, self).__init__()

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(model_config)
        self.decoder = Decoder(model_config)

        self.mel_linear = nn.Linear(model_config.decoder_dim, mel_config.num_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, src_seq, src_pos, mel_pos=None, mel_max_length=None, length_target=None, alpha=1.0, beta=1.0, gamma=1.0):
        enc_output, _ = self.encoder(src_seq, src_pos)
        if self.training:
            va_output, duration_predictor_output, pitch_prediction, energy_prediction  = self.variance_adaptor(enc_output, duration_target=length_target, alpha=alpha, beta=beta, gamma=gamma, mel_max_length=mel_max_length)

            dec_output = self.decoder(va_output, mel_pos)
            mel_output = self.mel_linear(dec_output)
            mel_output = self.mask_tensor(mel_output, mel_pos, mel_max_length)
            return mel_output, duration_predictor_output, pitch_prediction, energy_prediction

        else:
            va_output, decoder_pos = self.variance_adaptor(enc_output, alpha=alpha, beta=beta, gamma=gamma)
            decoder_output = self.decoder(va_output, decoder_pos)
            mel_output = self.mel_linear(decoder_output)
            return mel_output

