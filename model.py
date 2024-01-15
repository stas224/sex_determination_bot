import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F


def preprocess_audio(audio):
    spec = librosa.feature.melspectrogram(y=audio, sr=48000)
    return _preprocess_sample(spec)


def _preprocess_spec(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(float)
    return spec_scaled


def _preprocess_sample(spec):
    max_len = 1000
    spec = _preprocess_spec(spec)[:, :max_len]
    spec = np.pad(spec, [[0, 0], [0, max_len - spec.shape[1]]])
    return torch.tensor(spec.T, dtype=torch.float)


class Model(nn.Module):
    def __init__(self, window_sizes=(3, 4, 5)):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(window_size, 128), padding=(window_size - 1, 0))
            for window_size in window_sizes
        )

        self.fc = nn.Linear(in_features=128 * len(window_sizes), out_features=1)

    def forward(self, x):
      # x - [B, T, F(128)]
      x = x.unsqueeze(1)  # x - [B, 1, T, F(128)] add channel dim
      xs = []
      for conv in self.convs:
          x_new = F.gelu(conv(x))  # x_new - [B, 128, T_new, 1]
          x_new = x_new.squeeze(-1)  # x_new - [B, 128, T_new]
          x_new = F.max_pool1d(x_new, x_new.size(2))  # x_new - [B, 128, 1]
          xs.append(x_new)
      x = torch.cat(xs, 2)  # x - [B, 128, len(window_sizes)]

      x = x.reshape(x.size(0), -1)  # x - [B, 128 * len(window_sizes)]
      logit = self.fc(x)  # x - [B, 1]
      return logit.squeeze(-1)  # x - [B]

