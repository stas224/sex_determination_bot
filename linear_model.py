import functools

import numpy as np
import librosa


def preprocess_spec(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(float)
    return spec_scaled


def extract_features(audio):
    sr = 48000
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
    spec = preprocess_spec(spectrogram)
    functions = [
        np.mean,
        np.median,
        np.std
    ]
    for quantile in [0.05, 0.25, 0.75, 0.95]:
        functions.append(functools.partial(np.quantile, q=quantile))
    res = []
    for f in functions:
        res.extend(f(spec, axis=1))

    return np.array(res)

