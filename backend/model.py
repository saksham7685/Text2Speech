import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

# --- Neural net for pitch/rate selection ---
class PitchRateNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Sigmoid(),  # map to (0,1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

_pitch_rate_nn = PitchRateNN()


def _text_features(text: str) -> torch.Tensor:
    # Simple features: [length_normalized, vowel_ratio, punctuation_ratio]
    t = text or ""
    n = len(t)
    if n == 0:
        return torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    vowels = sum(ch.lower() in "aeiou" for ch in t)
    punct = sum(ch in ",.;!?" for ch in t)
    length_norm = min(n / 200.0, 1.0)
    vowel_ratio = vowels / n
    punct_ratio = punct / n
    return torch.tensor([[length_norm, vowel_ratio, punct_ratio]], dtype=torch.float32)


def predict_params(text: str) -> Tuple[float, float]:
    """Return (pitch, rate) in browser ranges [0.5, 2.0].
    This uses a tiny untrained NN for demo purposes; values are deterministic per text.
    """
    with torch.no_grad():
        x = _text_features(text)
        y = _pitch_rate_nn(x)[0]  # in (0,1)
        # Map to [0.5, 2.0]
        pitch = 0.5 + float(y[0]) * 1.5
        rate = 0.5 + float(y[1]) * 1.5
        return round(pitch, 2), round(rate, 2)


# --- Tiny neural-like classifier for language demo ---
# We'll classify among 4 labels: en, hi, es, hinglish based on simple features
_LANG_LABELS = ["en", "hi", "es", "hinglish"]

class TinyLangNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 6 simple features -> 4 classes
        # features: [latin_ratio, devanagari_ratio, tilde_count, exclaim_count, length_norm, hinglish_kw_ratio]
        self.fc = nn.Linear(6, 4)
        # Initialize with handcrafted weights to make it semi-reasonable
        with torch.no_grad():
            W = torch.tensor([
                # en: prefers latin, dislikes devanagari, neutral others
                [ 2.0, -2.0, 0.2, 0.3, 0.1, -0.2],
                # hi: prefers devanagari strongly
                [-2.0,  3.0, 0.0, 0.0, 0.1, -0.5],
                # es: prefers latin and tilde presence
                [ 1.5, -2.0, 1.2, 0.0, 0.1, -0.2],
                # hinglish: latin + hinglish keywords, dislikes devanagari
                [ 1.5, -1.5, 0.0, 0.1, 0.1,  2.5],
            ], dtype=torch.float32)
            b = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
            self.fc.weight.copy_(W)
            self.fc.bias.copy_(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

_lang_nn = TinyLangNN()


def _lang_features(text: str) -> torch.Tensor:
    t = text or ""
    n = max(len(t), 1)
    latin = sum('a' <= ch.lower() <= 'z' or ch in "áéíóúüñ" for ch in t)
    dev = sum('\u0900' <= ch <= '\u097F' for ch in t)  # Devanagari block
    tilde = t.count('ñ') + t.count('Ñ')
    exclam = t.count('!')
    latin_ratio = latin / n
    dev_ratio = dev / n
    length_norm = min(n / 200.0, 1.0)
    # Very small Hinglish lexicon (romanized Hindi/common code-mix tokens)
    tokens = [tok for tok in ''.join(ch if ch.isalnum() else ' ' for ch in t.lower()).split() if tok]
    hinglish_kw = {
        'hai','hain','nahi','haan','kyu','kyun','kya','kaise','kab','kal','bhai','dost','bhen','acha','achha','accha','theek','thik','haanji','yaar','sab','bahut','thoda','zyada'
    }
    kw_matches = sum(tok in hinglish_kw for tok in tokens)
    hinglish_kw_ratio = (kw_matches / max(len(tokens), 1)) if tokens else 0.0
    feats = torch.tensor([[latin_ratio, dev_ratio, float(tilde), float(exclam), length_norm, hinglish_kw_ratio]], dtype=torch.float32)
    return feats


def detect_language(text: str) -> Tuple[str, Dict[str, float]]:
    with torch.no_grad():
        x = _lang_features(text)
        logits = _lang_nn(x)
        probs = F.softmax(logits, dim=-1)[0]
        scores = {lbl: round(float(probs[i]), 3) for i, lbl in enumerate(_LANG_LABELS)}
        lang = _LANG_LABELS[int(torch.argmax(probs))]
        return lang, scores
