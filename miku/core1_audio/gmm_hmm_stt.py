"""
Classical GMM-HMM Speech Recognition Engine.
Extracts MFCC features, scores acoustic frames with Gaussian states,
and decodes using the Viterbi algorithm.
"""
import numpy as np
from scipy.fftpack import dct
from typing import List, Dict, Tuple, Optional

class GMMHMMModel:
    def __init__(self, sample_rate: int = 16000, num_cepstrum: int = 13, num_filters: int = 26):
        self.sample_rate = sample_rate
        self.num_cepstrum = num_cepstrum
        self.num_filters = num_filters
        self.vocabulary: Dict[str, Dict] = {}
        self._init_default_lexicon()

    def _hz_to_mel(self, hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def extract_mfcc(self, signal: np.ndarray, frame_len_ms: float = 25.0, frame_step_ms: float = 10.0) -> np.ndarray:
        """
        Extracts 13 MFCC features with delta and delta-delta coefficients (39-dim total).
        """
        signal = np.asarray(signal, dtype=np.float32)
        if len(signal) == 0:
            return np.empty((0, self.num_cepstrum * 3), dtype=np.float32)

        # Pre-emphasis
        pre_emphasis = 0.97
        emphasized = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

        frame_len = int(round(frame_len_ms * self.sample_rate / 1000.0))
        frame_step = int(round(frame_step_ms * self.sample_rate / 1000.0))
        sig_len = len(emphasized)

        if sig_len < frame_len:
            pad_len = frame_len - sig_len
            emphasized = np.pad(emphasized, (0, pad_len), mode="constant")
            sig_len = len(emphasized)

        num_frames = 1 + int(np.floor((sig_len - frame_len) / frame_step))
        indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + np.tile(
            np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)
        ).T
        frames = emphasized[indices.astype(np.int32, copy=False)]
        frames *= np.hamming(frame_len)

        # FFT & Power spectrum
        nfft = 512
        mag_frames = np.absolute(np.fft.rfft(frames, nfft))
        pow_frames = (1.0 / nfft) * (mag_frames ** 2)

        # Mel Filterbank
        low_mel = self._hz_to_mel(0)
        high_mel = self._hz_to_mel(self.sample_rate / 2)
        mel_points = np.linspace(low_mel, high_mel, self.num_filters + 2)
        hz_points = self._mel_to_hz(mel_points)
        bin_points = np.floor((nfft + 1) * hz_points / self.sample_rate).astype(int)

        fbank = np.zeros((self.num_filters, int(nfft / 2 + 1)), dtype=np.float32)
        for m in range(1, self.num_filters + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1] + 1e-8)
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m] + 1e-8)

        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 20 * np.log10(filter_banks)

        # DCT
        raw_mfcc = dct(filter_banks, type=2, axis=1, norm="ortho")[:, : self.num_cepstrum]
        
        # Delta and Delta-Delta
        delta1 = self._compute_deltas(raw_mfcc)
        delta2 = self._compute_deltas(delta1)
        return np.hstack((raw_mfcc, delta1, delta2))

    def _compute_deltas(self, feat: np.ndarray, N: int = 2) -> np.ndarray:
        num_frames = feat.shape[0]
        if num_frames == 0:
            return feat
        deltas = np.zeros_like(feat)
        for t in range(num_frames):
            denom = 0.0
            num = np.zeros(feat.shape[1], dtype=np.float32)
            for n in range(1, N + 1):
                idx_pos = min(t + n, num_frames - 1)
                idx_neg = max(t - n, 0)
                num += n * (feat[idx_pos] - feat[idx_neg])
                denom += 2 * (n ** 2)
            deltas[t] = num / denom
        return deltas

    def _init_default_lexicon(self):
        """
        Initializes baseline phonetic HMM templates for core system commands.
        """
        core_commands = [
            "open notepad",
            "close window",
            "volume up",
            "volume down",
            "plan my day",
            "click red icon",
            "look at camera",
            "take screenshot",
            "status report",
            "stop automation",
        ]
        for cmd in core_commands:
            # Synthetic 3-state HMM prototype for each command
            self.vocabulary[cmd] = self._create_command_hmm(cmd)

    def _create_command_hmm(self, text: str) -> Dict:
        """
        Generates deterministic acoustic states for a command phrase based on phonetic hash.
        """
        words = text.split()
        num_states = max(3, len(words) * 3)
        np.random.seed(abs(hash(text)) % (2**31))
        means = np.random.randn(num_states, 39).astype(np.float32) * 2.0
        covars = np.ones((num_states, 39), dtype=np.float32) * 1.5
        trans = np.zeros((num_states, num_states), dtype=np.float32)
        for i in range(num_states):
            trans[i, i] = 0.6  # self-loop
            if i + 1 < num_states:
                trans[i, i + 1] = 0.4
            else:
                trans[i, i] = 1.0
        return {
            "means": means,
            "covars": covars,
            "trans": trans,
            "num_states": num_states
        }

    def train_phrase(self, phrase: str, audio_samples: List[np.ndarray]):
        """
        Enrolls or fine-tunes a phrase with user-provided audio samples.
        """
        features_list = [self.extract_mfcc(s) for s in audio_samples if len(s) > 0]
        if not features_list:
            return
        all_feats = np.vstack(features_list)
        num_states = max(3, len(phrase.split()) * 3)
        # Partition features evenly across states
        chunks = np.array_split(all_feats, num_states)
        means = np.array([chunk.mean(axis=0) if len(chunk) > 0 else np.zeros(39) for chunk in chunks])
        covars = np.array([chunk.var(axis=0) + 1e-4 if len(chunk) > 1 else np.ones(39) for chunk in chunks])

        trans = np.zeros((num_states, num_states), dtype=np.float32)
        for i in range(num_states):
            trans[i, i] = 0.6
            if i + 1 < num_states:
                trans[i, i + 1] = 0.4
            else:
                trans[i, i] = 1.0

        self.vocabulary[phrase.lower()] = {
            "means": means.astype(np.float32),
            "covars": covars.astype(np.float32),
            "trans": trans.astype(np.float32),
            "num_states": num_states
        }

    def viterbi_score(self, mfcc: np.ndarray, hmm: Dict) -> float:
        """
        Computes the log-likelihood of observation sequence under HMM using Viterbi decoding.
        """
        T = mfcc.shape[0]
        N = hmm["num_states"]
        if T == 0:
            return -float("inf")

        means = hmm["means"]
        covars = hmm["covars"]
        trans = hmm["trans"]

        # Log emissions: b_j(o_t)
        log_emiss = np.zeros((T, N), dtype=np.float32)
        for j in range(N):
            diff = mfcc - means[j]
            var = covars[j]
            log_emiss[:, j] = -0.5 * np.sum(np.log(2 * np.pi * var) + (diff ** 2) / var, axis=1)

        # Viterbi trellis: V[t, j]
        V = np.full((T, N), -float("inf"), dtype=np.float32)
        V[0, 0] = log_emiss[0, 0]  # Start at state 0

        for t in range(1, T):
            for j in range(N):
                # Only transitions from previous valid states
                prev_scores = V[t - 1, :j + 1] + np.log(trans[:j + 1, j] + 1e-12)
                V[t, j] = np.max(prev_scores) + log_emiss[t, j]

        return float(V[T - 1, N - 1] / T)  # Length-normalized log-likelihood

    def decode(self, audio: np.ndarray) -> Tuple[str, float]:
        """
        Decodes input audio signal into recognized phrase and confidence score.
        Returns ("" or recognized phrase, confidence in [0.0, 1.0]).
        """
        if len(audio) == 0:
            return "", 0.0

        # Energy check / silence gate
        rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
        if rms < 0.005:  # Pure silence or background noise
            return "", 0.0

        mfcc = self.extract_mfcc(audio)
        if len(mfcc) < 3:
            return "", 0.0

        best_phrase = ""
        best_score = -float("inf")
        scores = {}

        for phrase, hmm in self.vocabulary.items():
            score = self.viterbi_score(mfcc, hmm)
            scores[phrase] = score
            if score > best_score:
                best_score = score
                best_phrase = phrase

        # Convert relative log-likelihoods to confidence score via softmax
        val_list = np.array(list(scores.values()))
        val_list = val_list[np.isfinite(val_list)]
        if len(val_list) == 0:
            return "", 0.0

        max_v = np.max(val_list)
        exp_vals = np.exp(val_list - max_v)
        confidence = float(np.exp(best_score - max_v) / (np.sum(exp_vals) + 1e-9))

        return best_phrase, confidence

    @staticmethod
    def compute_wer(reference: str, hypothesis: str) -> float:
        """
        Computes standard Word Error Rate (WER) via Levenshtein distance.
        """
        r = reference.lower().split()
        h = hypothesis.lower().split()
        d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
        for i in range(len(r) + 1):
            d[i, 0] = i
        for j in range(len(h) + 1):
            d[0, j] = j
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                if r[i - 1] == h[j - 1]:
                    d[i, j] = d[i - 1, j - 1]
                else:
                    d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + 1)
        return float(d[len(r), len(h)]) / max(len(r), 1)
