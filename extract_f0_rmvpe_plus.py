
import os, sys, numpy as np, logging
from lib.audio import load_audio
import parselmouth
from lib.rmvpe import RMVPE

logging.getLogger("numba").setLevel(logging.WARNING)

n_part = int(sys.argv[1])
i_part = int(sys.argv[2])
i_gpu = sys.argv[3]
os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
exp_dir = sys.argv[4]
is_half = sys.argv[5]
f = open(f"{exp_dir}/extract_f0_feature_plus.log", "a+")

def printt(strr):
    print(strr)
    f.write(f"{strr}\n")
    f.flush()

class FeatureInputPlus:
    def __init__(self, samplerate=16000, hop_size=160):
        self.fs = samplerate
        self.hop = hop_size

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def compute_f0(self, path, f0_method):
        x = load_audio(path, self.fs)
        p_len = x.shape[0] // self.hop

        if f0_method == "rmvpe+":
            if not hasattr(self, "model_rmvpe_plus"):
                print("Loading RMVPE+ model...")
                self.model_rmvpe_plus = RMVPE("rmvpe_plus.pt", is_half=True, device="cuda")

            # Enhanced F0 extraction with post-processing to reduce artifacts
            f0 = self.model_rmvpe_plus.infer_from_audio(x, thred=0.02)  # Lower threshold for sensitivity
            f0 = self.remove_noise_and_artifacts(f0)  # New noise reduction method
        return f0

    def remove_noise_and_artifacts(self, f0):
        # Simple post-processing to smooth out artifacts
        f0_smoothed = np.where(f0 < self.f0_min, 0, f0)  # Remove frequencies below the minimum
        return np.convolve(f0_smoothed, np.ones(5)/5, mode='same')  # Smooth using a simple moving average
