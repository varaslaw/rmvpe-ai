import os
import traceback
import sys
import parselmouth

now_dir = os.getcwd()
sys.path.append(now_dir)
from lib.audio import load_audio
import pyworld
import numpy as np
import logging

logging.getLogger("numba").setLevel(logging.WARNING)

n_part = int(sys.argv[1])
i_part = int(sys.argv[2])
i_gpu = sys.argv[3]
os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
exp_dir = sys.argv[4]
is_half = sys.argv[5]
f = open(f"{exp_dir}/extract_f0_feature.log", "a+")

def printt(strr):
    """Prints to the console and writes to the log file."""
    print(strr)
    f.write(f"{strr}\n")
    f.flush()

class FeatureInput(object):
    def __init__(self, samplerate=16000, hop_size=160):
        self.fs = samplerate  # Sampling rate
        self.hop = hop_size  # Hop size for F0 analysis

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def compute_f0(self, path, f0_method):
        """Computes the F0 (pitch) using different methods."""
        x = load_audio(path, self.fs)
        p_len = x.shape[0] // self.hop

        if f0_method == "rmvpe":
            if not hasattr(self, "model_rmvpe"):
                from lib.rmvpe import RMVPE
                print("Loading RMVPE model")
                # Initialize RMVPE model
                self.model_rmvpe = RMVPE("rmvpe.pt", is_half=True, device="cuda")
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        elif f0_method == "rmvpe+":
            # RMVPE+ method with additional processing
            f0 = self.compute_f0_rmvpe_plus(x)
        else:
            raise ValueError(f"Unsupported F0 extraction method: {f0_method}")

        return f0

    def compute_f0_rmvpe_plus(self, audio):
        """Enhanced F0 extraction using RMVPE+ with additional processing."""
        if not hasattr(self, "model_rmvpe"):
            from lib.rmvpe import RMVPE
            print("Loading RMVPE model")
            self.model_rmvpe = RMVPE("rmvpe.pt", is_half=True, device="cuda")

        # Use RMVPE model to extract F0
        f0 = self.model_rmvpe.infer_from_audio(audio, thred=0.03)

        # Additional processing: smoothing, noise reduction, etc.
        f0 = self.advanced_smoothing(f0)  # Smooth F0 values to remove abrupt changes
        return f0

    def advanced_smoothing(self, f0):
        """Smooths F0 values to reduce abrupt jumps."""
        window_size = 5
        smoothed_f0 = np.convolve(f0, np.ones(window_size) / window_size, mode="same")
        return smoothed_f0

    def coarse_f0(self, f0):
        """Converts continuous F0 values to discrete bins."""
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1

        # Ensure values are within range
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse

    def go(self, paths, f0_method):
        """Processes audio paths and extracts F0 features."""
        if not paths:
            printt("no-f0-todo")
        else:
            printt(f"todo-f0-{len(paths)}")
            n = max(len(paths) // 5, 1)  # Limit log output to 5 messages per process
            for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
                try:
                    if idx % n == 0:
                        printt(f"f0ing,now-{idx},all-{len(paths)},-{inp_path}")
                    if os.path.exists(opt_path1 + ".npy") and os.path.exists(opt_path2 + ".npy"):
                        continue
                    # Compute F0
                    featur_pit = self.compute_f0(inp_path, f0_method)
                    # Save features
                    np.save(opt_path2, featur_pit, allow_pickle=False)
                    coarse_pit = self.coarse_f0(featur_pit)
                    np.save(opt_path1, coarse_pit, allow_pickle=False)
                except Exception as e:
                    printt(f"f0fail-{idx}-{inp_path}-{traceback.format_exc()}")

if __name__ == "__main__":
    # Initialize feature input
    printt(sys.argv)
    featureInput = FeatureInput()
    paths = []
    inp_root = f"{exp_dir}/1_16k_wavs"
    opt_root1 = f"{exp_dir}/2a_f0"
    opt_root2 = f"{exp_dir}/2b-f0nsf"

    # Create directories if they do not exist
    os.makedirs(opt_root1, exist_ok=True)
    os.makedirs(opt_root2, exist_ok=True)

    # Collect paths of audio files
    for name in sorted(os.listdir(inp_root)):
        inp_path = f"{inp_root}/{name}"
        if "spec" in inp_path:
            continue
        opt_path1 = f"{opt_root1}/{name}"
        opt_path2 = f"{opt_root2}/{name}"
        paths.append([inp_path, opt_path1, opt_path2])

    try:
        # Execute the F0 extraction
        featureInput.go(paths[i_part::n_part], "rmvpe+")
    except Exception as e:
        printt(f"f0_all_fail-{traceback.format_exc()}")
