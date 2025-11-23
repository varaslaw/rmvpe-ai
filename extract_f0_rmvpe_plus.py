import os, traceback, sys
import numpy as np
from scipy.ndimage import median_filter

now_dir = os.getcwd()
sys.path.append(now_dir)

from lib.audio import load_audio
from lib.rmvpe import RMVPE

n_part = int(sys.argv[1])
i_part = int(sys.argv[2])
i_gpu = sys.argv[3]
os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
exp_dir = sys.argv[4]
is_half = sys.argv[5]

log_path = os.path.join(exp_dir, "extract_f0_feature_plus.log")
f = open(log_path, "a+", encoding="utf-8")


def printt(msg):
    print(msg)
    f.write(f"{msg}\n")
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

        printt("Loading RMVPE model (rmvpe+ mode)")
        self.model_rmvpe = RMVPE("rmvpe.pt", is_half=True, device="cuda")

    def compute_f0(self, path):
        x = load_audio(path, self.fs)
        # Base RMVPE extraction
        f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)

        # RMVPE+ enhancement: median filtering + interpolation
        f0_safe = f0.astype(np.float32)
        f0_safe[f0_safe <= 0] = np.nan
        
        # Median filter with window size 5
        f0_med = median_filter(f0_safe, size=5, mode="nearest")

        # Interpolate missing values
        n = len(f0_med)
        idx = np.arange(n)
        mask = ~np.isnan(f0_med)
        if mask.any():
            f0_interp = np.interp(idx, idx[mask], f0_med[mask])
        else:
            f0_interp = np.zeros_like(f0_med)

        return f0_interp

    def coarse_f0(self, f0):
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1

        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        return f0_coarse

    def go(self, paths):
        if len(paths) == 0:
            printt("no-f0-todo")
            return
        printt(f"Processing {len(paths)} files with rmvpe+")
        n = max(len(paths) // 5, 1)
        for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
            try:
                if idx % n == 0:
                    printt(f"f0+ progress: {idx}/{len(paths)} - {inp_path}")
                if os.path.exists(opt_path1 + ".npy") and os.path.exists(
                    opt_path2 + ".npy"
                ):
                    continue
                f0_cont = self.compute_f0(inp_path)
                np.save(opt_path2, f0_cont, allow_pickle=False)
                coarse = self.coarse_f0(f0_cont)
                np.save(opt_path1, coarse, allow_pickle=False)
            except Exception:
                printt(f"f0+ FAILED at {idx} - {inp_path}\n{traceback.format_exc()}")


if __name__ == "__main__":
    printt(str(sys.argv))
    featureInput = FeatureInputPlus()
    paths = []
    inp_root = f"{exp_dir}/1_16k_wavs"
    opt_root1 = f"{exp_dir}/2a_f0"
    opt_root2 = f"{exp_dir}/2b-f0nsf"

    os.makedirs(opt_root1, exist_ok=True)
    os.makedirs(opt_root2, exist_ok=True)

    for name in sorted(os.listdir(inp_root)):
        inp_path = f"{inp_root}/{name}"
        if "spec" in inp_path:
            continue
        opt_path1 = f"{opt_root1}/{name}"
        opt_path2 = f"{opt_root2}/{name}"
        paths.append([inp_path, opt_path1, opt_path2])

    try:
        featureInput.go(paths[i_part::n_part])
    except Exception:
        printt(f"RMVPE+ extraction FAILED:\n{traceback.format_exc()}")
