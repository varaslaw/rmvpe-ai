import os, traceback, sys, parselmouth
import numpy as np, logging
import pyworld
from multiprocessing import Process
import torch
import torch.nn as nn
import scipy.signal as signal
from datetime import datetime

now_dir = os.getcwd()
sys.path.append(now_dir)
from lib.audio import load_audio

logging.getLogger("numba").setLevel(logging.WARNING)

exp_dir = sys.argv[1]
f = open("%s/extract_f0_feature.log" % exp_dir, "a+")

def printt(strr):
    """Logging function for both console and file output"""
    print(strr)
    f.write("%s\n" % strr)
    f.flush()

n_p = int(sys.argv[2])
f0method = sys.argv[3]

class F0Preprocessor:
    """Advanced F0 preprocessing and enhancement class"""
    def __init__(self, window_length=64, alpha=0.3):
        self.window_length = window_length
        self.alpha = alpha
        
    def median_smoothing(self, f0, window_length=None):
        """Apply median smoothing to F0 contour"""
        if window_length is None:
            window_length = self.window_length
        return signal.medfilt(f0, kernel_size=window_length)

    def adaptive_smoothing(self, f0, threshold=0.1):
        """Apply adaptive smoothing based on F0 variations"""
        smooth_f0 = np.copy(f0)
        diff = np.abs(np.diff(f0))
        mask = diff > (diff.mean() + threshold * diff.std())
        for i in range(1, len(f0)-1):
            if mask[i-1] or mask[i]:
                smooth_f0[i] = np.mean(f0[max(0,i-2):min(len(f0),i+3)])
        return smooth_f0

    def interpolate_zeros(self, f0):
        """Interpolate zero values in F0 contour"""
        nzindex = np.nonzero(f0)[0]
        if len(nzindex) == 0:
            return f0
        f0_interp = np.interp(np.arange(len(f0)), nzindex, f0[nzindex])
        return f0_interp

class RNNSmoother(nn.Module):
    """RNN-based F0 smoother for advanced contour processing"""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.1):
        super(RNNSmoother, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, 1)
        
    def forward(self, x):
        x, _ = self.rnn(x)
        return self.fc(x)

class FeatureInput:
    """Main feature extraction class with improved F0 processing"""
    def __init__(self, samplerate=16000, hop_size=160):
        self.fs = samplerate
        self.hop = hop_size
        
        # F0 extraction parameters
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

        # Initialize processors
        self.f0_preprocessor = F0Preprocessor()
        self.rnn_smoother = RNNSmoother()
        
        # Load RNN model if exists
        model_path = "rnn_smoother.pth"
        if os.path.exists(model_path):
            try:
                self.rnn_smoother.load_state_dict(torch.load(model_path))
                self.rnn_smoother.eval()
            except Exception as e:
                printt(f"Failed to load RNN model: {str(e)}")

    def enhance_f0(self, f0):
        """Enhanced F0 processing pipeline"""
        # Remove zeros and interpolate
        f0 = self.f0_preprocessor.interpolate_zeros(f0)
        
        # Apply median smoothing
        f0 = self.f0_preprocessor.median_smoothing(f0)
        
        # Apply adaptive smoothing
        f0 = self.f0_preprocessor.adaptive_smoothing(f0)
        
        # Apply RNN smoothing if model is available
        try:
            with torch.no_grad():
                f0_tensor = torch.FloatTensor(f0).view(-1, 1, 1)
                f0_smooth = self.rnn_smoother(f0_tensor)
                f0 = f0_smooth.squeeze().numpy()
        except Exception as e:
            printt(f"RNN smoothing skipped: {str(e)}")
        
        return f0

    def compute_f0(self, path, f0_method):
        """Compute F0 with specified method and apply enhancements"""
        x = load_audio(path, self.fs)
        p_len = x.shape[0] // self.hop
        
        # Initialize f0 as None
        f0 = None
        
        try:
            if f0_method == "pm":
                # Praat-based F0 extraction
                time_step = 160 / 16000 * 1000
                f0 = (
                    parselmouth.Sound(x, self.fs)
                    .to_pitch_ac(
                        time_step=time_step / 1000,
                        voicing_threshold=0.6,
                        pitch_floor=self.f0_min,
                        pitch_ceiling=self.f0_max,
                    )
                    .selected_array["frequency"]
                )
                pad_size = (p_len - len(f0) + 1) // 2
                if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                    f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")

            elif f0_method == "harvest":
                # WORLD Harvest F0 extraction
                f0, t = pyworld.harvest(
                    x.astype(np.double),
                    fs=self.fs,
                    f0_ceil=self.f0_max,
                    f0_floor=self.f0_min,
                    frame_period=1000 * self.hop / self.fs,
                )
                f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)

            elif f0_method == "dio":
                # WORLD DIO F0 extraction
                f0, t = pyworld.dio(
                    x.astype(np.double),
                    fs=self.fs,
                    f0_ceil=self.f0_max,
                    f0_floor=self.f0_min,
                    frame_period=1000 * self.hop / self.fs,
                )
                f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)

            elif f0_method == "rmvpe":
                # RMVPE-based F0 extraction
                if not hasattr(self, "model_rmvpe"):
                    from lib.rmvpe import RMVPE
                    printt("Loading RMVPE model")
                    self.model_rmvpe = RMVPE("rmvpe.pt", is_half=False, device="cpu")
                f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)

            elif f0_method == "rmvpe+":
                # Enhanced RMVPE with additional processing
                if not hasattr(self, "model_rmvpe"):
                    from lib.rmvpe import RMVPE
                    printt("Loading RMVPE model")
                    self.model_rmvpe = RMVPE("rmvpe.pt", is_half=False, device="cpu")
                
                # Get base F0
                f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
                
                # Apply enhanced processing
                if f0 is not None:
                    f0 = self.enhance_f0(f0)

        except Exception as e:
            printt(f"Error in F0 computation: {str(e)}\n{traceback.format_exc()}")
            raise

        if f0 is None:
            raise ValueError(f"F0 extraction failed for method: {f0_method}")

        return f0

    def coarse_f0(self, f0):
        """Convert F0 to coarse-grained representation"""
        # Mel-scale conversion
        f0_mel = 1127 * np.log(1 + f0 / 700)
        
        # Scale to bin range
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1
        
        # Clip values
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        
        # Convert to integer bins
        f0_coarse = np.rint(f0_mel).astype(int)
        
        # Verify range
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        
        return f0_coarse

    def go(self, paths, f0_method):
        """Process multiple files with F0 extraction"""
        if len(paths) == 0:
            printt("no-f0-todo")
            return

        printt(f"todo-f0-{len(paths)}")
        n = max(len(paths) // 5, 1)
        
        for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
            try:
                if idx % n == 0:
                    printt(f"f0ing,now-{idx},all-{len(paths)},-{inp_path}")
                
                # Skip if already processed
                if os.path.exists(opt_path1 + ".npy") and os.path.exists(opt_path2 + ".npy"):
                    continue
                
                # Extract and process F0
                featur_pit = self.compute_f0(inp_path, f0_method)
                
                # Save results
                np.save(opt_path2, featur_pit, allow_pickle=False)
                coarse_pit = self.coarse_f0(featur_pit)
                np.save(opt_path1, coarse_pit, allow_pickle=False)
                
            except Exception as e:
                printt(f"f0fail-{idx}-{inp_path}-{traceback.format_exc()}")

if __name__ == "__main__":
    printt(sys.argv)
    featureInput = FeatureInput()
    
    # Setup paths
    paths = []
    inp_root = "%s/1_16k_wavs" % (exp_dir)
    opt_root1 = "%s/2a_f0" % (exp_dir)
    opt_root2 = "%s/2b-f0nsf" % (exp_dir)

    # Create output directories
    os.makedirs(opt_root1, exist_ok=True)
    os.makedirs(opt_root2, exist_ok=True)
    
    # Collect files for processing
    for name in sorted(os.listdir(inp_root)):
        if "spec" in name:
            continue
        inp_path = "%s/%s" % (inp_root, name)
        opt_path1 = "%s/%s" % (opt_root1, name)
        opt_path2 = "%s/%s" % (opt_root2, name)
        paths.append([inp_path, opt_path1, opt_path2])

    # Process files in parallel
    ps = []
    for i in range(n_p):
        p = Process(target=featureInput.go, args=(paths[i::n_p], f0method))
        ps.append(p)
        p.start()
    
    # Wait for completion
    for i in range(n_p):
        ps[i].join()
