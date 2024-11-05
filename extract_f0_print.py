import os
import traceback
import sys
import parselmouth
import numpy as np
import logging
import pyworld
import torch
import torch.nn as nn
import scipy.signal as signal
from multiprocessing import Process, Queue
from datetime import datetime

now_dir = os.getcwd()
sys.path.append(now_dir)
from lib.audio import load_audio

logging.getLogger("numba").setLevel(logging.WARNING)

exp_dir = sys.argv[1]
f = open("%s/extract_f0_feature.log" % exp_dir, "a+")

def printt(message):
    """Logging function with timestamp for both console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    f.write(f"{log_message}\n")
    f.flush()

n_p = int(sys.argv[2])
f0method = sys.argv[3]

class EnhancedF0Processor:
    """Advanced F0 processing with multiple enhancement techniques"""
    def __init__(self, hop_length=160):
        self.hop_length = hop_length
        self.f0_statistics = {'mean': None, 'std': None}
        
    def remove_outliers(self, f0, threshold=2.5):
        """Remove statistical outliers from F0 contour"""
        f0_voiced = f0[f0 > 0]
        if len(f0_voiced) > 0:
            mean = np.mean(f0_voiced)
            std = np.std(f0_voiced)
            threshold_up = mean + threshold * std
            threshold_down = mean - threshold * std
            f0_clean = np.where((f0 > threshold_down) & (f0 < threshold_up), f0, 0)
            return f0_clean
        return f0

    def adaptive_smoothing(self, f0, window_range=(3, 7)):
        """Apply adaptive window smoothing based on F0 stability"""
        if len(f0) == 0:
            return f0
            
        f0_diff = np.abs(np.diff(f0))
        window_size = int(np.mean(f0_diff) * (window_range[1] - window_range[0]) + window_range[0])
        window_size = window_size if window_size % 2 == 1 else window_size + 1
        
        return signal.medfilt(f0, kernel_size=min(window_size, len(f0)))

    def interpolate_gaps(self, f0, max_gap_size=20):
        """Interpolate small gaps in F0 contour"""
        if len(f0) == 0:
            return f0
            
        f0_interpolated = f0.copy()
        zero_regions = np.where(f0_interpolated == 0)[0]
        
        if len(zero_regions) == 0:
            return f0_interpolated
            
        # Find continuous zero regions
        gaps = np.split(zero_regions, np.where(np.diff(zero_regions) != 1)[0] + 1)
        
        for gap in gaps:
            if len(gap) > max_gap_size:
                continue
                
            start_idx = max(0, gap[0] - 1)
            end_idx = min(len(f0), gap[-1] + 2)
            
            if start_idx == 0 or end_idx == len(f0):
                continue
                
            start_val = f0[start_idx]
            end_val = f0[end_idx]
            
            if start_val == 0 or end_val == 0:
                continue
                
            interpolated_values = np.linspace(start_val, end_val, len(gap) + 2)[1:-1]
            f0_interpolated[gap] = interpolated_values
            
        return f0_interpolated

class FeatureInput:
    """Main feature extraction class with improved RMVPE+ implementation"""
    def __init__(self, sample_rate=16000, hop_size=160):
        self.fs = sample_rate
        self.hop = hop_size
        
        # F0 extraction parameters
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        
        # Initialize enhanced F0 processor
        self.f0_processor = EnhancedF0Processor(hop_size)
        
        # RMVPE model loading status
        self.rmvpe_loaded = False
        
    def load_rmvpe(self):
        """Load RMVPE model with error handling"""
        if not self.rmvpe_loaded:
            try:
                from lib.rmvpe import RMVPE
                printt("Loading RMVPE model...")
                self.model_rmvpe = RMVPE("rmvpe.pt", is_half=False, device="cpu")
                self.rmvpe_loaded = True
            except Exception as e:
                printt(f"Failed to load RMVPE model: {str(e)}")
                raise

    def compute_f0(self, path, f0_method):
        """Enhanced F0 computation with multiple methods and improved processing"""
        x = load_audio(path, self.fs)
        p_len = x.shape[0] // self.hop

        f0 = None
        error_msgs = []

        try:
            if f0_method in ["rmvpe", "rmvpe+"]:
                self.load_rmvpe()
                f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)

                if f0_method == "rmvpe+":
                    # Enhanced processing pipeline
                    f0 = self.f0_processor.remove_outliers(f0)
                    f0 = self.f0_processor.interpolate_gaps(f0)
                    f0 = self.f0_processor.adaptive_smoothing(f0)

            elif f0_method == "harvest":
                f0, t = pyworld.harvest(
                    x.astype(np.double),
                    fs=self.fs,
                    f0_ceil=self.f0_max,
                    f0_floor=self.f0_min,
                    frame_period=1000 * self.hop / self.fs,
                )
                f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)

            elif f0_method == "dio":
                f0, t = pyworld.dio(
                    x.astype(np.double),
                    fs=self.fs,
                    f0_ceil=self.f0_max,
                    f0_floor=self.f0_min,
                    frame_period=1000 * self.hop / self.fs,
                )
                f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)

            elif f0_method == "pm":
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

                # Pad if necessary
                pad_size = (p_len - len(f0) + 1) // 2
                if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                    f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")

        except Exception as e:
            error_msg = f"Error in F0 extraction ({f0_method}): {str(e)}"
            error_msgs.append(error_msg)
            printt(error_msg)

        # Validate F0 output
        if f0 is None or len(f0) == 0:
            raise ValueError(f"F0 extraction failed. Errors: {'; '.join(error_msgs)}")

        if len(f0) != p_len:
            f0 = np.interp(
                np.linspace(0, len(f0)-1, p_len),
                np.arange(len(f0)),
                f0
            )

        return f0

    def coarse_f0(self, f0):
        """Convert F0 to coarse-grained representation with validation"""
        if f0 is None or len(f0) == 0:
            raise ValueError("Invalid F0 input for coarse conversion")

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
        if f0_coarse.max() > 255 or f0_coarse.min() < 1:
            raise ValueError(f"F0 coarse values out of range: min={f0_coarse.min()}, max={f0_coarse.max()}")
        
        return f0_coarse

    def go(self, paths, f0_method):
        """Process multiple files with enhanced error handling and progress tracking"""
        if len(paths) == 0:
            printt("no-f0-todo")
            return

        printt(f"Starting F0 extraction for {len(paths)} files using {f0_method}")
        n = max(len(paths) // 5, 1)
        
        for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
            try:
                if idx % n == 0:
                    printt(f"Progress: {idx}/{len(paths)} ({idx/len(paths)*100:.1f}%) - {inp_path}")
                
                # Skip if already processed
                if os.path.exists(opt_path1 + ".npy") and os.path.exists(opt_path2 + ".npy"):
                    continue
                
                # Validate input file
                if not os.path.exists(inp_path):
                    printt(f"Input file not found: {inp_path}")
                    continue
                    
                if os.path.getsize(inp_path) == 0:
                    printt(f"Empty input file: {inp_path}")
                    continue

                # Extract and process F0
                featur_pit = self.compute_f0(inp_path, f0_method)
                
                # Validate output
                if featur_pit is None or len(featur_pit) == 0:
                    printt(f"Invalid F0 output for {inp_path}")
                    continue

                # Save results
                np.save(opt_path2, featur_pit, allow_pickle=False)
                coarse_pit = self.coarse_f0(featur_pit)
                np.save(opt_path1, coarse_pit, allow_pickle=False)

            except Exception as e:
                printt(f"Failed to process {inp_path}: {str(e)}\n{traceback.format_exc()}")
                continue

        printt(f"F0 extraction completed for {len(paths)} files")

if __name__ == "__main__":
    printt(f"Starting F0 extraction with arguments: {sys.argv}")
    
    try:
        feature_input = FeatureInput()
        
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
        processes = []
        for i in range(n_p):
            p = Process(target=feature_input.go, args=(paths[i::n_p], f0method))
            processes.append(p)
            p.start()
        
        # Wait for completion
        for p in processes:
            p.join()

        printt("F0 extraction completed successfully")

    except Exception as e:
        printt(f"Fatal error in F0 extraction: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)
