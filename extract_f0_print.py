def compute_f0(self, path, f0_method):
    """
    Enhanced F0 computation with failsafe mechanisms
    """
    x = load_audio(path, self.fs)
    p_len = x.shape[0] // self.hop
    
    # Initialize f0 as zeros array of correct length
    f0 = np.zeros(p_len, dtype=np.float32)
    
    try:
        # Base F0 extraction
        if f0_method in ["rmvpe", "rmvpe+"]:
            if not hasattr(self, "model_rmvpe"):
                from lib.rmvpe import RMVPE
                printt("Loading RMVPE model")
                self.model_rmvpe = RMVPE("rmvpe.pt", is_half=False, device="cpu")
            
            temp_f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
            
            # Validate and pad if necessary
            if temp_f0 is not None and len(temp_f0) > 0:
                if len(temp_f0) < p_len:
                    temp_f0 = np.pad(temp_f0, (0, p_len - len(temp_f0)), mode='constant')
                elif len(temp_f0) > p_len:
                    temp_f0 = temp_f0[:p_len]
                f0 = temp_f0
            
            # Additional processing for rmvpe+
            if f0_method == "rmvpe+" and np.any(f0 != 0):
                # 1. Remove outliers
                mean_f0 = np.mean(f0[f0 > 0])
                std_f0 = np.std(f0[f0 > 0])
                f0 = np.where((f0 > mean_f0 + 3 * std_f0) | (f0 < mean_f0 - 3 * std_f0), 0, f0)
                
                # 2. Interpolate gaps
                nzeros = np.nonzero(f0)[0]
                if len(nzeros) > 0:
                    f0_interp = np.interp(
                        np.arange(len(f0)),
                        nzeros,
                        f0[nzeros]
                    )
                    # Only fill small gaps (less than 30 frames)
                    gaps = np.where(f0 == 0)[0]
                    for gap in gaps:
                        if np.min(np.abs(nzeros - gap)) < 30:
                            f0[gap] = f0_interp[gap]
                
                # 3. Median filtering
                f0 = signal.medfilt(f0, kernel_size=3)
                
                # 4. Frequency constraints
                f0 = np.clip(f0, self.f0_min, self.f0_max)

        elif f0_method == "harvest":
            _f0, t = pyworld.harvest(
                x.astype(np.double),
                fs=self.fs,
                f0_ceil=self.f0_max,
                f0_floor=self.f0_min,
                frame_period=1000 * self.hop / self.fs,
            )
            f0 = pyworld.stonemask(x.astype(np.double), _f0, t, self.fs)
            
        elif f0_method == "dio":
            _f0, t = pyworld.dio(
                x.astype(np.double),
                fs=self.fs,
                f0_ceil=self.f0_max,
                f0_floor=self.f0_min,
                frame_period=1000 * self.hop / self.fs,
            )
            f0 = pyworld.stonemask(x.astype(np.double), _f0, t, self.fs)
            
        elif f0_method == "pm":
            time_step = 160 / 16000 * 1000
            _f0 = (
                parselmouth.Sound(x, self.fs)
                .to_pitch_ac(
                    time_step=time_step / 1000,
                    voicing_threshold=0.6,
                    pitch_floor=self.f0_min,
                    pitch_ceiling=self.f0_max,
                )
                .selected_array["frequency"]
            )
            pad_size = (p_len - len(_f0) + 1) // 2
            f0 = np.pad(_f0, [[pad_size, p_len - len(_f0) - pad_size]], mode="constant")
            
    except Exception as e:
        printt(f"Error in primary F0 extraction for {f0_method}: {str(e)}")
        # Try fallback methods if primary method fails
        try:
            # Fallback to harvest as it's generally most reliable
            _f0, t = pyworld.harvest(
                x.astype(np.double),
                fs=self.fs,
                f0_ceil=self.f0_max,
                f0_floor=self.f0_min,
                frame_period=1000 * self.hop / self.fs,
            )
            f0 = pyworld.stonemask(x.astype(np.double), _f0, t, self.fs)
        except Exception as e2:
            printt(f"Fallback F0 extraction also failed: {str(e2)}")

    # Final validation
    if f0 is None or len(f0) == 0:
        printt("F0 extraction failed, returning zeros")
        return np.zeros(p_len, dtype=np.float32)
        
    if len(f0) != p_len:
        # Ensure correct length
        f0 = np.interp(
            np.linspace(0, len(f0)-1, p_len),
            np.arange(len(f0)),
            f0
        )

    return f0
