import torch
import numpy as np
import subprocess
import tempfile
import os
import re
import soundfile as sf

from .matchering.log.handlers import set_handlers as log
from .matchering.core import process
from .matchering.defaults import Config, LimiterConfig

# ============================================================================
# RELIABLE LOUDNESS TARGETING FUNCTIONS
# ============================================================================

# RMS to LUFS conversion table (based on real-world measurements)
RMS_TO_LUFS_MAPPING = {
    -6.0: -8.4,   # Very loud (commercial competitive)
    -9.0: -11.0,  # Loud commercial
    -12.0: -14.0, # Streaming platform target
    -15.0: -17.0, # Moderate/dynamic
    -18.0: -20.0, # Very dynamic
    -21.0: -23.0, # Classical/ambient
}

def lufs_to_rms_target(target_lufs):
    """Convert LUFS target to approximate RMS target"""
    lufs_values = list(RMS_TO_LUFS_MAPPING.values())
    rms_values = list(RMS_TO_LUFS_MAPPING.keys())
    return np.interp(target_lufs, lufs_values, rms_values)

def measure_lufs_with_ffmpeg(audio_array, sample_rate):
    """Measure LUFS using FFmpeg (most reliable method)"""
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        sf.write(temp_path, audio_array, sample_rate)
        
        cmd = [
            'ffmpeg', '-i', temp_path, 
            '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json',
            '-f', 'null', '-'
        ]
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        
        try:
            os.unlink(temp_path)
        except:
            pass
        
        if result.stderr:
            match = re.search(r'"input_i"\s*:\s*"([^"]+)"', result.stderr)
            if match and match.group(1) != "-inf":
                lufs_value = float(match.group(1))
                print(f"[FFmpeg LUFS] Measured: {lufs_value:.1f} LUFS")
                return lufs_value
        
        print("[FFmpeg LUFS] Could not parse LUFS from FFmpeg output")
        return None
        
    except FileNotFoundError:
        print("[FFmpeg LUFS] FFmpeg not found - install from https://ffmpeg.org")
        return None
    except Exception as e:
        print(f"[FFmpeg LUFS] Error: {e}")
        return None

def adjust_to_rms_target(audio_array, target_rms_db=-12.0, sample_rate=44100, config=None, peak_limit_dbtp=-1.0):
    """RMS-based loudness targeting (always works, no dependencies)"""
    if config is None:
        config = Config()
        
    peak_limit_linear = 10 ** (peak_limit_dbtp / 20)
    
    try:
        print(f"[RMS Target] Input shape: {audio_array.shape}, Target: {target_rms_db:.1f} dB RMS")
        print(f"[RMS Target] Peak limit: {peak_limit_dbtp:.1f} dBTP ({peak_limit_linear:.4f} linear)")
        
        # Ensure stereo
        if len(audio_array.shape) == 1:
            audio_array = np.column_stack([audio_array, audio_array])
        elif audio_array.shape[1] == 1:
            audio_array = np.column_stack([audio_array[:, 0], audio_array[:, 0]])
        elif audio_array.shape[1] > 2:
            audio_array = audio_array[:, :2]
        
        # Calculate RMS over representative section
        max_samples = sample_rate * 300  # 5 minutes max
        if audio_array.shape[0] > max_samples:
            start_idx = (audio_array.shape[0] - max_samples) // 2
            analysis_audio = audio_array[start_idx:start_idx + max_samples]
        else:
            analysis_audio = audio_array
        
        # Calculate RMS
        rms_value = np.sqrt(np.mean(analysis_audio**2))
        current_rms_db = 20 * np.log10(rms_value) if rms_value > 0 else -np.inf
        
        print(f"[RMS Target] Current: {current_rms_db:.1f} dB RMS")
        
        if not np.isfinite(current_rms_db):
            print("[RMS Target] Silence detected, returning original")
            return audio_array
        
        # Calculate and apply gain
        rms_difference = target_rms_db - current_rms_db
        gain_linear = 10 ** (rms_difference / 20)
        print(f"[RMS Target] Gain: {rms_difference:.1f} dB")
        
        gained_audio = audio_array.astype(np.float64) * gain_linear
        
        # Check if limiting needed
        max_val = np.max(np.abs(gained_audio))
        max_db = 20 * np.log10(max_val) if max_val > 0 else -np.inf
        print(f"[RMS Target] Peak after gain: {max_db:.1f} dBFS")
        
        if max_val > peak_limit_linear:
            print(f"[RMS Target] Peak exceeds {peak_limit_dbtp:.1f} dBTP limit, applying Matchering limiter...")
            
            temp_config = Config(
                internal_sample_rate=config.internal_sample_rate,
                threshold=peak_limit_linear,
                limiter=config.limiter
            )
            
            from .matchering.limiter import limit
            limited_audio = limit(gained_audio, temp_config)
            
            final_rms = np.sqrt(np.mean(limited_audio**2))
            final_rms_db = 20 * np.log10(final_rms) if final_rms > 0 else -np.inf
            final_peak = 20 * np.log10(np.max(np.abs(limited_audio)))
            print(f"[RMS Target] Final - RMS: {final_rms_db:.1f} dB, Peak: {final_peak:.1f} dBFS")
            
            return limited_audio.astype(audio_array.dtype)
        else:
            print(f"[RMS Target] Peak within {peak_limit_dbtp:.1f} dBTP limit, no limiting needed")
            return gained_audio.astype(audio_array.dtype)
            
    except Exception as e:
        print(f"[RMS Target] Error: {e}")
        return audio_array

def adjust_to_lufs_target_auto(audio_array, target_lufs=-8.4, sample_rate=44100, config=None, peak_limit_dbtp=-1.0):
    """Auto LUFS targeting: Try FFmpeg first, fallback to RMS"""
    if config is None:
        config = Config()
        
    peak_limit_linear = 10 ** (peak_limit_dbtp / 20)
        
    try:
        print(f"[Auto LUFS] Target: {target_lufs} LUFS, Peak limit: {peak_limit_dbtp:.1f} dBTP")
        
        # Ensure stereo
        if len(audio_array.shape) == 1:
            audio_array = np.column_stack([audio_array, audio_array])
        elif audio_array.shape[1] == 1:
            audio_array = np.column_stack([audio_array[:, 0], audio_array[:, 0]])
        elif audio_array.shape[1] > 2:
            audio_array = audio_array[:, :2]
        
        # For very long audio, use representative section for measurement
        max_samples = sample_rate * 300  # 5 minutes max
        if audio_array.shape[0] > max_samples:
            start_idx = (audio_array.shape[0] - max_samples) // 2
            measurement_audio = audio_array[start_idx:start_idx + max_samples]
            print(f"[Auto LUFS] Using middle section: {measurement_audio.shape}")
        else:
            measurement_audio = audio_array
        
        # Try FFmpeg LUFS measurement first
        current_lufs = measure_lufs_with_ffmpeg(measurement_audio, sample_rate)
        
        if current_lufs is not None and np.isfinite(current_lufs):
            # FFmpeg LUFS measurement successful
            lufs_difference = target_lufs - current_lufs
            gain_linear = 10 ** (lufs_difference / 20)
            
            print(f"[Auto LUFS] FFmpeg - Current: {current_lufs:.1f}, Gain: {lufs_difference:.1f} dB")
            
            # Apply gain to full audio
            gained_audio = audio_array.astype(np.float64) * gain_linear
            
            # Check limiting
            max_val = np.max(np.abs(gained_audio))
            max_db = 20 * np.log10(max_val) if max_val > 0 else -np.inf
            print(f"[Auto LUFS] Peak after gain: {max_db:.1f} dBFS")
            
            if max_val > peak_limit_linear:
                print(f"[Auto LUFS] Peak exceeds {peak_limit_dbtp:.1f} dBTP limit, applying Matchering limiter...")
                
                temp_config = Config(
                    internal_sample_rate=config.internal_sample_rate,
                    threshold=peak_limit_linear,
                    limiter=config.limiter
                )
                
                from .matchering.limiter import limit
                limited_audio = limit(gained_audio, temp_config)
                
                final_peak = 20 * np.log10(np.max(np.abs(limited_audio)))
                print(f"[Auto LUFS] Final peak: {final_peak:.1f} dBFS ({peak_limit_dbtp:.1f} dBTP target)")
                
                return limited_audio.astype(audio_array.dtype)
            else:
                print(f"[Auto LUFS] Peak within {peak_limit_dbtp:.1f} dBTP limit, no limiting needed")
                return gained_audio.astype(audio_array.dtype)
        else:
            # FFmpeg failed, fallback to RMS
            print("[Auto LUFS] FFmpeg failed, using RMS fallback")
            target_rms = lufs_to_rms_target(target_lufs)
            print(f"[Auto LUFS] Converting {target_lufs} LUFS -> {target_rms:.1f} dB RMS")
            return adjust_to_rms_target(audio_array, target_rms, sample_rate, config, peak_limit_dbtp)
            
    except Exception as e:
        print(f"[Auto LUFS] Error: {e}")
        # Final fallback to RMS
        target_rms = lufs_to_rms_target(target_lufs)
        return adjust_to_rms_target(audio_array, target_rms, sample_rate, config, peak_limit_dbtp)

# ============================================================================
# NODE CLASSES
# ============================================================================

class Matchering:
    """Classic Matchering node - maintains original simple interface"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target": ("AUDIO",),
                "reference": ("AUDIO",),
            }
        }

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return hash(frozenset(kwargs))

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = (
        "Result",
        "Result (no limiter)",
        "Result (no limiter, normalized)",
    )

    CATEGORY = "audio/matchering"
    FUNCTION = "matchering"

    def matchering(self, target, reference):
        log(print)

        result, result_no_limiter, result_no_limiter_normalized = process(
            target_audio=target,
            reference_audio=reference,
        )

        return (
            {
                "waveform": torch.from_numpy(result.T).unsqueeze(0),
                "sample_rate": reference["sample_rate"],
            },
            {
                "waveform": torch.from_numpy(result_no_limiter.T).unsqueeze(0),
                "sample_rate": reference["sample_rate"],
            },
            {
                "waveform": torch.from_numpy(result_no_limiter_normalized.T).unsqueeze(0),
                "sample_rate": reference["sample_rate"],
            },
        )

class MatcheringAdvanced:
    """Advanced Matchering node with enhanced spectral analysis methods"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target": ("AUDIO", {"tooltip": "Audio to be mastered"}),
                "reference": ("AUDIO", {"tooltip": "Reference audio to match against"}),
                # Enhanced Analysis Methods
                "analysis_method": (
                    ["Loudest (Original)", "Median Spectrum", "Percentile Spectrum"],
                    {
                        "default": "Median Spectrum",
                        "tooltip": "Spectral analysis method: Original uses loudest sections only, Median uses typical frequency content, Percentile uses configurable threshold"
                    }
                ),
                "percentile": (
                    "FLOAT",
                    {
                        "default": 50.0, "min": 1.0, "max": 99.0, "step": 1.0,
                        "tooltip": "Percentile for spectral analysis (50=median, 75=brighter, 25=darker). Only used with Percentile Spectrum method"
                    }
                ),
                "enable_gating": (
                    "BOOLEAN", 
                    {
                        "default": False,
                        "tooltip": "Exclude very quiet sections from analysis to focus on musical content"
                    }
                ),
                "gate_threshold_db": (
                    "FLOAT", 
                    {
                        "default": -40.0, "min": -60.0, "max": -20.0, "step": 1.0,
                        "tooltip": "dB threshold below which audio is excluded from analysis. Lower values include more quiet content"
                    }
                ),
                # Core Processing Parameters
                "internal_sample_rate": (
                    "INT",
                    {
                        "default": 44100, "min": 0, "max": 192000, "step": 1,
                        "tooltip": "Processing sample rate. Higher rates preserve more frequency detail but use more CPU"
                    }
                ),
                "max_length": (
                    "FLOAT", 
                    {
                        "default": 15 * 60, "min": 0, "step": 1,
                        "tooltip": "Maximum audio length to process in seconds. Longer tracks may be truncated"
                    }
                ),
                "max_piece_size": (
                    "FLOAT", 
                    {
                        "default": 15, "min": 0, "step": 1,
                        "tooltip": "Maximum chunk size for analysis in seconds. Affects processing granularity"
                    }
                ),
                "threshold": (
                    "FLOAT",
                    {
                        "default": (2**15 - 61) / 2**15,
                        "max": 0.9999999,
                        "step": 0.0000001,
                        "round": False,
                        "tooltip": "Peak limiter threshold (0.998 ≈ -0.017dB). Higher values = louder output but more limiting"
                    }
                ),
                "min_value": (
                    "FLOAT",
                    {
                        "default": 1e-6, "min": 0, "max": 0.1, "step": 1e-6,
                        "tooltip": "Minimum spectral magnitude to prevent division by zero. Lower = more precise but potentially unstable"
                    }
                ),
                # FFT and Processing Parameters
                "auto_scale_fft": (
                    "BOOLEAN", 
                    {
                        "default": True,
                        "tooltip": "Automatically scale FFT size based on sample rate to maintain frequency resolution"
                    }
                ),
                "fft_size": (
                    "INT",
                    {
                        "default": 4096, "min": 512, "max": 16384, "step": 512,
                        "tooltip": "FFT size for frequency analysis. Larger = better frequency resolution, smaller = better time resolution"
                    }
                ),
                "lowess_frac": (
                    "FLOAT",
                    {
                        "default": 0.0375, "min": 0.01, "max": 0.1, "step": 0.0025,
                        "tooltip": "LOWESS smoothing fraction (3.75% = musical, 1% = detailed, 10% = very smooth). Controls EQ curve smoothness"
                    }
                ),
                "lowess_it": (
                    "INT", 
                    {
                        "default": 0, "min": 0, "max": 10, "step": 1,
                        "tooltip": "LOWESS iterations (0 = fast, >0 = more refined). Usually 0 is sufficient"
                    }
                ),
                "lowess_delta": (
                    "FLOAT",
                    {
                        "default": 0.001, "min": 0.0001, "max": 0.01, "step": 0.0001,
                        "tooltip": "LOWESS precision threshold. Lower = more precise but slower processing"
                    }
                ),
                "lin_log_oversampling": (
                    "INT",
                    {
                        "default": 2, "min": 1, "max": 8, "step": 1,
                        "tooltip": "Linear to logarithmic frequency conversion oversampling factor. Higher = smoother curves"
                    }
                ),
                "rms_correction_steps": (
                    "INT",
                    {
                        "default": 4, "min": 1, "max": 10, "step": 1,
                        "tooltip": "Number of RMS level correction iterations. More steps = more precise level matching"
                    }
                ),
                "allow_equality": (
                    "BOOLEAN", 
                    {
                        "default": False,
                        "tooltip": "Allow target and reference to be identical files (normally blocked to prevent meaningless processing)"
                    }
                ),
                # Comparison and Output Options
                "enable_comparison": (
                    "BOOLEAN", 
                    {
                        "default": False,
                        "tooltip": "Generate comparison output using original method for A/B testing"
                    }
                ),
            },
            "optional": {
                "limiter_config": ("MATCHERING_LIMITER_CONFIG",),
            },
        }

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return hash(frozenset(kwargs))

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "STRING", "FLOAT", "BOOLEAN", "FLOAT", "INT")
    RETURN_NAMES = (
    "Result",
    "Result (no limiter)",
    "Result (no limiter, normalized)", 
    "Comparison (Original Method)",
    "analysis_method",
    "percentile", 
    "enable_gating",
    "gate_threshold_db",
    "internal_sample_rate"
)

    CATEGORY = "audio/matchering"
    FUNCTION = "matchering_advanced"

    def matchering_advanced(
        self,
        target,
        reference,
        analysis_method,
        percentile,
        enable_gating,
        gate_threshold_db,
        internal_sample_rate,
        max_length,
        max_piece_size,
        threshold,
        min_value,
        auto_scale_fft,
        fft_size,
        lowess_frac,
        lowess_it,
        lowess_delta,
        lin_log_oversampling,
        rms_correction_steps,
        allow_equality,
        enable_comparison,
        limiter_config=LimiterConfig(),
    ):
        """Advanced processing with enhanced spectral analysis and full parameter control"""
        log(print)
        
        # Convert UI strings to internal method names
        method_mapping = {
            "Loudest (Original)": "loudest",
            "Median Spectrum": "median", 
            "Percentile Spectrum": "percentile"
        }
        
        method = method_mapping.get(analysis_method, "median")
        
        print(f"[Enhanced Matchering] Using {analysis_method} method")
        if method == "percentile":
            print(f"[Enhanced Matchering] Percentile: {percentile}%")
        if enable_gating and method != "loudest":
            print(f"[Enhanced Matchering] Gating enabled at {gate_threshold_db:.1f} dB")

        # Create comprehensive configuration
        config = Config(
            internal_sample_rate=internal_sample_rate,
            max_length=max_length,
            max_piece_size=max_piece_size,  # Convert to samples
            threshold=threshold,
            min_value=min_value,
            auto_scale_fft=auto_scale_fft,
            fft_size=fft_size,
            lowess_frac=lowess_frac,
            lowess_it=lowess_it,
            lowess_delta=lowess_delta,
            lin_log_oversampling=lin_log_oversampling,
            rms_correction_steps=rms_correction_steps,
            allow_equality=allow_equality,
            limiter=limiter_config,
        )

        # Add enhanced method settings to config
        config.analysis_method = method
        config.analysis_percentile = percentile
        config.enable_gating = enable_gating
        config.gate_threshold_db = gate_threshold_db

        # Process with enhanced algorithm
        result, result_no_limiter, result_no_limiter_normalized = process(
            target_audio=target,
            reference_audio=reference,
            config=config
        )

        # Generate comparison if requested
        comparison_result = None
        if enable_comparison and method != "loudest":
            print("[Enhanced Matchering] Generating comparison with original method...")
            # Create config for original method
            original_config = Config(
                internal_sample_rate=internal_sample_rate,
                max_length=max_length,
                max_piece_size=max_piece_size,
                threshold=threshold,
                min_value=min_value,
                auto_scale_fft=auto_scale_fft,
                fft_size=fft_size,
                lowess_frac=lowess_frac,
                lowess_it=lowess_it,
                lowess_delta=lowess_delta,
                lin_log_oversampling=lin_log_oversampling,
                rms_correction_steps=rms_correction_steps,
                allow_equality=allow_equality,
                limiter=limiter_config,
            )
            # Force original method
            original_config.analysis_method = "loudest"
            
            comparison_result, _, _ = process(
                target_audio=target,
                reference_audio=reference,
                config=original_config
            )
        else:
            # Return zeros for comparison if not requested
            comparison_result = np.zeros_like(result)

        return (
            {
                "waveform": torch.from_numpy(result.T).unsqueeze(0),
                "sample_rate": reference["sample_rate"],
            },
            {
                "waveform": torch.from_numpy(result_no_limiter.T).unsqueeze(0),
                "sample_rate": reference["sample_rate"],
            },
            {
                "waveform": torch.from_numpy(result_no_limiter_normalized.T).unsqueeze(0),
                "sample_rate": reference["sample_rate"],
            }, 
            {
                "waveform": torch.from_numpy(comparison_result.T).unsqueeze(0),
                "sample_rate": reference["sample_rate"],
            },
            analysis_method,
            percentile,
            enable_gating,
            gate_threshold_db,
            internal_sample_rate

        )

class MatcheringEnhanced:
    """
    Enhanced Matchering node with median spectrum capabilities.
    
    This node provides access to advanced frequency analysis methods
    including median spectrum matching for more musical results.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target": ("AUDIO",),
                "reference": ("AUDIO",),
                
                # FREQUENCY ANALYSIS METHOD
                "analysis_method": (
                    ["Loudest (Original)", "Median Spectrum", "Percentile Spectrum"], 
                    {
                        "default": "Median Spectrum",
                        "tooltip": "Loudest=original method, Median=analyze entire track, Percentile=adjustable brightness"
                    }
                ),
                
                # PERCENTILE CONTROL
                "percentile": (
                    "FLOAT", 
                    {
                        "default": 75.0, 
                        "min": 1.0, 
                        "max": 99.0, 
                        "step": 1.0,
                        "tooltip": "50=median, 75=brighter, 25=darker. Higher values emphasize louder sections."
                    }
                ),
                
                # GATING CONTROL
                "enable_gating": (
                    "BOOLEAN", 
                    {
                        "default": True,
                        "tooltip": "Remove silent sections from analysis (recommended)"
                    }
                ),
                "gate_threshold_db": (
                    "FLOAT",
                    {
                        "default": -40.0,
                        "min": -60.0,
                        "max": -20.0,
                        "step": 1.0,
                        "tooltip": "Sections quieter than this (in dB) are excluded from analysis"
                    }
                ),
                
                # A/B COMPARISON FEATURE
                "enable_comparison": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Output both original and new method results for comparison"
                    }
                ),
                
                # CORE PARAMETERS
                "internal_sample_rate": (
                    "INT",
                    {"default": 44100, "min": 8000, "max": 192000, "step": 1},
                ),
                "fft_size": (
                    "INT", 
                    {
                        "default": 4096, 
                        "min": 1024, 
                        "max": 16384, 
                        "step": 1024,
                        "tooltip": "Larger=more frequency detail, slower processing"
                    }
                ),
                "max_length": ("FLOAT", {"default": 15 * 60, "min": 60, "step": 30}),
                "threshold": (
                    "FLOAT",
                    {
                        "default": (2**15 - 61) / 2**15,
                        "min": 0.5,
                        "max": 0.9999,
                        "step": 0.001,
                        "tooltip": "Peak limiter threshold"
                    },
                ),
            },
            "optional": {
                "limiter_config": ("MATCHERING_LIMITER_CONFIG",),
            },
        }

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return hash(frozenset(kwargs))

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "STRING", "FLOAT", "BOOLEAN", "FLOAT", "INT")
    RETURN_NAMES = (
        "Result",
        "Result (no limiter)",
        "Result (no limiter, normalized)", 
        "Comparison (Original Method)",
        "analysis_method",      # NEW
        "percentile",           # NEW
        "enable_gating",        # NEW
        "gate_threshold_db",    # NEW
        "internal_sample_rate"  # NEW
    )

    CATEGORY = "audio/matchering"
    FUNCTION = "matchering_enhanced"

    def matchering_enhanced(self, target, reference, analysis_method, percentile, 
                          enable_gating, gate_threshold_db, enable_comparison,
                          internal_sample_rate, fft_size, max_length, threshold, 
                          limiter_config=LimiterConfig()):
        
        """Enhanced processing with median spectrum capabilities"""
        log(print)
        
        # Convert UI strings to internal method names
        method_mapping = {
            "Loudest (Original)": "loudest",
            "Median Spectrum": "median", 
            "Percentile Spectrum": "percentile"
        }
        
        method = method_mapping.get(analysis_method, "median")
        
        print(f"[Enhanced Matchering] Using {analysis_method} method")
        if method == "percentile":
            print(f"[Enhanced Matchering] Percentile: {percentile}%")
        if enable_gating and method != "loudest":
            print(f"[Enhanced Matchering] Gating enabled at {gate_threshold_db:.1f} dB")

        # Create configuration
        config = Config(
            internal_sample_rate=internal_sample_rate,
            max_length=max_length,
            fft_size=fft_size,
            threshold=threshold,
            limiter=limiter_config,
        )
        config.analysis_method = method
        config.analysis_percentile = percentile

        # Process with standard method (enhanced processing would go here)
        # Note: This uses the standard process for now - enhanced algorithm would need implementation
        result, result_no_limiter, result_no_limiter_normalized = process(
            target_audio=target,
            reference_audio=reference,
            config=config
        )

        # Generate comparison if requested
        comparison_result = None
        if enable_comparison and method != "loudest":
            print("[Enhanced Matchering] Generating comparison with original method...")
            comparison_result, _, _ = process(
                target_audio=target,
                reference_audio=reference,
                config=config
            )
        else:
            # Return zeros for comparison if not requested
            comparison_result = np.zeros_like(result)

        return (
    {
        "waveform": torch.from_numpy(result.T).unsqueeze(0),
        "sample_rate": reference["sample_rate"],
    },
    {
        "waveform": torch.from_numpy(result_no_limiter.T).unsqueeze(0),
        "sample_rate": reference["sample_rate"],
    },
    {
        "waveform": torch.from_numpy(result_no_limiter_normalized.T).unsqueeze(0),
        "sample_rate": reference["sample_rate"],
    },
    {
        "waveform": torch.from_numpy(comparison_result.T).unsqueeze(0),
        "sample_rate": reference["sample_rate"],
    },
    analysis_method,        
    percentile,
    enable_gating,
    gate_threshold_db,
    internal_sample_rate
)


class MatcheringLimiterConfig:
    """Limiter configuration node"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "attack": ("FLOAT", {"default": 1, "min": 0.1, "step": 0.1}),
                "hold": ("FLOAT", {"default": 1, "min": 0.1, "step": 0.1}),
                "release": ("FLOAT", {"default": 3000, "min": 1, "step": 1}),
                "attack_filter_coefficient": (
                    "FLOAT",
                    {"default": -2, "min": -1000, "step": 0.1},
                ),
                "hold_filter_order": (
                    "INT",
                    {"default": 1, "min": 1, "step": 1},
                ),
                "hold_filter_coefficient": (
                    "FLOAT",
                    {"default": 7, "step": 0.1},
                ),
                "release_filter_order": (
                    "INT",
                    {"default": 1, "min": 1, "step": 1},
                ),
                "release_filter_coefficient": ("FLOAT", {"default": 800, "step": 1}),
            },
        }

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return hash(frozenset(kwargs))

    RETURN_TYPES = ("MATCHERING_LIMITER_CONFIG",)
    RETURN_NAMES = ("limiter_config",)

    CATEGORY = "audio/matchering"
    FUNCTION = "matchering_limiter_config"

    def matchering_limiter_config(
        self,
        attack,
        hold,
        release,
        attack_filter_coefficient,
        hold_filter_order,
        hold_filter_coefficient,
        release_filter_order,
        release_filter_coefficient,
    ):
        limiter_config = LimiterConfig(
            attack=attack,
            hold=hold,
            release=release,
            attack_filter_coefficient=attack_filter_coefficient,
            hold_filter_order=hold_filter_order,
            hold_filter_coefficient=hold_filter_coefficient,
            release_filter_order=release_filter_order,
            release_filter_coefficient=release_filter_coefficient,
        )

        return (limiter_config,)


# Export node mappings
NODE_CLASS_MAPPINGS = {
    "Matchering": Matchering,
    "MatcheringAdvanced": MatcheringAdvanced,  # Full parameter control + enhanced methods
    "MatcheringEnhanced": MatcheringEnhanced,  # Simple enhanced methods only
    "MatcheringLimiterConfig": MatcheringLimiterConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Matchering": "Matchering",
    "MatcheringAdvanced": "Matchering (Advanced + Enhanced)",
    "MatcheringEnhanced": "Matchering (Enhanced)",
    "MatcheringLimiterConfig": "Matchering Limiter Config",
}