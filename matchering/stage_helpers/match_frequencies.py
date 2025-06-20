# -*- coding: utf-8 -*-

"""
Matchering - Audio Matching and Mastering Python Library
Copyright (C) 2016-2022 Sergree

Enhanced with Median Spectrum Matching capabilities and Auto FFT Scaling

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
from time import time
from scipy import signal, interpolate

from ..log import debug
from .. import Config
from ..dsp import ms_to_lr, smooth_lowess


def get_scaled_fft_size(sample_rate, base_fft=4096, base_rate=44100):
    """Scale FFT size to maintain frequency resolution across sample rates"""
    scale_factor = sample_rate / base_rate
    scaled_size = int(base_fft * scale_factor)
    # Round to nearest power of 2
    return 2 ** round(np.log2(scaled_size))


def apply_fft_scaling(config, auto_scale_fft=True):
    """Apply FFT size scaling based on sample rate if enabled"""
    if auto_scale_fft:
        original_fft = config.fft_size
        scale_factor = config.internal_sample_rate / 44100
        config.fft_size = 2 ** round(np.log2(config.fft_size * scale_factor))
        debug(f"FFT scaling: {original_fft} -> {config.fft_size} for {config.internal_sample_rate}Hz")
    return config


def __get_effective_fft_size(config, auto_scale_fft=True):
    """Get the effective FFT size to use, applying auto-scaling if enabled"""
    if auto_scale_fft:
        scale_factor = config.internal_sample_rate / 44100
        scaled_size = int(config.fft_size * scale_factor)
        # Keep it as power of 2 and within reasonable bounds
        scaled_size = max(1024, min(16384, 2 ** round(np.log2(scaled_size))))
        debug(f"Auto FFT scaling: {config.fft_size} -> {scaled_size} for {config.internal_sample_rate}Hz")
        return scaled_size
    else:
        debug(f"Using fixed FFT size: {config.fft_size}")
        return config.fft_size


def __average_fft(
    loudest_pieces: np.ndarray, sample_rate: int, fft_size: int, auto_scale_fft: bool = False
) -> np.ndarray:
    """
    Original Matchering method: average FFT of loudest pieces.
    Now with optional auto-scaling support.
    """
    # Apply auto-scaling if enabled
    if auto_scale_fft:
        scale_factor = sample_rate / 44100
        effective_fft_size = max(1024, min(16384, 2 ** round(np.log2(fft_size * scale_factor))))
        debug(f"Auto-scaled FFT size: {fft_size} -> {effective_fft_size}")
    else:
        effective_fft_size = fft_size
    
    *_, specs = signal.stft(
        loudest_pieces,
        sample_rate,
        window="boxcar",
        nperseg=effective_fft_size,
        noverlap=0,
        boundary=None,
        padded=False,
    )
    spectrum = np.abs(specs).mean((0, 2))
    debug(f"Original method: {len(loudest_pieces)} samples, {specs.shape[-1]} time frames")
    debug(f"Original spectrum range: {spectrum.min():.6f} to {spectrum.max():.6f}")
    return spectrum


def __compute_median_spectrum_stft(
    audio: np.ndarray, 
    sample_rate: int, 
    fft_size: int,
    auto_scale_fft: bool = False
) -> np.ndarray:
    """
    Compute median spectrum using STFT approach (like original Matchering).
    Now with auto-scaling support.
    """
    # Apply auto-scaling if enabled
    if auto_scale_fft:
        scale_factor = sample_rate / 44100
        effective_fft_size = max(1024, min(16384, 2 ** round(np.log2(fft_size * scale_factor))))
        debug(f"Auto-scaled FFT size for median: {fft_size} -> {effective_fft_size}")
    else:
        effective_fft_size = fft_size
    
    # Use the same STFT parameters as original Matchering
    _, _, specs = signal.stft(
        audio,
        sample_rate,
        window="boxcar",
        nperseg=effective_fft_size,
        noverlap=0,
        boundary=None,
        padded=False,
    )
    
    # Take absolute magnitude and compute median across time
    abs_specs = np.abs(specs)
    if len(abs_specs.shape) == 3:
        # Multiple channels: median across time (axis 2), then average across channels (axis 0)
        median_spectrum = np.median(abs_specs, axis=2).mean(axis=0)
    else:
        # Single channel: median across time (axis 1)
        median_spectrum = np.median(abs_specs, axis=1)
    
    # Mask high frequency bins (>20kHz)
    freq_bins = np.fft.rfftfreq(effective_fft_size, 1/sample_rate)
    mask = freq_bins <= 20000
    median_spectrum = median_spectrum * mask
    
    debug(f"Median method: {len(audio)} samples, {abs_specs.shape[-1]} time frames")
    debug(f"Masked bins above 20kHz, keeping {mask.sum()}/{len(mask)} bins")
    debug(f"Median spectrum range: {median_spectrum.min():.6f} to {median_spectrum.max():.6f}")
    
    # Compare with mean for debugging
    if len(abs_specs.shape) == 3:
        mean_spectrum = abs_specs.mean((0, 2)) * mask
    else:
        mean_spectrum = abs_specs.mean(axis=1) * mask
    
    ratio = median_spectrum / (mean_spectrum + 1e-10)
    debug(f"Median/Mean ratio range: {ratio.min():.3f} to {ratio.max():.3f}")
    
    return median_spectrum


def __compute_percentile_spectrum_stft(
    audio: np.ndarray,
    sample_rate: int,
    fft_size: int,
    percentile: float = 75.0,
    auto_scale_fft: bool = False
) -> np.ndarray:
    """
    Compute percentile spectrum using STFT approach (like original Matchering).
    Now with auto-scaling support.
    """
    if not (0 <= percentile <= 100):
        raise ValueError(f"Percentile must be 0-100, got {percentile}")
    
    # Apply auto-scaling if enabled
    if auto_scale_fft:
        scale_factor = sample_rate / 44100
        effective_fft_size = max(1024, min(16384, 2 ** round(np.log2(fft_size * scale_factor))))
        debug(f"Auto-scaled FFT size for percentile: {fft_size} -> {effective_fft_size}")
    else:
        effective_fft_size = fft_size
        
    # Use the same STFT parameters as original Matchering
    _, _, specs = signal.stft(
        audio,
        sample_rate,
        window="boxcar",
        nperseg=effective_fft_size,
        noverlap=0,
        boundary=None,
        padded=False,
    )
    
    # Take absolute magnitude and compute percentile across time
    abs_specs = np.abs(specs)
    if len(abs_specs.shape) == 3:
        # Multiple channels: percentile across time (axis 2), then average across channels (axis 0)
        percentile_spectrum = np.percentile(abs_specs, percentile, axis=2).mean(axis=0)
    else:
        # Single channel: percentile across time (axis 1)
        percentile_spectrum = np.percentile(abs_specs, percentile, axis=1)
    
    # Mask high frequency bins (>20kHz)
    freq_bins = np.fft.rfftfreq(effective_fft_size, 1/sample_rate)
    mask = freq_bins <= 20000
    percentile_spectrum = percentile_spectrum * mask
    
    debug(f"Percentile spectrum: computed {percentile}th percentile across {abs_specs.shape[-1]} time frames")
    debug(f"Masked bins above 20kHz, keeping {mask.sum()}/{len(mask)} bins")
    
    return percentile_spectrum


def get_original_loudest_chunks(audio, sample_rate, chunk_duration=15.0):
    """
    Replicate original Matchering's chunk selection logic.
    
    This implements Sergey's first heuristic: divide into 15-second chunks
    and keep only those with RMS above the track's average.
    """
    chunk_size = int(chunk_duration * sample_rate)
    
    # Divide into 15-second chunks
    num_chunks = len(audio) // chunk_size
    chunks = []
    chunk_rms = []
    
    for i in range(num_chunks):
        start = i * chunk_size
        chunk = audio[start:start + chunk_size]
        rms = np.sqrt(np.mean(chunk**2))
        chunks.append(chunk)
        chunk_rms.append(rms)
    
    # Only keep chunks with RMS > average (original logic)
    avg_rms = np.mean(chunk_rms)
    selected_chunks = [chunk for chunk, rms in zip(chunks, chunk_rms) 
                      if rms > avg_rms]
    
    debug(f"Original chunk selection: kept {len(selected_chunks)}/{num_chunks} chunks")
    
    return np.concatenate(selected_chunks) if selected_chunks else audio


def __smooth_exponentially(matching_fft: np.ndarray, config: Config, effective_fft_size: int = None) -> np.ndarray:
    """
    Apply logarithmic smoothing to the matching curve.
    This is crucial for musical results - without smoothing, the EQ would
    have very sharp peaks and notches that sound unnatural.
    
    Parameters:
    - matching_fft: The matching curve to smooth
    - config: Matchering configuration
    - effective_fft_size: The actual FFT size used (for auto-scaling), if None uses config.fft_size
    """
    # Use effective FFT size if provided (for auto-scaling), otherwise use config FFT size
    if effective_fft_size is None:
        effective_fft_size = config.fft_size
    
    debug(f"LOWESS input: sample_rate={config.internal_sample_rate}, effective_fft_size={effective_fft_size}")
    debug(f"LOWESS params: frac={config.lowess_frac}, it={config.lowess_it}, delta={config.lowess_delta}")
    debug(f"matching_fft length: {len(matching_fft)}")
    
    # Create frequency grid based on effective FFT size (not config FFT size)
    grid_linear = (
        config.internal_sample_rate * 0.5 * np.linspace(0, 1, effective_fft_size // 2 + 1)
    )
    debug(f"Frequency grid: {grid_linear[0]:.1f} to {grid_linear[-1]:.1f} Hz, {len(grid_linear)} bins")
    
    # Verify array lengths match
    if len(grid_linear) != len(matching_fft):
        debug(f"ERROR: Length mismatch - grid_linear: {len(grid_linear)}, matching_fft: {len(matching_fft)}")
        debug(f"Config FFT size: {config.fft_size}, Effective FFT size: {effective_fft_size}")
        raise ValueError(f"Grid and matching_fft length mismatch: {len(grid_linear)} vs {len(matching_fft)}")

    grid_logarithmic = (
        config.internal_sample_rate
        * 0.5
        * np.logspace(
            np.log10(4 / effective_fft_size),  # Use effective_fft_size here too
            0,
            (effective_fft_size // 2) * config.lin_log_oversampling + 1,
        )
    )

    interpolator = interpolate.interp1d(grid_linear, matching_fft, "cubic")
    matching_fft_log = interpolator(grid_logarithmic)

    matching_fft_log_filtered = smooth_lowess(
        matching_fft_log, config.lowess_frac, config.lowess_it, config.lowess_delta
    )

    interpolator = interpolate.interp1d(
        grid_logarithmic, matching_fft_log_filtered, "cubic", fill_value="extrapolate"
    )
    matching_fft_filtered = interpolator(grid_linear)

    # Preserve DC and lowest frequency bin
    matching_fft_filtered[0] = 0
    matching_fft_filtered[1] = matching_fft[1]

    debug(f"Smoothing effect: input range {matching_fft.min():.6f}-{matching_fft.max():.6f}, output range {matching_fft_filtered.min():.6f}-{matching_fft_filtered.max():.6f}")

    return matching_fft_filtered


def get_fir_enhanced(
    target_loudest_pieces: np.ndarray,
    reference_loudest_pieces: np.ndarray,
    target_full_audio: np.ndarray,
    reference_full_audio: np.ndarray,
    name: str,
    config: Config,
    method: str = "loudest",  # "loudest", "median", "percentile"
    percentile: float = 50.0,
    auto_scale_fft: bool = False
) -> np.ndarray:
    
   
    """
    Enhanced FIR calculation with multiple analysis methods and auto-scaling.
    
    Uses original Matchering's chunk selection logic (15-sec chunks, RMS > average)
    then applies median/percentile instead of mean for spectral analysis.
    
    Parameters:
    - target_loudest_pieces, reference_loudest_pieces: Pre-selected loud sections (for "loudest" method)
    - target_full_audio, reference_full_audio: Complete audio arrays (for other methods)
    - name: Name for debug output ("mid" or "side")
    - config: Matchering configuration object
    - method: Analysis method ("loudest", "median", "percentile")
    - percentile: Percentile value if method="percentile" (0-100)
    - auto_scale_fft: Whether to auto-scale FFT size based on sample rate
    """
    debug(f"Computing {name} FIR using '{method}' method with auto_scale_fft={auto_scale_fft}")
    debug(f"Config sample rate: {config.internal_sample_rate}")
    debug(f"target_loudest_pieces length: {len(target_loudest_pieces)}")
    debug(f"=== METHOD CHECK: {method} ===")
    
    if method == "loudest":
        # Original Matchering approach - use pre-selected loud pieces
        target_average_fft = __average_fft(
            target_loudest_pieces, config.internal_sample_rate, config.fft_size, auto_scale_fft
        )
        reference_average_fft = __average_fft(
            reference_loudest_pieces, config.internal_sample_rate, config.fft_size, auto_scale_fft
        )
        # DEBUG: Log FFT differences for identical files
        debug(f"=== FFT DEBUG {name} ===")
        debug(f"Target FFT shape: {target_average_fft.shape}")
        debug(f"Reference FFT shape: {reference_average_fft.shape}")

        # Check for differences (should be zero for identical files)
        fft_diff = target_average_fft - reference_average_fft
        max_diff = np.max(np.abs(fft_diff))
        debug(f"Max FFT difference: {max_diff}")

        # Focus on high frequencies (21kHz area)
        nyquist = config.internal_sample_rate / 2
        freq_bins = np.linspace(0, nyquist, len(target_average_fft))
        high_freq_mask = freq_bins >= 18000  # 18kHz and above
        high_freq_diff = fft_diff[high_freq_mask]
        debug(f"High freq (18kHz+) max diff: {np.max(np.abs(high_freq_diff))}")

        # Log specific 21kHz bin if it exists
        target_21k_bin = int(21000 * len(target_average_fft) / nyquist)
        if target_21k_bin < len(target_average_fft):
         debug(f"21kHz bin ({target_21k_bin}): target={target_average_fft[target_21k_bin]:.8f}, ref={reference_average_fft[target_21k_bin]:.8f}, diff={fft_diff[target_21k_bin]:.8f}")
         #Test with: Two identical 21kHz tone files using "loudest" method. Expected max_diff = 0.
        
    else:
        # Enhanced approaches - use same input as original for fair comparison
        debug("Using same input as original method for fair comparison")
        audio_to_analyze_target = target_loudest_pieces
        audio_to_analyze_reference = reference_loudest_pieces
        debug(f"Using target_loudest_pieces length: {len(audio_to_analyze_target)}")
        debug(f"Using reference_loudest_pieces length: {len(audio_to_analyze_reference)}")
        
        # Compute spectra based on selected method using STFT approach
        if method == "median":
            target_average_fft = __compute_median_spectrum_stft(
                audio_to_analyze_target, config.internal_sample_rate, config.fft_size, auto_scale_fft
            )
            reference_average_fft = __compute_median_spectrum_stft(
                audio_to_analyze_reference, config.internal_sample_rate, config.fft_size, auto_scale_fft
            )
            
        elif method == "percentile":
            debug(f"Using percentile: {percentile}")
            target_average_fft = __compute_percentile_spectrum_stft(
                audio_to_analyze_target, config.internal_sample_rate, 
                config.fft_size, percentile, auto_scale_fft
            )
            reference_average_fft = __compute_percentile_spectrum_stft(
                audio_to_analyze_reference, config.internal_sample_rate, 
                config.fft_size, percentile, auto_scale_fft
            )
        
        else:
            raise ValueError(f"Unknown method: {method}. Choose 'loudest', 'median', or 'percentile'")

    # Ensure no division by zero
    np.maximum(config.min_value, target_average_fft, out=target_average_fft)
    
    # Calculate the matching curve (this is the EQ that transforms target to reference)
    matching_fft = reference_average_fft / target_average_fft

    # Apply smoothing for musical results
    matching_fft_filtered = __smooth_exponentially(matching_fft, config)

    # Convert to time-domain FIR filter
    fir = np.fft.irfft(matching_fft_filtered)
    fir = np.fft.ifftshift(fir) * signal.windows.hann(len(fir))

    debug(f"{name} FIR calculation complete using {method} method")
    
    return fir


def get_fir(
    target_loudest_pieces: np.ndarray,
    reference_loudest_pieces: np.ndarray,
    name: str,
    config: Config,
) -> np.ndarray:
    """
    Original get_fir function - maintained for backward compatibility.
    Now automatically uses auto-scaling from config.
    """
    debug(f"=== ORIGINAL GET_FIR: {name} ===")
    debug(f"Calculating the {name} FIR for the matching EQ... (auto_scale_fft={config.auto_scale_fft})")

    target_average_fft = __average_fft(
        target_loudest_pieces, config.internal_sample_rate, config.fft_size, config.auto_scale_fft
    )
    reference_average_fft = __average_fft(
        reference_loudest_pieces, config.internal_sample_rate, config.fft_size, config.auto_scale_fft
    )

    # Add FFT debug code here, BEFORE np.maximum modifies target_average_fft:
    fft_diff = target_average_fft - reference_average_fft
    max_diff = np.max(np.abs(fft_diff))
    debug(f"FFT max difference: {max_diff}")

    # 21kHz bin check
    nyquist = config.internal_sample_rate / 2
    bin_21k = int(21000 * len(target_average_fft) / nyquist)
    if bin_21k < len(target_average_fft):
        debug(f"21kHz bin: diff={fft_diff[bin_21k]:.8f}")

    np.maximum(config.min_value, target_average_fft, out=target_average_fft)
    matching_fft = reference_average_fft / target_average_fft

    matching_fft_filtered = __smooth_exponentially(matching_fft, config)

    fir = np.fft.irfft(matching_fft_filtered)
    fir = np.fft.ifftshift(fir) * signal.windows.hann(len(fir))

    return fir


def convolve(
    target_mid: np.ndarray,
    mid_fir: np.ndarray,
    target_side: np.ndarray,
    side_fir: np.ndarray,
) -> (np.ndarray, np.ndarray):
    """
    Convolve the target audio with the calculated FIR filters.
    This applies the EQ matching to the actual audio.
    """
    debug("Convolving the TARGET audio with calculated FIRs...")
    timer = time()
    result_mid = signal.fftconvolve(target_mid, mid_fir, "same")
    result_side = signal.fftconvolve(target_side, side_fir, "same")
    debug(f"The convolution is done in {time() - timer:.2f} seconds")

    debug("Converting MS to LR...")
    result = ms_to_lr(result_mid, result_side)

    return result, result_mid