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

from .log import Code, info, debug, debug_line
from . import Config
from .utils import to_db
from .dsp import amplify, normalize, clip, ms_to_lr, smooth_lowess
from .stage_helpers import (
    normalize_reference,
    analyze_levels,
    get_average_rms,
    get_lpis_and_match_rms,
    get_rms_c_and_amplify_pair,
)
from .limiter import limit


# Enhanced frequency matching functions
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
    
    # Calculate effective FFT size for auto-scaling
    if auto_scale_fft:
        scale_factor = config.internal_sample_rate / 44100
        effective_fft_size = max(1024, min(16384, 2 ** round(np.log2(config.fft_size * scale_factor))))
        debug(f"Auto-scaled FFT size: {config.fft_size} -> {effective_fft_size}")
    else:
        effective_fft_size = config.fft_size
    
    if method == "loudest":
        # Original Matchering approach - use pre-selected loud pieces
        target_average_fft = __average_fft(
            target_loudest_pieces, config.internal_sample_rate, config.fft_size, auto_scale_fft
        )
        reference_average_fft = __average_fft(
            reference_loudest_pieces, config.internal_sample_rate, config.fft_size, auto_scale_fft
        )
        
    else:
        # Enhanced approaches - use full audio for true analysis
        debug("Using full audio for enhanced analysis")
        audio_to_analyze_target = target_full_audio
        audio_to_analyze_reference = reference_full_audio
        debug(f"Using target_full_audio length: {len(audio_to_analyze_target)}")
        debug(f"Using reference_full_audio length: {len(audio_to_analyze_reference)}")

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

    # Apply smoothing for musical results - pass effective FFT size
    matching_fft_filtered = __smooth_exponentially(matching_fft, config, effective_fft_size)

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
    Now automatically uses auto-scaling from config if available.
    """
    debug(f"=== ORIGINAL get_fir() called for {name} - enhanced methods NOT running ===")
    
    debug(f"Calculating the {name} FIR for the matching EQ...")
    
    # Check if config has auto_scale_fft attribute, default to False if not
    auto_scale_fft = getattr(config, 'auto_scale_fft', False)
    
    if auto_scale_fft:
        debug(f"Using auto-scaling: {auto_scale_fft}")

    # Calculate effective FFT size for auto-scaling
    if auto_scale_fft:
        scale_factor = config.internal_sample_rate / 44100
        effective_fft_size = max(1024, min(16384, 2 ** round(np.log2(config.fft_size * scale_factor))))
        debug(f"Auto-scaled FFT size: {config.fft_size} -> {effective_fft_size}")
    else:
        effective_fft_size = config.fft_size

    target_average_fft = __average_fft(
        target_loudest_pieces, config.internal_sample_rate, config.fft_size, auto_scale_fft
    )
    reference_average_fft = __average_fft(
        reference_loudest_pieces, config.internal_sample_rate, config.fft_size, auto_scale_fft
    )

    np.maximum(config.min_value, target_average_fft, out=target_average_fft)
    matching_fft = reference_average_fft / target_average_fft

    # Pass effective FFT size to smoothing function
    matching_fft_filtered = __smooth_exponentially(matching_fft, config, effective_fft_size)

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


# Original Matchering stages functions
def __match_levels(
    target: np.ndarray, reference: np.ndarray, config: Config
) -> (
    np.ndarray, 
    np.ndarray, 
    float, 
    np.ndarray, 
    np.ndarray, 
    np.ndarray, 
    np.ndarray, 
    np.ndarray, 
    np.ndarray,  # Add reference_mid, reference_side
    float,
    float, 
    float,
):
    debug_line()
    info(Code.INFO_MATCHING_LEVELS)

    debug(
        f"The maximum size of the analyzed piece: {config.max_piece_size} samples "
        f"or {config.max_piece_size / config.internal_sample_rate:.2f} seconds"
    )

    reference, final_amplitude_coefficient = normalize_reference(reference, config)

    (
        target_mid,
        target_side,
        target_mid_loudest_pieces,
        target_side_loudest_pieces,
        target_match_rms,
        target_divisions,
        target_piece_size,
    ) = analyze_levels(target, "target", config)

    (
        reference_mid,
        reference_side,
        reference_mid_loudest_pieces,
        reference_side_loudest_pieces,
        reference_match_rms,
        *_,
    ) = analyze_levels(reference, "reference", config)

    rms_coefficient, target_mid, target_side = get_rms_c_and_amplify_pair(
        target_mid,
        target_side,
        target_match_rms,
        reference_match_rms,
        config.min_value,
        "target",
    )

    debug("Modifying the amplitudes of the extracted loudest TARGET pieces...")
    target_mid_loudest_pieces = amplify(target_mid_loudest_pieces, rms_coefficient)
    target_side_loudest_pieces = amplify(target_side_loudest_pieces, rms_coefficient)

    return (
        target_mid, target_side,
        final_amplitude_coefficient,
        target_mid_loudest_pieces,
        target_side_loudest_pieces,
        reference_mid_loudest_pieces, 
        reference_side_loudest_pieces,
        reference_mid, reference_side,  # Add full reference channels
        target_divisions,
        target_piece_size,
        reference_match_rms,
    )

def __match_frequencies(
    target_mid: np.ndarray,
    target_side: np.ndarray,
    target_mid_loudest_pieces: np.ndarray,
    reference_mid_loudest_pieces: np.ndarray,
    target_side_loudest_pieces: np.ndarray,
    reference_side_loudest_pieces: np.ndarray,
    reference_mid: np.ndarray,      # Add this
    reference_side: np.ndarray,     # Add this
    config: Config,
) -> (np.ndarray, np.ndarray):
    debug_line()
    info(Code.INFO_MATCHING_FREQS)

    # Check for enhanced method settings
    method = getattr(config, 'analysis_method', 'loudest')
    percentile = getattr(config, 'analysis_percentile', 75.0)
    
    debug(f"=== Method detection: {method}, Percentile: {percentile} ===")
    
    if method in ['median', 'percentile']:
        debug(f"=== Using ENHANCED {method} method ===")
        # Use full target audio + reference loudest pieces for enhanced methods
        mid_fir = get_fir_enhanced(
            target_mid_loudest_pieces, reference_mid_loudest_pieces,
            target_mid, reference_mid, "mid", config, 
            method=method, percentile=percentile, auto_scale_fft=config.auto_scale_fft
        )
        side_fir = get_fir_enhanced(
            target_side_loudest_pieces, reference_side_loudest_pieces,
            target_side, reference_side, "side", config, 
            method=method, percentile=percentile, auto_scale_fft=config.auto_scale_fft
        )
    else:
        debug(f"=== Using ORIGINAL method ===")
        mid_fir = get_fir(target_mid_loudest_pieces, reference_mid_loudest_pieces, "mid", config)
        side_fir = get_fir(target_side_loudest_pieces, reference_side_loudest_pieces, "side", config)

    del (target_mid_loudest_pieces, reference_mid_loudest_pieces, 
         target_side_loudest_pieces, reference_side_loudest_pieces)

    result, result_mid = convolve(target_mid, mid_fir, target_side, side_fir)
    return result, result_mid


def __correct_levels(
    result: np.ndarray,
    result_mid: np.ndarray,
    target_divisions: int,
    target_piece_size: int,
    reference_match_rms: float,
    config: Config,
) -> np.ndarray:
    debug_line()
    info(Code.INFO_CORRECTING_LEVELS)

    for step in range(1, config.rms_correction_steps + 1):
        debug(f"Applying RMS correction #{step}...")
        result_mid_clipped = clip(result_mid)

        _, clipped_rmses, clipped_average_rms = get_average_rms(
            result_mid_clipped, target_piece_size, target_divisions, "result"
        )

        _, result_mid_clipped_match_rms = get_lpis_and_match_rms(
            clipped_rmses, clipped_average_rms
        )

        rms_coefficient, result_mid, result = get_rms_c_and_amplify_pair(
            result_mid,
            result,
            result_mid_clipped_match_rms,
            reference_match_rms,
            config.min_value,
            "result",
        )

    return result


def __finalize(
    result_no_limiter: np.ndarray,
    final_amplitude_coefficient: float,
    need_default: bool,
    need_no_limiter: bool,
    need_no_limiter_normalized: bool,
    config: Config,
) -> (np.ndarray, np.ndarray, np.ndarray):
    debug_line()
    info(Code.INFO_FINALIZING)

    result_no_limiter_normalized = None
    if need_no_limiter_normalized:
        result_no_limiter_normalized, coefficient = normalize(
            result_no_limiter,
            config.threshold,
            config.min_value,
            normalize_clipped=True,
        )
        debug(
            f"The amplitude of the normalized RESULT should be adjusted by {to_db(coefficient)}"
        )
        if not np.isclose(final_amplitude_coefficient, 1.0):
            debug(
                f"And by {to_db(final_amplitude_coefficient)} after applying some brickwall limiter to it"
            )

    result = None
    if need_default:
        result = limit(result_no_limiter, config)
        result = amplify(result, final_amplitude_coefficient)

    result_no_limiter = result_no_limiter if need_no_limiter else None

    return result, result_no_limiter, result_no_limiter_normalized


def main(
    target: np.ndarray,
    reference: np.ndarray,
    config: Config,
    need_default: bool = True,
    need_no_limiter: bool = False,
    need_no_limiter_normalized: bool = False,
) -> (np.ndarray, np.ndarray, np.ndarray):
    (target_mid, target_side, final_amplitude_coefficient,
     target_mid_loudest_pieces, target_side_loudest_pieces,
     reference_mid_loudest_pieces, reference_side_loudest_pieces,
     reference_mid, reference_side,  # Add these
     target_divisions, target_piece_size, reference_match_rms,
    ) = __match_levels(target, reference, config)
    
    del target, reference

    result_no_limiter, result_no_limiter_mid = __match_frequencies(
    target_mid,
    target_side,
    target_mid_loudest_pieces,
    reference_mid_loudest_pieces,
    target_side_loudest_pieces,
    reference_side_loudest_pieces,
    reference_mid,      # Add this
    reference_side,     # Add this
    config,
)

    del (
        target_mid,
        target_side,
        target_mid_loudest_pieces,
        reference_mid_loudest_pieces,
        target_side_loudest_pieces,
        reference_side_loudest_pieces,
    )

    result_no_limiter = __correct_levels(
        result_no_limiter,
        result_no_limiter_mid,
        target_divisions,
        target_piece_size,
        reference_match_rms,
        config,
    )

    del result_no_limiter_mid

    result, result_no_limiter, result_no_limiter_normalized = __finalize(
        result_no_limiter,
        final_amplitude_coefficient,
        need_default,
        need_no_limiter,
        need_no_limiter_normalized,
        config,
    )

    return result, result_no_limiter, result_no_limiter_normalized