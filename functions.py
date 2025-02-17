# Import modules
import numpy as np
import pywt
import scipy.io
from scipy.signal import butter, filtfilt

def load_mat_file(file_path):
    """Load datasets of file type .mat."""

    # Load .mat file and extract the content
    data = scipy.io.loadmat(file_path)

    # Flatten 'd' for easier processing
    d = data.get("d", None).flatten()

    # Extract variables if they exist in the .mat file
    Index = data.get("Index", None).flatten() if "Index" in data else None
    Class = data.get("Class", None).flatten() if "Class" in data else None
    
    return d, Index, Class

def bandpass_filter(data, lowcut=300, highcut=3000, fs=25000, order=5):
    """Apply bandpass filter to data using butterworth filter."""
    
    # Nyquist frequency
    nyq = 0.5 * fs

    # Normalise the cutoff frequencies
    low = lowcut / nyq
    high = highcut / nyq 
    
    # Design Butterworth bandpass filter
    b, a = butter(order, [low, high], btype="band") # Filter coefficients
    
    # Apply the filter twice to prevent phase distortion
    return filtfilt(b, a, data)

def wavelet_denoising(data, wavelet="db4", level=3, threshold_factor=1.0):
    """Apply wavelet-based denoising to the signal."""

    # Decompose signal into wavelet coefficients
    coeffs = pywt.wavedec(data, wavelet, level=level)

    # Estimate noise level from the detail coefficients at the finest scale
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = threshold_factor * sigma

    # Apply thresholding to detail coefficients
    denoised_coeffs = [pywt.threshold(c, threshold, mode="soft") if i > 0 else c for i, c in enumerate(coeffs)]

    # Reconstruct the signal
    denoised_data = pywt.waverec(denoised_coeffs, wavelet)
    return denoised_data

def detect_spikes(data, threshold):
    """Detect spikes based on threshold with refractory period to avoid overlaps."""

    spike_indices = []
    refractory_period = 25  # Minimum distance between consecutive spikes
    
    for i in range(len(data)):
        # Check if the current data point exceeds the threshold
        if data[i] > threshold:
            # Ensure no recent spike was detected within the refractory period
            if not spike_indices or (i - spike_indices[-1] > refractory_period):
                spike_indices.append(i)

    return spike_indices

def baseline_correction(data, window_size=100):
    """Baseline correction using a moving average filter."""
    
    # Correct baseline by substracting the moving average
    baseline = np.convolve(data, np.ones(window_size)/window_size, mode="same")
    corrected_data = data - baseline
    
    return corrected_data

def extract_features(data, indices, window_size=20):
    """Extract data features around spike indices within a specified window size."""
    
    features = []
    for idx in indices:
        # Handle boundary conditions by padding the data
        start_idx = max(0, idx - window_size)
        end_idx = min(len(data), idx + window_size)
        
        # Extract the segment and pad with zeros if too short
        segment = data[start_idx:end_idx]
        if len(segment) < 2 * window_size:
            segment = np.pad(segment, (0, 2 * window_size - len(segment)), mode="constant")
        
        features.append(segment)
    
    return np.array(features)

def save_results(file_path, spike_indices, spike_classes):
    """Save spike indices and classes to a .mat file."""
    
    # Save output to .mat file
    scipy.io.savemat(file_path, {"Index": spike_indices, "Class": spike_classes})