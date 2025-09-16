import numpy as np
from scipy.signal import resample, butter, filtfilt

def resample_signal(signal: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
    """
    Resample signal to target frequency
    Args:
        signal: Input signal
        orig_fs: Original sampling frequency
        target_fs: Target sampling frequency
    Returns:
        Resampled signal
    """
    if orig_fs == target_fs:
        return signal
    n_samples = int(len(signal) * target_fs / orig_fs)
    return resample(signal, n_samples)

def bandpass_filter(signal: np.ndarray, fs: float, lowcut: float = 0.5,
                   highcut: float = 45.0) -> np.ndarray:
    """
    Apply bandpass filter to signal
    Args:
        signal: Input signal
        fs: Sampling frequency
        lowcut: Lower cutoff frequency
        highcut: Upper cutoff frequency
    Returns:
        Filtered signal
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(N=2, Wn=[low, high], btype='band')
    return filtfilt(b, a, signal)

def add_gaussian_noise(signal: np.ndarray, std_ratio: float = 0.01) -> np.ndarray:
    """
    Add Gaussian noise to signal
    Args:
        signal: Input signal
        std_ratio: Standard deviation ratio for noise
    Returns:
        Signal with added noise
    """
    std = np.std(signal)
    noise = np.random.normal(0, std * std_ratio, size=signal.shape)
    return signal + noise

def normalize_signal(sig: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize signal using specified method
    Args:
        sig: Input signal
        method: Normalization method ('zscore' or 'minmax')
    Returns:
        Normalized signal
    """
    if method == 'zscore':
        mean = np.mean(sig)
        std = np.std(sig)
        return (sig - mean) / std if std != 0 else sig - mean
    elif method == 'minmax':
        min_val = np.min(sig)
        max_val = np.max(sig)
        return (sig - min_val) / (max_val - min_val) if max_val != min_val else sig - min_val
    else:
        raise ValueError(f"Unknown normalization method: {method}")