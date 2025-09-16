import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wfdb
import os
from typing import List, Tuple, Dict
from ..utils.preprocessing import resample_signal, bandpass_filter, normalize_signal

class ECGDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        """
        Initialize ECG dataset
        Args:
            X: Signal data of shape (N, T)
            Y: Labels of shape (N, window_size)
        """
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, T)
        self.Y = torch.tensor(Y, dtype=torch.long)                  # (N, window_size)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]

def load_record(record_path: str, fs_expected: int = 360) -> Tuple[wfdb.Record, wfdb.Annotation]:
    """
    Load a record and its annotations from the specified path
    Args:
        record_path: Path to the record
        fs_expected: Expected sampling frequency
    Returns:
        record: WFDB record
        ann: WFDB annotation
    """
    try:
        record = wfdb.rdrecord(record_path)
        ann = wfdb.rdann(record_path, 'atr')
        if record.fs != fs_expected:
            return None, None
        return record, ann
    except:
        return None, None

def clean_rhythm_label(note: str) -> str:
    """Clean rhythm label from annotation"""
    if note.startswith('('):
        note = note[1:]
    note = note.replace('\x00', '').strip()
    return note

def label_from_aux_note(signal: np.ndarray, ann: wfdb.Annotation, mapping: Dict[str, int]) -> np.ndarray:
    """
    Extract rhythm labels from annotation
    Args:
        signal: ECG signal
        ann: WFDB annotation
        mapping: Dictionary mapping rhythm types to integer labels
    Returns:
        labels: Array of integer labels for each sample
    """
    aux_notes = ann.aux_note
    sample_indices = ann.sample
    labels = np.zeros(len(signal), dtype=int)

    for i in range(len(aux_notes)):
        if not aux_notes[i].startswith('('):
            continue
        note = clean_rhythm_label(aux_notes[i]).upper()
        if note.startswith('V'):
            continue
        start = sample_indices[i]
        labels[start:] = mapping.get(note, 0)

    return labels

def process_record_pipeline(record_path: str, mapping: Dict[str, int],
                          orig_fs: int = 360, target_fs: int = 200,
                          window_sec: int = 20, stride_sec: int = 10,
                          bandpass: Tuple[float, float] = (0.5, 40)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a single ECG record
    Args:
        record_path: Path to the record
        mapping: Dictionary mapping rhythm types to integer labels
        orig_fs: Original sampling frequency
        target_fs: Target sampling frequency
        window_sec: Window size in seconds
        stride_sec: Stride size in seconds
        bandpass: Bandpass filter cutoff frequencies
    Returns:
        X_segments: Processed signal segments
        Y_segments: Corresponding labels
    """
    record, ann = load_record(record_path, fs_expected=orig_fs)
    if record is None:
        return None, None
        
    signal = record.p_signal[:, 0]
    labels = label_from_aux_note(signal, ann, mapping)
    window_len_raw = window_sec * orig_fs
    default_stride_raw = stride_sec * orig_fs
    single_stride_raw = orig_fs
    total_len = len(signal)
    X_segments, Y_segments = [], []

    current_stride = default_stride_raw
    start = 0

    while start + window_len_raw <= total_len:
        end = start + window_len_raw
        x_raw = signal[start:end]
        y_raw = labels[start:end]

        y_sec = y_raw.reshape(window_sec, orig_fs)
        y_hard = [np.bincount(sec, minlength=len(mapping)).argmax() for sec in y_sec]
        y_hard = np.array(y_hard, dtype=np.int64)

        x_res = resample_signal(x_raw, orig_fs, target_fs)
        x_filt = bandpass_filter(x_res, fs=target_fs, lowcut=bandpass[0], highcut=bandpass[1])
        x_norm = normalize_signal(x_filt, 'minmax')

        X_segments.append(x_norm.astype(np.float32))
        Y_segments.append(y_hard)

        if np.all(y_hard == y_hard[0]):
            current_stride = default_stride_raw
        else:
            current_stride = single_stride_raw

        start += current_stride

    return np.stack(X_segments), np.stack(Y_segments)