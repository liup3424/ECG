import torch
import numpy as np
from typing import List, Tuple, Optional
import os
from ..data.dataset import load_record
from ..utils.preprocessing import resample_signal, bandpass_filter, normalize_signal

def predict_full_record(model: torch.nn.Module,
                       signal: np.ndarray,
                       orig_fs: int = 360,
                       target_fs: int = 200,
                       window_sec: int = 20,
                       stride_sec: int = 10,
                       bandpass: Tuple[float, float] = (0.5, 40)) -> np.ndarray:
    """
    Make predictions for a full ECG record
    Args:
        model: Trained model
        signal: Input signal
        orig_fs: Original sampling frequency
        target_fs: Target sampling frequency
        window_sec: Window size in seconds
        stride_sec: Stride size in seconds
        bandpass: Bandpass filter parameters
    Returns:
        Predictions for each second
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    win_len_raw = window_sec * orig_fs
    stride_raw = stride_sec * orig_fs
    X_windows = []

    for start in range(0, len(signal), stride_raw):
        if start + win_len_raw > len(signal):
            break
        x_raw = signal[start:start + win_len_raw]
        x_res = resample_signal(x_raw, orig_fs, target_fs)
        x_filt = bandpass_filter(x_res, target_fs, *bandpass)
        x_norm = normalize_signal(x_filt)
        X_windows.append(x_norm.astype(np.float32))

    X_tensor = torch.tensor(np.stack(X_windows)).unsqueeze(1).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    num_windows, win_len, num_classes = probs.shape
    total_len = (num_windows - 1) * stride_sec + win_len
    result = np.zeros((total_len, num_classes))
    counts = np.zeros((total_len, 1))

    for i in range(num_windows):
        start = i * stride_sec
        end = start + win_len
        result[start:end] += probs[i]
        counts[start:end] += 1

    avg_probs = result / counts
    pred_labels = np.argmax(avg_probs, axis=1)
    return pred_labels

def extract_segments_from_labels(labels: np.ndarray,
                               target_classes: Optional[Tuple[int, ...]] = None) -> List[Tuple[int, int, int]]:
    """
    Extract continuous segments from labels
    Args:
        labels: Array of labels
        target_classes: Classes to consider (if None, consider all)
    Returns:
        List of (start, end, label) tuples
    """
    segments = []
    start, current_label = None, None

    for i, lbl in enumerate(labels):
        if target_classes is None or lbl in target_classes:
            if start is None:
                start = i
                current_label = lbl
            elif lbl != current_label:
                segments.append((start, i - 1, current_label))
                start = i
                current_label = lbl
        else:
            if start is not None:
                segments.append((start, i - 1, current_label))
                start, current_label = None, None

    if start is not None:
        segments.append((start, len(labels) - 1, current_label))
    return segments

def evaluate_segments(pred_segments: List[Tuple[int, int, int]],
                     true_segments: List[Tuple[int, int, int]],
                     min_length: int = 15) -> Tuple[float, float]:
    """
    Evaluate prediction segments against true segments
    Args:
        pred_segments: Predicted segments (start, end, label)
        true_segments: True segments (start, end, label)
        min_length: Minimum segment length to consider
    Returns:
        Tuple of (match_ratio, average_overlap)
    """
    match_count = 0
    total_overlap = []

    for t_start, t_end, t_label in true_segments:
        if t_end - t_start + 1 < min_length:
            continue

        matched = False
        best_overlap = 0
        for p_start, p_end, p_label in pred_segments:
            if p_label != t_label:
                continue
            overlap_start = max(t_start, p_start)
            overlap_end = min(t_end, p_end)
            if overlap_start <= overlap_end:
                matched = True
                overlap = overlap_end - overlap_start + 1
                length = t_end - t_start + 1
                best_overlap = max(best_overlap, overlap / length)
        
        if matched:
            match_count += 1
        total_overlap.append(best_overlap)

    metric1 = match_count / len(true_segments) if true_segments else 0
    metric2 = np.mean(total_overlap) if total_overlap else 0
    return metric1, metric2

def evaluate_single_record(model: torch.nn.Module,
                         record_path: str,
                         orig_fs: int = 360,
                         target_fs: int = 200,
                         window_sec: int = 20,
                         stride_sec: int = 10,
                         bandpass: Tuple[float, float] = (0.5, 40)) -> Tuple[float, float]:
    """
    Evaluate model performance on a single record
    """
    record, ann = load_record(record_path, fs_expected=orig_fs)
    if record is None:
        return 0.0, 0.0
        
    signal = record.p_signal[:, 0]
    pred_labels = predict_full_record(model, signal, orig_fs, target_fs,
                                    window_sec, stride_sec, bandpass)

    # Extract true labels (implement this based on your annotation format)
    true_labels = np.zeros(len(signal) // orig_fs)  # Placeholder

    pred_segments = extract_segments_from_labels(pred_labels)
    true_segments = extract_segments_from_labels(true_labels)

    return evaluate_segments(pred_segments, true_segments)

def evaluate_test_set(model: torch.nn.Module,
                     test_record_list: List[str],
                     base_path: str) -> Tuple[float, float]:
    """
    Evaluate model performance on a test set
    """
    scores1, scores2 = [], []
    for rid in test_record_list:
        print(f"Evaluating record {rid}")
        record_path = os.path.join(base_path, str(rid))
        m1, m2 = evaluate_single_record(model, record_path)
        print(f"{rid}: overlap-match = {m1:.2f}, avg-coverage = {m2:.2f}")
        scores1.append(m1)
        scores2.append(m2)

    avg_score1 = np.mean(scores1)
    avg_score2 = np.mean(scores2)
    print("==== Overall Evaluation ====")
    print(f"Metric 1 (≥1s overlap): {avg_score1:.4f}")
    print(f"Metric 2 (avg coverage): {avg_score2:.4f}")
    
    return avg_score1, avg_score2