import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import numpy as np
from sklearn.model_selection import train_test_split

from src.data.dataset import ECGDataset, process_record_pipeline
from src.models.ecgnet import ECGNet, train_model
from src.models.transformer import ECGTransformer
from src.utils.evaluation import evaluate_test_set

def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def get_model(model_name: str, num_classes: int) -> nn.Module:
    """
    Get model based on name
    Args:
        model_name: Name of the model to use
        num_classes: Number of output classes
    Returns:
        Model instance
    """
    if model_name == 'ecgnet':
        return ECGNet(num_classes=num_classes)
    elif model_name == 'transformer':
        return ECGTransformer(
            input_dim=1,
            d_model=128,
            nhead=8,
            num_layers=4,
            num_classes=num_classes,
            dropout=0.1,
            output_length=20
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train ECG classification model')
    parser.add_argument('--model', type=str, default='ecgnet', choices=['ecgnet', 'transformer'],
                      help='Model architecture to use')
    args = parser.parse_args()
    
    # Configuration
    DATA_PATH = '/fionaLiu/data'
    FS = 360
    WINDOW_SIZE = 20
    STRIDE_SIZE = 10
    MAPPING = {'AFIB': 1, 'AFL': 1}  # binary classification
    RECORD_RANGE = range(100, 235)
    TEST_PERC = 0.2
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    EARLY_STOP_ROUNDS = 10
    MODEL_PATH = f'best_{args.model}.pt'
    
    # Set random seed
    set_seed(42)
    
    # Process all records and split into train/test
    all_data = []
    for rec_id in RECORD_RANGE:
        if rec_id in [102, 104, 107, 217]:  # Skip problematic records
            continue
        record_path = os.path.join(DATA_PATH, str(rec_id))
        X, Y = process_record_pipeline(record_path, MAPPING)
        if X is not None:
            all_data.append((rec_id, X, Y))
    
    # Split records into train/test
    test_size = int(len(all_data) * TEST_PERC)
    test_indices = random.sample(range(len(all_data)), test_size)
    train_indices = [i for i in range(len(all_data)) if i not in test_indices]
    
    # Prepare data
    X_train = np.concatenate([all_data[i][1] for i in train_indices])
    Y_train = np.concatenate([all_data[i][2] for i in train_indices])
    test_records = [all_data[i][0] for i in test_indices]
    
    # Split train into train/val
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size=0.2)
    
    # Create data loaders
    train_loader = DataLoader(ECGDataset(X_tr, Y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ECGDataset(X_val, Y_val), batch_size=BATCH_SIZE)
    
    # Initialize model and training components
    model = get_model(args.model, num_classes=len(MAPPING) + 1)  # +1 for normal class
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Train model
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=NUM_EPOCHS,
        early_stop_rounds=EARLY_STOP_ROUNDS,
        model_path=MODEL_PATH,
        device=device
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    evaluate_test_set(model, test_records, DATA_PATH)

if __name__ == "__main__":
    main()