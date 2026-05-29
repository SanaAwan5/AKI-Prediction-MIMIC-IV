"""

Uses FLamby's fed_heart_disease baseline model, loss, and parameters
Loads MIMIC-IV AKI data (leakage columns removed in assign_iid_centers.py)
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

import sys
import os.path as osp
parent_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, parent_dir)

from model import Baseline as BaselineOriginal
from loss import BaselineLoss
from metric import metric
from common import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Optimizer
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# UPDATE THIS PATH TO YOUR CSV FILE!
DATA_PATH = "/Users/awans/Documents/GitHub/AKI_FL_Project/aki_features_iid.csv"

# Number of centers in your data
NUM_CLIENTS = 10


class FedHeartDisease(Dataset):
    """
    MIMIC-IV AKI Dataset for Federated Learning
    Compatible with FLamby's benchmark infrastructure
    
    CSV contains:
    - Features: admission_type, insurance, gender, age, baseline_scr, etc.
    - center_id: Used to SPLIT data by center (not a feature)
    - AKI_label: Target variable (not a feature)
    
    Leakage columns already removed in assign_iid_centers.py:
    - aki_criterion (removed)
    - max_scr_during_admission (removed)
    """
    
    def __init__(self, center=0, train=True, pooled=False):
        """
        Args:
            center (int): Center ID (0-9)
            train (bool): Training set if True, test if False
            pooled (bool): Use all centers if True
        """
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(
                f"Data file not found: {DATA_PATH}\n"
                f"Please update DATA_PATH in dataset_2.py"
            )
        
        # Load CSV
        df = pd.read_csv(DATA_PATH)
        
        # Filter by center (uses center_id column)
        if not pooled:
            df = df[df['center_id'] == center].copy()
        
        # Train/test split (80/20)
        np.random.seed(42)
        mask = np.random.rand(len(df)) < 0.8
        df = df[mask if train else ~mask].copy()
        
        # Extract labels (y)
        self.labels = df['AKI_label'].values.astype(np.float32)
        
        # ================================================================
        # EXTRACT FEATURES (X)
        # Exclude:
        #   - AKI_label
        #   - center_id: Not a feature :)
        # ================================================================
        exclude_cols = ['AKI_label', 'center_id']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        df_features = df[feature_cols].copy()
        
        # One-hot encode categorical columns
        categorical_cols = df_features.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            df_features = pd.get_dummies(df_features, columns=categorical_cols, drop_first=True)
        
        # Convert to numeric
        for col in df_features.columns:
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
        
        df_features = df_features.fillna(0)
        
        # Convert to numpy
        self.features = df_features.values.astype(np.float32)
        
        # Normalize (z-score)
        self.mean = self.features.mean(axis=0)
        self.std = self.features.std(axis=0) + 1e-8
        self.features = (self.features - self.mean) / self.std
        
        self.num_features = self.features.shape[1]
        self.feature_names = list(df_features.columns)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        X = torch.from_numpy(self.features[idx])
        y = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return X, y

class Baseline(BaselineOriginal):
    """
    FLamby's Baseline model adapted for MIMIC-IV AKI feature count
    """
    def __init__(self):
        # Get actual number of features from your data
        num_features = get_num_features()
        # Initialize with correct input dimension
        super().__init__(input_dim=num_features)



def get_num_samples(center=0, train=True, pooled=False):
    """Get number of samples for a center"""
    dataset = FedHeartDisease(center=center, train=train, pooled=pooled)
    return len(dataset)


def get_num_features():
    """Get number of features (after one-hot encoding)"""
    dataset = FedHeartDisease(center=0, train=True)
    return dataset.num_features


def get_nb_max_rounds(num_updates, batch_size=BATCH_SIZE):
    """Calculate number of rounds (for FLamby compatibility)"""
    return num_updates


def print_dataset_info():
    """Print dataset information"""
    print("="*80)
    print("MIMIC-IV AKI DATASET (CLEAN - NO LEAKAGE)")
    print("="*80)
    print(f"\nData: {DATA_PATH}")
    
    # Check CSV columns
    df_check = pd.read_csv(DATA_PATH)
    print(f"\nColumns in CSV ({len(df_check.columns)} total):")
    for i, col in enumerate(df_check.columns, 1):
        marker = ""
        if col == 'AKI_label':
            marker = " ← Target variable"
        elif col == 'center_id':
            marker = " ← For data splitting"
        print(f"  {i}. {col}{marker}")
    
    print(f"\nFeatures after one-hot encoding: {get_num_features()}")
    
    print(f"\n{'Center':<10} {'Train':<12} {'Test':<12} {'Total':<12} {'AKI %'}")
    print("-" * 60)
    
    for c in range(NUM_CLIENTS):
        train_size = get_num_samples(center=c, train=True)
        test_size = get_num_samples(center=c, train=False)
        total = train_size + test_size
        
        dataset = FedHeartDisease(center=c, train=True)
        aki_pct = (dataset.labels.sum() / len(dataset.labels) * 100) if len(dataset.labels) > 0 else 0
        
        print(f"{c:<10} {train_size:<12,} {test_size:<12,} {total:<12,} {aki_pct:>6.2f}%")
    
    pooled_train = get_num_samples(pooled=True, train=True)
    pooled_test = get_num_samples(pooled=True, train=False)
    dataset_pooled = FedHeartDisease(pooled=True, train=True)
    pooled_aki = (dataset_pooled.labels.sum() / len(dataset_pooled.labels) * 100)
    
    print("-" * 60)
    print(f"{'TOTAL':<10} {pooled_train:<12,} {pooled_test:<12,} {pooled_train+pooled_test:<12,} {pooled_aki:>6.2f}%")
    print("="*80)


# FedClass for FLamby compatibility
FedClass = FedHeartDisease

# Export all required components for FLamby's fed_benchmark.py
__all__ = [
    'FedHeartDisease',      # Your custom dataset
    'FedClass',             # Alias for compatibility
    'Baseline',             # FLamby's model (adapted)
    'BaselineLoss',         # FLamby's loss
    'metric',               # FLamby's metric
    'BATCH_SIZE',           # FLamby's batch size
    'LR',                   # FLamby's learning rate
    'NUM_CLIENTS',          # Your number of centers
    'NUM_EPOCHS_POOLED',    # FLamby's epochs
    'Optimizer',            # FLamby's optimizer
    'get_nb_max_rounds',    # FLamby compatibility
    'get_num_features',     # Helper function
    'get_num_samples',      # Helper function
]


if __name__ == "__main__":
    print_dataset_info()