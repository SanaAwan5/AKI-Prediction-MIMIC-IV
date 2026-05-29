#!/usr/bin/env python3
"""
Run FL experiments using FLamby's baseline components

"""

import sys
sys.path.insert(0, '/Users/awans/Documents/GitHub/FLamby')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import os
from datetime import datetime
import argparse

# Import FLamby components and your dataset
from flamby.datasets.fed_heart_disease.dataset_2 import (
    FedHeartDisease,
    Baseline,
    BaselineLoss,
    metric,
    BATCH_SIZE,
    LR,
    NUM_CLIENTS,
    NUM_EPOCHS_POOLED,
    Optimizer
)


def load_config(config_path='/Users/awans/Documents/GitHub/AKI_FL_Project/config_flamby_benchmark.json'):
    """Load configuration file"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

CONFIG = load_config()
NUM_ROUNDS = CONFIG.get('strategy', {}).get('nrounds', 50)
LOCAL_EPOCHS = CONFIG.get('strategy', {}).get('nlocal', 5)
SEED = CONFIG.get('seed', 42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)

print("="*80)
print("FEDERATED LEARNING WITH FLAMBY BASELINE")
print("="*80)
print(f"Using FLamby's fed_heart_disease baseline components")
print(f"With MIMIC-IV AKI data (221k patients)")
print("="*80)
print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LR}")
print(f"Optimizer: {Optimizer.__name__}")
print(f"Centers: {NUM_CLIENTS}")
print(f"Rounds: {NUM_ROUNDS}, Local epochs: {LOCAL_EPOCHS}")
print(f"Seed: {SEED}")
print("="*80)

# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, device):
    """Evaluate model using FLamby's metric"""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            all_preds.extend(out.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    preds = np.array(all_preds).flatten()
    labels = np.array(all_labels).flatten()
    
    # Use FLamby's metric function
    try:
        metric_value = metric(labels, preds)
    except:
        # Fallback to sklearn if FLamby's metric fails
        from sklearn.metrics import roc_auc_score
        metric_value = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.0
    
    return metric_value

# ============================================================================
# FEDAVG
# ============================================================================

def fedavg():
    """Federated Averaging using FLamby's Baseline"""
    print("\n" + "="*80)
    print("FEDAVG - FLamby's Baseline Model")
    print("="*80)
    
    # Load data
    train_loaders = []
    test_loaders = []
    
    print("\nLoading data...")
    for c in range(NUM_CLIENTS):
        train_ds = FedHeartDisease(center=c, train=True)
        test_ds = FedHeartDisease(center=c, train=False)
        train_loaders.append(DataLoader(train_ds, BATCH_SIZE, shuffle=True))
        test_loaders.append(DataLoader(test_ds, BATCH_SIZE))
        print(f"  Center {c}: {len(train_ds):,} train, {len(test_ds):,} test")
    
    # Initialize model
    global_model = Baseline().to(DEVICE)
    criterion = BaselineLoss()
    
    print(f"\nModel: {global_model.__class__.__name__}")
    print(f"Loss: {criterion.__class__.__name__}")
    print(f"Metric: AUC")
    
    results = {'config': CONFIG, 'rounds': [], 'metrics': []}
    
    print(f"\nTraining for {NUM_ROUNDS} rounds...")
    
    for round_num in range(1, NUM_ROUNDS + 1):
        local_models, local_weights = [], []
        
        # Local training at each center
        for c in range(NUM_CLIENTS):
            local_model = Baseline().to(DEVICE)
            local_model.load_state_dict(global_model.state_dict())
            optimizer = Optimizer(local_model.parameters(), lr=LR)
            
            for _ in range(LOCAL_EPOCHS):
                train_epoch(local_model, train_loaders[c], optimizer, criterion, DEVICE)
            
            local_models.append(local_model.state_dict())
            local_weights.append(len(train_loaders[c].dataset))
        
        # Aggregate models
        total = sum(local_weights)
        global_dict = global_model.state_dict()
        
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])
            for i, local in enumerate(local_models):
                global_dict[key] += local[key] * (local_weights[i] / total)
        
        global_model.load_state_dict(global_dict)
        
        # Evaluate
        if round_num % 5 == 0 or round_num == NUM_ROUNDS:
            metrics = [evaluate(global_model, test_loaders[c], DEVICE) for c in range(NUM_CLIENTS)]
            avg_metric = np.mean(metrics)
            print(f"  Round {round_num}/{NUM_ROUNDS}: AUC={avg_metric:.4f}")
            results['rounds'].append(round_num)
            results['metrics'].append(avg_metric)
    
    return results

# ============================================================================
# CENTRALIZED
# ============================================================================

def centralized():
    """Centralized training using FLamby's Baseline"""
    print("\n" + "="*80)
    print("CENTRALIZED - FLamby's Baseline Model")
    print("="*80)
    
    train_ds = FedHeartDisease(pooled=True, train=True)
    test_ds = FedHeartDisease(pooled=True, train=False)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, BATCH_SIZE)
    
    print(f"\nTrain: {len(train_ds):,}, Test: {len(test_ds):,}")
    
    model = Baseline().to(DEVICE)
    criterion = BaselineLoss()
    optimizer = Optimizer(model.parameters(), lr=LR)
    
    results = {'config': CONFIG, 'epochs': [], 'metrics': []}
    
    print(f"\nTraining for {NUM_EPOCHS_POOLED} epochs...")
    
    for epoch in range(1, NUM_EPOCHS_POOLED + 1):
        train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        
        if epoch % 25 == 0 or epoch == NUM_EPOCHS_POOLED:
            metric_value = evaluate(model, test_loader, DEVICE)
            print(f"  Epoch {epoch}/{NUM_EPOCHS_POOLED}: AUC={metric_value:.4f}")
            results['epochs'].append(epoch)
            results['metrics'].append(metric_value)
    
    return results

# ============================================================================
# LOCAL
# ============================================================================

def local_training():
    """Local training (no federation) using FLamby's Baseline"""
    print("\n" + "="*80)
    print("LOCAL - FLamby's Baseline Model (No Federation)")
    print("="*80)
    
    all_results = []
    num_epochs = NUM_ROUNDS * LOCAL_EPOCHS
    
    print(f"\nTraining each center independently for {num_epochs} epochs...")
    
    for c in range(NUM_CLIENTS):
        train_ds = FedHeartDisease(center=c, train=True)
        test_ds = FedHeartDisease(center=c, train=False)
        train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, BATCH_SIZE)
        
        model = Baseline().to(DEVICE)
        criterion = BaselineLoss()
        optimizer = Optimizer(model.parameters(), lr=LR)
        
        for _ in range(num_epochs):
            train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        
        metric_value = evaluate(model, test_loader, DEVICE)
        print(f"  Center {c}: AUC={metric_value:.4f}")
        all_results.append(metric_value)
    
    avg_metric = np.mean(all_results)
    print(f"\n  Average across centers: AUC={avg_metric:.4f}")
    
    return {'config': CONFIG, 'results': all_results, 'average': avg_metric}

# ============================================================================
# UTILITIES
# ============================================================================

def save_results(name, results):
    """Save results to JSON file"""
    os.makedirs('results', exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    file = f"results/{name}_{ts}.json"
    
    with open(file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to: {file}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run FL experiments with FLamby baseline')
    parser.add_argument('--strategy', type=str, default='FedAvg',
                       choices=['FedAvg', 'FedAdam','FedYogi','Scaffold','Fedprox', 'Local', 'All'],
                       help='FL strategy to use')
    parser.add_argument('--config', type=str, default='config_dataset.json',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    if args.strategy == 'All':
        print("\n" + "="*80)
        print("RUNNING ALL STRATEGIES")
        print("="*80)
        
        # Run FedAvg
        fed = fedavg()
        save_results('fedavg', fed)
        
        # Run Centralized
        cent = centralized()
        save_results('centralized', cent)
        
        # Run Local
        loc = local_training()
        save_results('local', loc)
        
        # Summary
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        print(f"\n{'Strategy':<15} {'Final AUC':<12} {'Status'}")
        print("-" * 45)
        print(f"{'FedAvg':<15} {fed['metrics'][-1]:<12.4f} ✓")
        print(f"{'Centralized':<15} {cent['metrics'][-1]:<12.4f} ✓")
        print(f"{'Local':<15} {loc['average']:<12.4f} ✓")
        print("="*80)
        
        print("\nKey Finding:")
        gap_fed = cent['metrics'][-1] - fed['metrics'][-1]
        gap_local = cent['metrics'][-1] - loc['average']
        print(f"  Centralized vs FedAvg gap: {gap_fed:.4f}")
        print(f"  Centralized vs Local gap: {gap_local:.4f}")
        print(f"  FedAvg recovered {(1 - gap_fed/gap_local)*100:.1f}% of centralized performance")
        
    elif args.strategy == 'FedAvg':
        results = fedavg()
        save_results('fedavg', results)
        
    elif args.strategy == 'Centralized':
        results = centralized()
        save_results('centralized', results)
        
    elif args.strategy == 'Local':
        results = local_training()
        save_results('local', results)


if __name__ == '__main__':
    main()