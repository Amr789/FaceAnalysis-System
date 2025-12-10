import os
import glob
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.dataset import UTKFaceDataset, get_transforms
from src.networks import AgeEstimator

def compute_age_weights(dataset_path, max_age=120):
    """
    Scans the dataset to calculate 'Inverse Frequency Weights'.
    Rare ages get higher weights. Common ages get lower weights.
    """
    print("⚖️ Computing age weights to fix imbalance...")
    all_files = glob.glob(os.path.join(dataset_path, "*.jpg"))
    
    # 1. Extract all ages
    ages = []
    for f in all_files:
        try:
            # Parse filename: age_gender_race_date.jpg
            age = int(os.path.basename(f).split('_')[0])
            ages.append(age)
        except:
            continue
            
    # 2. Count frequencies
    counts = np.bincount(ages, minlength=max_age)
    
    # 3. Calculate Weights (Inverse Frequency)
    # Add +1 to counts to avoid division by zero for missing ages
    weights = 1.0 / (counts + 1.0)
    
    # 4. Normalize (so mean weight is 1.0)
    # This ensures our Learning Rate doesn't need huge adjustments
    weights = weights / weights.mean()
    
    # Convert to Tensor
    return torch.FloatTensor(weights)

def train(dataset_path, epochs, batch_size, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # --- 1. PREPARE DATA ---
    all_files = glob.glob(os.path.join(dataset_path, "*.jpg"))
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

    train_loader = DataLoader(
        UTKFaceDataset(train_files, transform=get_transforms('train')),
        batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        UTKFaceDataset(val_files, transform=get_transforms('val')),
        batch_size=batch_size, shuffle=False, num_workers=2
    )

    # --- 2. PREPARE WEIGHTS ---
    # Calculate weights based on the whole dataset
    age_weights = compute_age_weights(dataset_path).to(device)

    # --- 3. MODEL SETUP ---
    model = AgeEstimator(pretrained=True).to(device)
    
    # Key Change: reduction='none' allows us to weight each sample individually
    criterion = nn.L1Loss(reduction='none') 
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # EfficientNet likes lower LR
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # --- 4. TRAINING LOOP ---
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, ages in train_loader:
            images, ages = images.to(device), ages.to(device)
            
            optimizer.zero_grad()
            preds = model(images).squeeze()
            
            # --- WEIGHTED LOSS CALCULATION ---
            # 1. Calculate raw error per image
            raw_loss = criterion(preds, ages)
            
            # 2. Look up the weight for each age in this batch
            # clamp(0, 119) ensures we don't crash if an age is > 119
            batch_weights = age_weights[ages.long().clamp(0, 119)]
            
            # 3. Apply weights and average
            weighted_loss = (raw_loss * batch_weights).mean()
            
            weighted_loss.backward()
            optimizer.step()
            running_loss += weighted_loss.item()

        # Validation (Standard unweighted MAE for fair reporting)
        model.eval()
        val_loss = 0.0
        val_criterion = nn.L1Loss() # Standard MAE for validation
        
        with torch.no_grad():
            for images, ages in val_loader:
                images, ages = images.to(device), ages.to(device)
                val_loss += val_criterion(model(images).squeeze(), ages).item()

        avg_train = running_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} | Train Weighted Loss: {avg_train:.2f} | Val MAE: {avg_val:.2f}")
        
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved better model to {save_path}")
            
        scheduler.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to UTKFace folder")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--save", type=str, default="models/utk_age_model.pth")
    args = parser.parse_args()
    
    os.makedirs("models", exist_ok=True)
    train(args.data, args.epochs, args.batch, args.save)