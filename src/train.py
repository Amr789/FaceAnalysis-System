import os
import glob
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.dataset import UTKFaceDataset, get_transforms
from src.networks import AgeEstimator

def train(dataset_path, epochs, batch_size, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Data Prep
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

    # Model Setup
    model = AgeEstimator(pretrained=True).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Loop
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, ages in train_loader:
            images, ages = images.to(device), ages.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images).squeeze(), ages)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, ages in val_loader:
                images, ages = images.to(device), ages.to(device)
                val_loss += criterion(model(images).squeeze(), ages).item()

        avg_train = running_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} | Train MAE: {avg_train:.2f} | Val MAE: {avg_val:.2f}")
        
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved better model to {save_path}")
            
        scheduler.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to UTKFace folder")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--save", type=str, default="models/utk_age_model.pth")
    args = parser.parse_args()
    
    os.makedirs("models", exist_ok=True)
    train(args.data, args.epochs, args.batch, args.save)