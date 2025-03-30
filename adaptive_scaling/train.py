import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt

# Custom Dataset
class MaxDepthDataset(Dataset):
    def __init__(self, data_file):
        self.data = np.load(data_file, allow_pickle=True)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, log_scale_factor = self.data[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        return image, torch.tensor(float(log_scale_factor), dtype=torch.float32)

# Neural Network Model
class MaxDepthPredictor(nn.Module):
    def __init__(self):
        super(MaxDepthPredictor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 56 * 56, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x.squeeze()

# Necessary due to multiprocessing on certain platforms
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)

    # Hyperparameters
    batch_size = 30
    learning_rate = 1e-3
    epochs = 30
    checkpoint_path = "adaptive_scaling/max_depth_predictor2.pth"

    # Load Dataset and Split into Training/Validation
    dataset = MaxDepthDataset("adaptive_scaling/training_data_office_0026.npy")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    # Device setup
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = MaxDepthPredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Load Checkpoint if Exists
    checkpoint_path = "adaptive_scaling/max_depth_predictor2.pth"
    start_epoch = 0
    # if os.path.exists(checkpoint_path):
    #     checkpoint = torch.load(checkpoint_path, map_location=device)
    #     model.load_state_dict(checkpoint["model_state"])
    #     optimizer.load_state_dict(checkpoint["optimizer_state"])
    #     start_epoch = checkpoint["epoch"] + 1
    #     print(f"Resuming training from epoch {start_epoch}")

    # Training Loop
    epochs = 30
    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, epochs):
        model.train()
        total_train_loss = 0

        for images, targets in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Training"):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_loss_avg = total_train_loss / len(train_loader)
        train_losses.append(train_loss_avg)

        print(f"Epoch [{epoch+1}/{epochs}] - Training Loss: {train_loss_avg:.6f}")

        # Perform validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for images, targets in tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] Validation"):
                    images, targets = images.to(device), targets.to(device)
                    predictions = model(images)
                    loss = criterion(predictions, targets)
                    total_val_loss += loss.item()

            val_loss_avg = total_val_loss / len(val_loader)
            val_losses.append(val_loss_avg)
            print(f"Epoch [{epoch+1}/{epochs}] - Validation Loss: {val_loss_avg:.6f}")

            # Save Checkpoint periodically (every 5 epochs)
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict()
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")

        scheduler.step()

    # Plot training and validation losses at the end
    plt.figure(figsize=(8,6))
    plt.plot(range(epochs), train_losses, label="Training Loss")
    plt.plot(np.arange(5, epochs+1, 5), val_losses, label="Validation Loss")  # validation every 5 epochs
    plt.yscale('log')  # Set y-axis to log scale
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Log Scale)")
    plt.legend()
    plt.title("Training & Validation Loss (Log Scale)")
    plt.savefig("adaptive_scaling/loss_curves.png")
    plt.show()

    print("Training completed!")