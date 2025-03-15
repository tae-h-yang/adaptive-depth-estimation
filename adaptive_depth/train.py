import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms

# Custom Dataset
class MaxDepthDataset(Dataset):
    def __init__(self, data_file):
        self.data = np.load(data_file, allow_pickle=True)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),  # Resize to match model input size
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, scale_factor = self.data[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        return image, torch.tensor(float(scale_factor), dtype=torch.float32)

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

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)

    # Hyperparameters
    batch_size = 16
    learning_rate = 1e-3
    epochs = 30
    checkpoint_path = "max_depth/max_depth_predictor3.pth"

    # Load Dataset
    dataset = MaxDepthDataset("max_depth/training_data_office_0026.npy")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize Model
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = MaxDepthPredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Load Checkpoint if Exists
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch}...")

    # Training Loop
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0

        for images, targets in tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]"):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)  # Gradient clipping
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss / len(dataloader):.6f}")

        # Save Checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict()
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")

    print("Training completed!")
