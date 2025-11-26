import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict, Any

# --- Configuration ---
class Config:
    """Configuration for the Lung Segmentation training pipeline."""
    SEED = 42
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 20
    NUM_WORKERS = 0
    PIN_MEMORY = True
    LOAD_MODEL = False
    TRAIN_IMG_DIR = "Train/Images/"
    TRAIN_MASK_DIR = "Train/Masks/"
    VAL_IMG_DIR = "Test/Images/"
    VAL_MASK_DIR = "Test/Masks/"
    DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(seed: int = 42):
    """Sets the seed for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Dataset ---
class LungDataset(Dataset):
    """
    Custom Dataset for Lung Segmentation.
    Expects images and masks in separate directories.
    """
    def __init__(self, image_dir: str, mask_dir: str, transform: Optional[A.Compose] = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir) if os.path.exists(image_dir) else []

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index]) # Assuming same filename

        # Load image and mask
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        
        # Preprocess mask: 0.0 or 1.0
        mask[mask == 255.0] = 1.0
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

# --- Data Augmentation ---
def get_train_transforms(height: int, width: int) -> A.Compose:
    """Returns the training data augmentation pipeline."""
    return A.Compose(
        [
            A.Resize(height=height, width=width),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

def get_val_transforms(height: int, width: int) -> A.Compose:
    """Returns the validation data augmentation pipeline (resize & normalize only)."""
    return A.Compose(
        [
            A.Resize(height=height, width=width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

# --- Model: Residual U-Net ---
class ResidualBlock(nn.Module):
    """Residual Block with two convolution layers."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResUNet(nn.Module):
    """
    Residual U-Net Architecture.
    Combines the strengths of U-Net (skip connections) and ResNet (residual learning).
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: List[int] = [64, 128, 256, 512]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Downsampling)
        for feature in features:
            self.downs.append(ResidualBlock(in_channels, feature))
            in_channels = feature

        # Decoder (Upsampling)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(ResidualBlock(feature * 2, feature))

        self.bottleneck = ResidualBlock(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = transforms.functional.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

# --- Loss Function ---
class DiceBCELoss(nn.Module):
    """
    Combination of Binary Cross Entropy Loss and Dice Loss.
    Helps with class imbalance and improves segmentation quality (IoU).
    """
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        # Flatten inputs and targets
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Dice Loss
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        # BCE Loss
        bce_loss = nn.BCELoss()(inputs, targets)
        
        return bce_loss + dice_loss

# --- Metrics ---
def calculate_iou(pred_mask: torch.Tensor, true_mask: torch.Tensor, threshold: float = 0.5) -> float:
    """Calculates Intersection over Union (IoU) score."""
    pred_mask = (torch.sigmoid(pred_mask) > threshold).float()
    
    intersection = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum() - intersection
    
    if union == 0:
        return 1.0
    
    return (intersection / union).item()

# --- Training Loop ---
def train_fn(loader: DataLoader, model: nn.Module, optimizer: optim.Optimizer, loss_fn: nn.Module, scaler: torch.cuda.amp.GradScaler):
    """Training function for one epoch."""
    model.train()
    loop = tqdm(loader, desc="Training")
    epoch_loss = 0
    epoch_iou = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(Config.DEVICE)
        targets = targets.float().unsqueeze(1).to(Config.DEVICE)

        # Forward
        if Config.DEVICE == "cuda":
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(predictions, targets)
        else:
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward
        optimizer.zero_grad()
        if Config.DEVICE == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Metrics
        iou = calculate_iou(predictions, targets)
        epoch_loss += loss.item()
        epoch_iou += iou

        # Update tqdm loop
        loop.set_postfix(loss=loss.item(), iou=iou)
    
    return epoch_loss / len(loader), epoch_iou / len(loader)

def check_accuracy(loader: DataLoader, model: nn.Module, device: str = "cuda"):
    """Evaluates the model on validation set."""
    model.eval()
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    iou_score = 0
    
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            
            # Dice Score
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
            
            # IoU Score
            intersection = (preds * y).sum()
            union = preds.sum() + y.sum() - intersection
            iou_score += (intersection + 1e-8) / (union + 1e-8)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    print(f"IoU score: {iou_score/len(loader)}")
    
    return iou_score/len(loader)

def save_predictions_as_imgs(loader: DataLoader, model: nn.Module, folder: str = "saved_images/", device: str = "cuda"):
    """Saves predicted masks as images."""
    os.makedirs(folder, exist_ok=True)
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        
        # Save only first batch to avoid clutter
        if idx == 0:
            for i in range(x.shape[0]):
                torchvision.utils.save_image(preds[i], f"{folder}/pred_{i}.png")
                torchvision.utils.save_image(y[i].unsqueeze(0), f"{folder}/true_{i}.png")
            break

# --- Main ---
def main():
    seed_everything(Config.SEED)
    
    # Initialize model, loss, optimizer
    model = ResUNet(in_channels=3, out_channels=1).to(Config.DEVICE)
    loss_fn = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    # Data Loaders
    # Note: Users need to set Config.TRAIN_IMG_DIR etc. correctly
    # For demonstration, we assume the directories exist or user will set them.
    
    # Check if data directories exist, if not, print warning
    if not os.path.exists(Config.TRAIN_IMG_DIR):
        print(f"Warning: Data directory {Config.TRAIN_IMG_DIR} not found.")
        print("Please update Config.TRAIN_IMG_DIR and Config.TRAIN_MASK_DIR.")
        # Create dummy dataset for code verification if needed
        # return 

    train_ds = LungDataset(
        image_dir=Config.TRAIN_IMG_DIR,
        mask_dir=Config.TRAIN_MASK_DIR,
        transform=get_train_transforms(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
    )

    val_ds = LungDataset(
        image_dir=Config.VAL_IMG_DIR,
        mask_dir=Config.VAL_MASK_DIR,
        transform=get_val_transforms(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        shuffle=False,
    )

    # Training Loop
    best_iou = 0.0
    for epoch in range(Config.NUM_EPOCHS):
        print(f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}]")
        loss, iou = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
        # Check accuracy
        val_iou = check_accuracy(val_loader, model, device=Config.DEVICE)
        
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), "best_resunet_model.pth")
            print("Model saved!")

if __name__ == "__main__":
    import torchvision # Imported here to avoid issues if not installed, though it should be
    main()
