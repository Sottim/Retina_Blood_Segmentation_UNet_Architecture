import os
import time
from glob import glob

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from data import DriveDataset
from model import build_unet
from loss import DiceLoss, DiceBCELoss
from utils import seeding, create_dir, epoch_time

def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0

    model.train()
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def evaluate(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)
    return epoch_loss

if __name__ == "__main__":
    """Seeding """
    seeding(42)

    """Directory"""
    create_dir("files")

    """load the dataset"""
    train_x = sorted(glob("../new_data/train/image/*"))
    train_y = sorted(glob("../new_data/train/mask/*"))

    valid_x = sorted(glob("../new_data/test/image/*"))
    valid_y = sorted(glob("../new_data/test/mask/*"))

    data_str = f"Dataset Size(Augumented):\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    """ Hyperparameters: For the training set """
    H = 512
    W = 512
    size = (H, W)
    batch_size = 1
    num_epochs = 50
    lr = 1e-4
    checkpoint_path = "files/checkpoint.pth"

    """Dataset and loader: train_x(images) and train_y(mask) goes through DriveDataset in data.py 
    to return them as tensor from numpy array after doing some preprocessing like normalization and transpose."""
    train_dataset = DriveDataset(train_x, train_y)
    valid_dataset = DriveDataset(valid_x, valid_y)

    train_loader = DataLoader(
        dataset= train_dataset,
        batch_size= batch_size,
        shuffle= True,
        num_workers= 2
    )

    validation_loader = DataLoader(
        dataset= valid_dataset,
        batch_size= batch_size,
        shuffle= False,
        num_workers= 2
    )

    """Building the model"""

    if torch.cuda.is_available(): # Check if GPU is available
        device = torch.device('cuda')
        print('GPU is available. Using GPU for training.')

    else:
        device = torch.device('cpu')
        print('GPU is not available. Using CPU for training.')

    model = build_unet()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()

    """ Training the model """
    
    best_valid_loss = float("inf") #In order to compare the loss with the best valid loss found so far.

    for epoch in range(num_epochs):
        start_time = time.time()

        #First it will call the train function to train the dataset on
        train_loss = train(model, train_loader, optimizer, loss_fn, device)

        #Finding the validation loss by calling the evaluate function
        valid_loss = evaluate(model, validation_loader, loss_fn, device)

        """ Saving the model """
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print(data_str)





















