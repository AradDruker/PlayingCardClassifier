import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from classes import BasicCNN, BasicCNN_v2, BasicCNN_v3, LeNet, PlayingCardDataset 
from utils import config

from tqdm import tqdm

transform = transforms.Compose([
transforms.Resize((128, 128)),
transforms.ToTensor(),
])

filename = config['filename']
train_dataset = PlayingCardDataset(config['train_folder'], transform=transform)
val_dataset = PlayingCardDataset(config['valid_folder'], transform=transform)

def train_model(batch_size, learning_rate, model_class):
    model = call_function_by_name(model_class)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 8  # Patience for early stopping

    for images, labels in train_loader:
        break

    torch.manual_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    #print(next(model.parameters()).device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4)

    train_losses, val_losses = [], []

    num_epochs = config['num_epochs']
    for epoch in range(num_epochs):

        #Training phase
        model.train()
        running_loss = 0.0
        total_samples = 0
        for images, labels in tqdm(train_loader, desc='Training loop'):
            # Move inputs and labels to the device
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Eval phase
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation loop'):
                # Move inputs and labels to the device
                images, labels = images.to(device), labels.to(device)
            
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)  # Keep track of total samples processed
        val_loss = running_loss / total_samples
        
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        #patience for epochs
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset patience if validation loss improves
        else:
            patience_counter += 1
        if patience_counter >= patience:
            #print("Early stopping triggered")
            return train_losses, val_losses, epoch, scheduler.get_last_lr()[-1], model, device

        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}, Learning rate: {scheduler.get_last_lr()[0]}, Model class: {model_class}")

    return train_losses, val_losses, epoch, scheduler.get_last_lr()[-1], model, device

#need to make the train_model one function
def train_model_optimal(batch_size, learning_rate, epochs, model_class):
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    model = call_function_by_name(model_class)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 8  # Patience for early stopping

    for images, labels in train_loader:
        break

    torch.manual_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    #print(next(model.parameters()).device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4)

    train_losses, val_losses = [], []

    num_epochs = epochs
    for epoch in range(num_epochs):

        #Training phase
        model.train()
        running_loss = 0.0
        total_samples = 0
        for images, labels in tqdm(train_loader, desc='Training loop'):
            # Move inputs and labels to the device
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Eval phase
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation loop'):
                # Move inputs and labels to the device
                images, labels = images.to(device), labels.to(device)
            
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)  # Keep track of total samples processed
        val_loss = running_loss / total_samples
        
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        #patience for epochs
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset patience if validation loss improves
        else:
            patience_counter += 1
        if patience_counter >= patience:
            #print("Early stopping triggered")
            return train_losses, val_losses, epoch, scheduler.get_last_lr()[-1], model, device

        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}, Learning rate: {scheduler.get_last_lr()[0]}")

    return train_losses, val_losses, epoch, scheduler.get_last_lr()[-1], model, device

def call_function_by_name(func_name):
    # Check if the function name exists in the global scope
    if func_name in globals():
        # Get the object associated with the name
        func = globals()[func_name]
        # Check if it's a callable object (i.e., a function)
        if callable(func):
            # Call the function and return its result
            return func()
        else:
            return "The name exists but is not a function."
    else:
        return "No function by that name exists."