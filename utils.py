import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from datetime import datetime


def train_model(model, train_dataset, test_dataset, args, fine_tune=False, num_workers=24):
    # Create logs directory if it doesn't exist
    log_dir = os.path.join('logs', 'train' if not fine_tune else 'fine_tune', 'experiment_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    print(f"TensorBoard logs will be saved to: {os.path.abspath(log_dir)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if fine_tune:
        train_loader = DataLoader(train_dataset, batch_size=args.fine_tune_batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.fine_tune_batch_size, shuffle=False, num_workers=num_workers)
        
        for param in model.model.parameters():
            param.requires_grad = False
        
        for param in model.channel_wise_multiplier.parameters():
            param.requires_grad = True
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.fine_tune_lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.fine_tune_epoch)
        criterion = nn.CrossEntropyLoss()

        num_epochs = args.fine_tune_epoch

        model.train()
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=num_workers)
        
        for param in model.model.parameters():
            param.requires_grad = True

        for param in model.channel_wise_multiplier.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(model.parameters(), lr=args.train_lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch)
        criterion = nn.CrossEntropyLoss()

        num_epochs = args.train_epoch

        model.train()
    
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training phase
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
            
            # Log batch loss
            writer.add_scalar('train/loss_batch', loss.item(), epoch * len(train_loader) + batch_idx)
        
        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Log training metrics
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/accuracy', train_acc, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate validation metrics
        val_loss /= len(test_loader)
        val_acc = 100 * correct / total

        # Log validation metrics
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/accuracy', val_acc, epoch)
        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Step the learning rate scheduler
        lr_scheduler.step()
        
        # Add model weights and gradients to TensorBoard
        for name, param in model.named_parameters():
            writer.add_histogram(f'weights/{name}', param, epoch)
            if param.grad is not None:
                writer.add_histogram(f'grads/{name}', param.grad, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model_'+str(fine_tune)+'.pth'))
    
    # Close the SummaryWriter
    writer.close()
    return model