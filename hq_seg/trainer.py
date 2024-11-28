from . import datasets
from .models import image_segmenter
import os
import torch.utils.data
import torch
import loguru
import numpy as np
import torch.nn.functional as F

logger = loguru.logger

def run_train(
        data_path,
        num_epoches,
        device='cuda:0'):
    path_output = 'output'
    path_train = os.path.join(data_path, 'train')
    path_val = os.path.join(data_path, 'val')
    dataset_train = datasets.ImageSegmentationDataset(path_train)
    dataset_val = datasets.ImageSegmentationDataset(path_val)

    print(f'Number of classes: {dataset_train.num_classes}')
    model = image_segmenter.ImageSegmenter(dataset_train.num_classes)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=0)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=4, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1.0)

    model.to(device)
    model.train()
    for i_epoch in range(num_epoches):
        # Training process
        train_losses = []
        train_accuracies = []
        for i_batch, batch_data in enumerate(dataloader_train):
            img, mask = batch_data
            img = img.to(device)
            mask = mask.to(device)

            # Forward pass
            pixel_scores = model(img)
            # Compute loss
            loss = model.compute_loss(pixel_scores, mask)
            # calculate accuracy
            pred_mask = torch.argmax(pixel_scores, dim=1)
            accuracy = (pred_mask == mask).float().mean()
            train_accuracies.append(accuracy.item())


            train_losses.append(loss.item())
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pass

        # Validation process
        model.eval()
        val_losses = []
        val_accuracies = []
        for i_batch, batch_data in enumerate(dataloader_val):
            img, mask = batch_data
            img = img.to(device)
            mask = mask.to(device)

            # Forward pass
            pixel_scores = model(img)
            # Compute loss
            loss = model.compute_loss(pixel_scores, mask)
            pred_mask = torch.argmax(pixel_scores, dim=1)
            accuracy = (pred_mask == mask).float().mean()
            val_accuracies.append(accuracy.item())
            val_losses.append(loss.item())
        
            pass

        logger.info(
            f'Epoch {i_epoch}, train loss: {np.mean(train_losses)}, validation loss: {np.mean(val_losses)}, lr: {scheduler.get_last_lr()[0]}'
            f'train accuracy: {np.mean(train_accuracies)}, validation accuracy: {np.mean(val_accuracies)}'
        )

        scheduler.step()
        pass

    # Save model
    os.makedirs(path_output, exist_ok=True)
    path_model = os.path.join(path_output, 'model.pth')
    torch.save(
        {
            'num_classes': dataset_train.num_classes,
            'state_dict': model.state_dict()
        }, 
        path_model)
    pass
