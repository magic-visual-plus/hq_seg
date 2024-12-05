from . import datasets
from .models import image_segmenter
import os
import torch.utils.data
import torch
import loguru
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

logger = loguru.logger

def iou_score(pred, target, class_index):
    intersection = ((pred == class_index) & (target == class_index)).sum()
    union = ((pred == class_index) | (target == class_index)).sum()
    if union == 0:
        return 1.0
    return intersection / union
    pass

def run_train(
        data_path,
        num_epoches,
        device='cuda:0'):
    path_output = 'output'
    path_train = os.path.join(data_path, 'train')
    path_val = os.path.join(data_path, 'val')
    dataset_train = datasets.ImageSegmentationDataset(path_train, random_transform=True)
    dataset_val = datasets.ImageSegmentationDataset(path_val)

    print(f'Number of classes: {dataset_train.num_classes}')
    model = image_segmenter.ImageSegmenter(dataset_train.num_classes)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=2, shuffle=True, num_workers=0)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=2, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    model.to(device)
    model.train()
    for i_epoch in range(num_epoches):
        # Training process
        train_losses = []
        train_ious = []
        for i_batch, batch_data in enumerate(dataloader_train):
            img, mask = batch_data
            img = img.to(device)
            mask = mask.to(device)

            # Forward pass
            pixel_scores = model(img)
            # Compute loss
            loss = model.compute_loss(pixel_scores, mask)
            # calculate averge iou
            pred_mask = torch.argmax(pixel_scores, dim=1)
            iou = iou_score(pred_mask, mask, 2)
            train_ious.append(iou.item())

            train_losses.append(loss.item())
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pass

        # Validation process
        model.eval()
        val_losses = []
        val_ious = []
        for i_batch, batch_data in enumerate(dataloader_val):
            img, mask = batch_data
            img = img.to(device)
            mask = mask.to(device)

            # Forward pass
            pixel_scores = model(img)
            # Compute loss
            loss = model.compute_loss(pixel_scores, mask)
            pred_mask = torch.argmax(pixel_scores, dim=1)
            iou = iou_score(pred_mask, mask, 2)
            val_ious.append(iou.item())
            val_losses.append(loss.item())
        
            pass

        logger.info(
            f'Epoch {i_epoch}, train loss: {np.mean(train_losses)}, validation loss: {np.mean(val_losses)}, lr: {scheduler.get_last_lr()[0]}'
            f'train iou: {np.mean(train_ious)}, validation iou: {np.mean(val_ious)}'
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


def run_train_decoder(
        data_path,
        num_epoches,
        device='cuda:0'):
    path_output = 'output'
    path_train = os.path.join(data_path, 'train')
    path_val = os.path.join(data_path, 'val')
    dataset_train = datasets.DecoderDataset(path_train)
    dataset_val = datasets.DecoderDataset(path_val)

    model = image_segmenter.ImageSegmenterDecoderTransformer(
        3, 128, 32
    )
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=0)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=64, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    model.to(device)
    model.train()
    for i_epoch in range(num_epoches):
        # Training process
        train_losses = []
        train_ious = []
        bar = tqdm(dataloader_train)
        for i_batch, batch_data in enumerate(bar):
            patch, patch_mask, x = batch_data
            patch = patch.to(device)
            patch_mask = patch_mask.to(device)
            x = x.to(device)

            patch_score = model(x, patch)
            # Compute loss
            loss = F.cross_entropy(patch_score, patch_mask)
            # calculate averge iou
            pred_mask = torch.argmax(patch_score, dim=1)
            iou = iou_score(pred_mask, patch_mask, 2)
            train_ious.append(iou.item())
            train_losses.append(loss.item())
            bar.set_postfix({'loss':loss.item()})
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pass

        # Validation process
        model.eval()
        val_losses = []
        val_ious = []
        for i_batch, batch_data in enumerate(tqdm(dataloader_val)):
            patch, patch_mask, x = batch_data
            patch = patch.to(device)
            patch_mask = patch_mask.to(device)
            x = x.to(device)

            patch_score = model(x, patch)
            # Compute loss
            loss = F.cross_entropy(patch_score, patch_mask)
            pred_mask = torch.argmax(patch_score, dim=1)
            iou = iou_score(pred_mask, patch_mask, 2)
            val_ious.append(iou.item())
            val_losses.append(loss.item())
        
            pass

        logger.info(
            f'Epoch {i_epoch}, train loss: {np.mean(train_losses)}, validation loss: {np.mean(val_losses)}, lr: {scheduler.get_last_lr()[0]}'
            f'train iou: {np.mean(train_ious)}, validation iou: {np.mean(val_ious)}'
        )

        scheduler.step()
        pass

    # Save model
    os.makedirs(path_output, exist_ok=True)
    path_model = os.path.join(path_output, 'model_decoder.pth')
    torch.save(
        {
            'num_classes': dataset_train.num_classes,
            'state_dict': model.state_dict()
        }, 
        path_model)
    pass