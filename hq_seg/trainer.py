from . import datasets
from .models import image_segmenter
import os
import torch.utils.data
import torch
import loguru

logger = loguru.logger

def run_train(
        data_path,
        num_epoches,
        device='cuda:0'):
    path_train = os.path.join(data_path, 'train')
    path_val = os.path.join(data_path, 'val')
    dataset_train = datasets.ImageSegmentationDataset(path_train)
    dataset_val = datasets.ImageSegmentationDataset(path_val)

    print(f'Number of classes: {dataset_train.num_classes}')
    model = image_segmenter.ImageSegmenter(dataset_train.num_classes)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=0)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=4, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1.0)

    model.to(device)
    model.train()
    for i_epoch in range(num_epoches):
        # Training process
        for i_batch, batch_data in enumerate(dataloader_train):
            img, mask = batch_data
            img = img.to(device)
            mask = mask.to(device)

            # Forward pass
            pixel_scores = model(img)
            # Compute loss
            loss = model.compute_loss(pixel_scores, mask)

            logger.info(f'Epoch {i_epoch}, batch {i_batch}, loss: {loss.item()}')
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pass


        # Validation process
        model.eval()
        losses = []
        for i_batch, batch_data in enumerate(dataloader_val):
            img, mask = batch_data

            # Forward pass
            pixel_scores = model(img)
            # Compute loss
            loss = model.compute_loss(pixel_scores, mask)

            losses.append(loss.item())
            pass

        logger.info(f'Epoch {i_epoch}, validation loss: {sum(losses)/len(losses)}')

        scheduler.step()
        pass
    pass
