import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import timm
from timm.models import *
import time

lr = 3e-3
lr_warmup_epochs = 5
num_epochs = 100
batch_size = 32
grad_acc_epochs = 1
weight_decay = 1e-4


device = 'cuda:0'


def getDataLoader(dataset):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers = True,
        prefetch_factor=2, 
        pin_memory = True, 
        drop_last=True, 
        generator=torch.Generator().manual_seed(41)
    )
    
    
if __name__ == '__main__':

    ds = torchvision.datasets.ImageFolder(
        './data/nightshade_448/different_set',
        transform=transforms.Compose([
            transforms.Resize((448,448)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    )
    train_ds, test_ds = tuple(torch.utils.data.random_split(ds, [0.9, 0.1]))

    datasets = {'train':train_ds,'val':test_ds}
    dataloaders = {x: getDataLoader(datasets[x]) for x in datasets}
    
    model = timm.create_model('resnet18', num_classes=1)
    model=model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
        max_lr=lr, 
        steps_per_epoch=len(dataloaders['train']),
        epochs=num_epochs, 
        pct_start=lr_warmup_epochs/num_epochs
    )
    
    cycleTime = time.time()
    epochTime = time.time()
    stepsPerPrintout = 50
    for epoch in range( num_epochs):
        for phase in ['train', 'val']:
            samples = 0
            correct = 0
            if phase == 'train':
                model.train()  # Set model to training mode
                print("training set")
            else:
                model.eval()   # Set model to evaluate mode
                print("validation set")
            for i,(image,labels) in enumerate(dataloaders[phase]):
                image = image.to(device, non_blocking=True)
                labels = labels.to(device)
                
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(image)

                    preds = torch.sigmoid(outputs)

                    accuracy = 0
                    labelBatch = torch.eye(1, device=device)[labels]
                    loss = criterion(outputs, labelBatch)

                    if phase == 'train':
                        loss.backward()
                        if(i % grad_acc_epochs == 0):
                            '''
                            nn.utils.clip_grad_norm_(
                                model.parameters(), 
                                max_norm=1.0, 
                                norm_type=2
                            )
                            '''
                            optimizer.step()
                            optimizer.zero_grad()
                        scheduler.step()
                    
                    if i % stepsPerPrintout == 0:
                        
                        imagesPerSecond = (batch_size * stepsPerPrintout)/(time.time() - cycleTime)
                        cycleTime = time.time()
                        torch.set_printoptions(linewidth = 200, sci_mode = False)
                        print(f"[{epoch}/{num_epochs}][{i}/{len(dataloaders[phase])}]\tLoss: {loss:.4f}\tImages/Second: {imagesPerSecond:.4f}\tTop-1: {accuracy:.2f}")
                        torch.set_printoptions(profile='default')


        print(f'finished epoch {epoch} in {time.time()-epochTime}')
        epochTime = time.time()