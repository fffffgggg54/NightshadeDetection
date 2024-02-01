import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import timm
from timm.models import *
import time

from timm.loss import AsymmetricLossMultiLabel

lr = 3e-3
lr_warmup_epochs = 5
num_epochs = 100
batch_size = 32
grad_acc_epochs = 1
weight_decay = 1e-4


device = 'cuda:0'

def getSingleMetric(preds, targs, metric):
    epsilon = 1e-12

    #preds = torch.sigmoid(preds)

    targs_inv = 1 - targs
    batchSize = targs.size(dim=0)
    P = targs * preds
    N = targs_inv * preds
    
    # [K]
    TP = P.sum(dim=0) / batchSize
    FN = (targs - P).sum(dim=0) / batchSize
    FP = N.sum(dim=0) / batchSize
    TN = (targs_inv - N).sum(dim=0) / batchSize
    
    return metric(TP, FN, FP, TN, epsilon)
    
# recall
def Precall(TP, FN, FP, TN, epsilon):
    #zero_grad(FP)
    #zero_grad(TN)
    return TP / (TP + FN + epsilon)
    
# specificity
def Nrecall(TP, FN, FP, TN, epsilon):
    #zero_grad(FN)
    return TN / (TN + FP + epsilon)

# precision
def Pprecision(TP, FN, FP, TN, epsilon):
    return TP / (TP + FP + epsilon)

# negative predictive value (NPV)
def Nprecision(TP, FN, FP, TN, epsilon):
    return TN / (TN + FN + epsilon)

# P4 metric
def P4(TP, FN, FP, TN, epsilon):
    return (4 * TP * TN) / ((4 * TN * TP) + (TN + TP) * (FP + FN) + epsilon)
    
# F1 metric
def F1(TP, FN, FP, TN, epsilon):
    return (2 * TP) / (2 * TP + FP + FN + epsilon)


# TODO test as boundary opt metric
# https://www.cs.uic.edu/~liub/publications/icml-03.pdf
# metric proposed in 
# Lee, W. S., & Liu, B. (2003).
# Learning with positive and unlabeled examples using weighted logistic regression.
# In Proceedings of the twentieth international conference on machine learning (pp. 448–455).
def PU_F_Metric(TP, FN, FP, TN, epsilon):
    return (Precall(TP, FN, FP, TN, epsilon) ** 2) / (FP + TP + epsilon)


# tracking for performance metrics that can be computed from confusion matrix
class MetricTracker():
    def __init__(self):
        self.running_confusion_matrix = None
        self.epsilon = 1e-12
        self.sampleCount = 0
        self.metrics = [Precall, Nrecall, Pprecision, Nprecision, P4, F1, PU_F_Metric]
        
    def get_full_metrics(self):
        with torch.no_grad():
            TP, FN, FP, TN = self.running_confusion_matrix / self.sampleCount
            
            #Precall = TP / (TP + FN + self.epsilon)
            #Nrecall = TN / (TN + FP + self.epsilon)
            #Pprecision = TP / (TP + FP + self.epsilon)
            #Nprecision = TN / (TN + FN + self.epsilon)
            
            #P4 = (4 * TP * TN) / ((4 * TN * TP) + (TN + TP) * (FP + FN) + self.epsilon)
            
            metrics = [metric(TP, FN, FP, TN, self.epsilon) for metric in self.metrics]
        
            return torch.column_stack([TP, FN, FP, TN, *metrics])
        
    def get_aggregate_metrics(self):
        '''
        with torch.no_grad():
        
            TP, FN, FP, TN = (self.running_confusion_matrix / self.sampleCount).mean(dim=1)
            
            Precall = TP / (TP + FN + self.epsilon)
            Nrecall = TN / (TN + FP + self.epsilon)
            Pprecision = TP / (TP + FP + self.epsilon)
            Nprecision = TN / (TN + FN + self.epsilon)
            
            P4 = (4 * TP * TN) / ((4 * TN * TP) + (TN + TP) * (FP + FN) + self.epsilon)
            return torch.stack([TP, FN, FP, TN, Precall, Nrecall, Pprecision, Nprecision, P4])
        '''
        return self.get_full_metrics().mean(dim=0)
    
    def update(self, preds, targs):
        self.sampleCount += targs.size(dim=0)
        
        targs_inv = 1 - targs
        P = targs * preds
        N = targs_inv * preds
        
        
        TP = P.sum(dim=0)
        FN = (targs - P).sum(dim=0)
        FP = N.sum(dim=0)
        TN = (targs_inv - N).sum(dim=0)
        
        output = torch.stack([TP, FN, FP, TN])
        if self.running_confusion_matrix is None:
            self.running_confusion_matrix = output
        
        else:
            self.running_confusion_matrix += output
            
        return self.get_aggregate_metrics()
        

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
    criterion = AsymmetricLossMultiLabel(gamma_neg=0, gamma_pos=0, clip=0.0, eps=1e-8)
    
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
            cm_tracker = MetricTracker()

            if phase == 'train':
                model.train()  # Set model to training mode
                print("training set")
            else:
                model.eval()   # Set model to evaluate mode
                print("validation set")
            for i,(image,labels) in enumerate(dataloaders[phase]):
                image = image.to(device, non_blocking=True)
                labels = labels.float().to(device)
                
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(image)
                    
                    preds = torch.sigmoid(outputs)

                    accuracy = 0
                    labels = labels.unsqueeze(-1)
                    loss = criterion(outputs, labels)
                    multiAccuracy = cm_tracker.update(preds.to(device), labels.to(device))
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
                        print(f"[{epoch}/{num_epochs}][{i}/{len(dataloaders[phase])}]\tLoss: {loss:.4f}\tImages/Second: {imagesPerSecond:.4f}\tTop-1: {multiAccuracy}")
                        torch.set_printoptions(profile='default')


        print(f'finished epoch {epoch} in {time.time()-epochTime}')
        epochTime = time.time()