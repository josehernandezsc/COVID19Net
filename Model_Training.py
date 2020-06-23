#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


# Import libraries

import os
from tqdm import tqdm
from PIL import Image
import copy
import random

import pandas as pd
import numpy as np
import itertools
from skimage import io, transform
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.optim import lr_scheduler

from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, roc_curve, auc, confusion_matrix

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(123456)


# In[ ]:


# Define Dataloader

batch=32

dtype = torch.cuda.FloatTensor

def get_datasplit(path):
    with open(path) as f:
        data=f.readlines()
        
        files=[line.strip() for line in data]
        
    return files

class CTScanDataset(Dataset):
    
    def __init__(self,root_dir,txt_cov,txt_nonCov,transform=None):
        """
        Args
        -----
        
        root_dir:   directory with CT scan images
        
        txt_cov:    txt file with COVID labeled images
        
        txt_nonCov: txt file with Non_COVID labeled images
        
        Folders structure:
        
        - root_dir
          - COVID
             - img1.png
             - img2.png
             - ...
          - Non_COVID
             - img1.png
             - img2.pmg
             - ...
        
        """
        self.root_dir=root_dir
        self.txt=[txt_cov,txt_nonCov]
        
        self.transform=transform
        self.classes=["CT_COVID","CT_NonCOVID"]
        self.total_list=[]
        
        for i in range(len(self.classes)):
            class_list=[[os.path.join(self.root_dir,self.classes[i],c),i] for c in get_datasplit(self.txt[i])]
            self.total_list.extend(class_list)
    
    def __len__(self):
        return len(self.total_list)
        
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx=idx.tolist()
        #print(self.total_list[1,0])
        img_name=self.total_list[idx][0]
        
        image = Image.open(img_name).convert('RGB')
        
        target=int(self.total_list[idx][1])
        #target=np.asarray(target)
        #target=target.astype('float').reshape(-1,1)
        
        
        if self.transform:
            image=self.transform(image)
        sample={'image':image.type(dtype),'target':target}
        return sample
        


# In[ ]:


# Define transformations

transform={'train': transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])]),
          'val':transforms.Compose([
              transforms.Resize([224,224]),
              transforms.ToTensor(),
              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
          ])}


# In[ ]:


# Set paths from which dataloader will feed

root='/Images'
tr_cov='/Data_split/trainCT_COVID.txt'
tr_ncov='/Data_split/trainCT_NonCOVID.txt'
val_cov='/Data_split/valCT_COVID.txt'
val_ncov='/Data_split/valCT_NonCOVID.txt'
test_cov='/Data_split/testCT_COVID.txt'
test_ncov='/Data_split/testCT_NonCOVID.txt'

trainSet=CTScanDataset(root_dir=root,txt_cov=tr_cov,txt_nonCov=tr_ncov,transform=transform['train'])
valSet=CTScanDataset(root_dir=root,txt_cov=val_cov,txt_nonCov=val_ncov,transform=transform['val'])
testSet=CTScanDataset(root_dir=root,txt_cov=test_cov,txt_nonCov=test_ncov,transform=transform['val'])

trainLoad=DataLoader(trainSet,batch_size=batch,shuffle=True,num_workers=0)
valLoad=DataLoader(valSet,batch_size=batch,shuffle=False,num_workers=0)
testLoad=DataLoader(testSet,batch_size=batch,shuffle=False,num_workers=0)


# In[ ]:


# Define train, validate and test functions to be used on each model

def train(model,criterion,optimizer,scheduler):
    """
    Function trains model for a single epoch, executing 
    forward pass and backpropagation parameters optimization
    
    Args
    -----
    model:     defined model to be trained
    
    criterion: loss functions to be optimized
    
    optimizer: optimizer used for training
    
    scheduler: learning rate scheduler to be used for trainig
    
    
    Returns
    -------
    running_loss: total training loss for current epoch
    
    accuracy:     training accuracy for current epoch
    
    """
    
    running_loss=0
    acc=0
    
    model.train()
    
    
    for data in tqdm(trainLoad):
        inputs, labels = data['image'], data['target']

        optimizer.zero_grad()
        
        outputs = model(inputs)
        outputs=torch.reshape(outputs,(labels.shape[0],))
        
        labels = labels.type_as(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()

        running_loss += loss.item()
        score=torch.sigmoid(outputs)
        pred=torch.round(score)
        acc += pred.eq(labels.long().view_as(pred)).sum().item()
    
    accuracy=acc/len(trainLoad.dataset)
        
    print("\n Train set. Loss: {:.2f}\n Accuracy: {:.2f}".format(running_loss,accuracy))
    
    return running_loss,accuracy

    
def val(model,criterion):
    """
    Function performs a validation run on a single epoch. 
    This comes after each training epoch is completed to
    estimate validation accuracy and loss to determine
    model propensity to overfitting.
    
    Args
    -----
    model:     defined model to be trained
    
    criterion: loss functions to be optimized
    
    
    Returns
    -------
    running_loss: total training loss for current epoch
    
    accuracy:     training accuracy for current epoch
    
    scores:       validation set scores in a list
    
    v_labels:     validation set true labels in a list
    
    
    """
    running_loss=0
    acc=0
    scores=[]
    model.eval()
    v_labels=[]
    
    
    with torch.no_grad():
        for data in tqdm(valLoad):
            inputs, labels = data['image'],data['target']
            
            outputs=model(inputs)
            outputs=torch.reshape(outputs,(labels.shape[0],))
            
            labels = labels.type_as(outputs)
            loss=criterion(outputs,labels)
            
            running_loss+=loss.item()
            score=torch.sigmoid(outputs)
            pred=torch.round(score)
            scores.extend(score)
            
            acc+=pred.eq(labels.long().view_as(pred)).sum().item()
            v_labels.extend(labels)
            
            
        accuracy=acc/len(valLoad.dataset)
        print("\n Validation set. Loss: {:.2f}\n Accuracy: {:.2f}".format(running_loss,accuracy))
        
    return running_loss,accuracy,scores,v_labels


def test(model,criterion):
    """
    Function evaluates trained model on testing set.
    
    Args
    -----
    model:     defined model to be trained
    
    criterion: loss functions to be optimized
    
    
    Returns
    -------
    preds:    model predictive label in a list
    
    f_labels: test set true labels in a list
    
    scores:   test set model output in a list
    
    """
    
    preds=[]
    f_labels=[]
    scores=[]
    running_loss=0
    acc=0
    model.eval()
    
    
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    
    with torch.no_grad():
        for data in tqdm(testLoad):
            inputs, labels = data['image'],data['target']
            
            outputs=model(inputs)
            
            outputs=torch.reshape(outputs,(labels.shape[0],))
            
            labels = labels.type_as(outputs)
            loss=criterion(outputs,labels)
            
            running_loss+=loss.item()
            
            score=torch.sigmoid(outputs)
            
            pred=torch.round(score)
            acc+=pred.eq(labels.long().view_as(pred)).sum().item()
            
            TP += ((pred == 1) & (labels.long()[:].view_as(pred).data == 1)).cpu().sum()
            TN += ((pred == 0) & (labels.long()[:].view_as(pred) == 0)).cpu().sum()
            FN += ((pred == 0) & (labels.long()[:].view_as(pred) == 1)).cpu().sum()
            FP += ((pred == 1) & (labels.long()[:].view_as(pred) == 0)).cpu().sum()
            preds.extend(pred)
            f_labels.extend(labels)
            scores.extend(score)
            
        precision=int(TP)/(int(TP)+int(FP))
        recall=int(TP)/(int(TP)+int(FN))
        spec=int(TN)/(int(TN)+int(FP))
        
        scores_auc=[e.item() for e in scores]
        labels_auc=[e.item() for e in f_labels]
        
        auc=roc_auc_score(labels_auc,scores_auc)
        accuracy=acc/len(testLoad.dataset)
        print("\n Test set loss: {:.4f}\n Test set accuracy: {:.4f}".format(running_loss,accuracy))
        print("\n Precision: {:.4f} \n Recall/Sensitivity: {:.4f}".format(precision,recall))
        print(" Specificity: {:.4f}".format(spec))
        print("\n F1 Score: {:.4f}".format(2*precision*recall/(precision+recall)))
        print("\n AUC: {:.4f}".format(auc))
        
    return preds,f_labels,scores
    


# In[ ]:


criterion = nn.BCEWithLogitsLoss()

def train_model(model,epochs_1,epochs_2,criterion):
    """
    Function performs 2-stage transfer learning training 
    on pretrained models. Please refer to research paper
    for method description.
    
    Args
    -----
    
    model:     defined model to be trained
    
    epochs_1:  number of epochs for stage 1
    
    epochs_2:  number of epochs for stage 2
    
    criterion: loss function to be optimized
    
    
    Returns
    -------
    
    best_model:     best model parameters dictionary found 
                    based on best validation accuracy
    
    best_val_score: best validation score found in stage 2
    
    v_labels:       validation set true labels in a list
    
    """
    
    # 1st training stage starts
    
    print("Starting 1st training stage on fully connected layers...")
    model.cuda()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    train_loss_l=[]
    train_acc_l=[]
    val_loss_l=[]
    val_acc_l=[]
    best_acc=0
    
    lr=1e-3
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,patience=10,verbose=True)
    
    for epoch in range(epochs_1):
        print("\n Epoch {}".format(epoch+1))

        train_loss,train_acc=train(model=model,criterion=criterion,optimizer=optimizer,scheduler=scheduler)
        
        val_loss,val_acc,scores,v_labels=val(model,criterion)
        if val_acc>=best_acc:
            best_acc=val_acc
            best_model = copy.deepcopy(model.state_dict())
            best_val_score=scores
        best_val_score=scores
        scheduler.step(val_acc)
        train_loss_l.append(train_loss)
        train_acc_l.append(train_acc)
        val_loss_l.append(val_loss)
        val_acc_l.append(val_acc)
        
    if epochs_2>0:
        for param in model.parameters():
            param.requires_grad = True
    
    # 2nd training stage starts
    
    print("\nStarting 2nd training stage on full net with reduced LR...")
    best_acc=0
    lr=1e-4
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,patience=10,verbose=True)
    
    for epoch in range(epochs_2):
        print("\n Epoch {}".format(epoch+1))

        train_loss,train_acc=train(model,criterion,optimizer,scheduler)
        val_loss,val_acc,scores,v_labels=val(model,criterion)
        if val_acc>=best_acc:
            best_acc=val_acc
            best_model = copy.deepcopy(model.state_dict())
            best_val_score=scores
        scheduler.step(val_acc)
        train_loss_l.append(train_loss)
        train_acc_l.append(train_acc)
        val_loss_l.append(val_loss)
        val_acc_l.append(val_acc)
    
    
    epochs_total=epochs_1+epochs_2
    
    fig=plt.figure(figsize=(14,6))
    
    x=np.linspace(start=1,stop=epochs_total,num=epochs_total)
    
    plt.subplot(121)
    
    plt.plot(x,train_loss_l,label="Training Set",color='blue')
    plt.plot(x,val_loss_l,label="Validation Set",color='red')
    plt.legend()
    plt.title("Loss Plot")
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    
    plt.subplot(122)
    
    plt.plot(x,train_acc_l,label="Training Set",color='blue')
    plt.plot(x,val_acc_l,label="Validation Set",color='red')
    plt.legend()
    plt.title("Accuracy Plot")
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    
    plt.show()
    
    return (best_model,best_val_score,v_labels)


# ## Models Training

# ## VGG16 with BN

# In[ ]:


# VGG16 model definition

model1 = models.vgg16_bn(pretrained=True)

for param in model1.features.parameters():
    param.requires_grad = False
    
model1.classifier[6] = nn.Sequential(
                      nn.Linear(4096, 128),  
                      nn.ReLU(), 
                      nn.Dropout(0.5),
                      nn.Linear(128, 1))


# In[ ]:


# VGG16 model training

best_model,val_score1,v_labels=train_model(model=model1,epochs_1=30,epochs_2=20,criterion=criterion)
model1.load_state_dict(best_model)


# In[ ]:


# VGG16 model evaluation

preds1,f_labels,scores1 = test(model1,criterion)


# ## ResNet50

# In[ ]:


# ResNet50 model definition

model2=models.resnet50(pretrained=True)

for param in model2.parameters():
    param.requires_grad = False   
    
model2.fc = nn.Sequential(
                      nn.Linear(2048, 256),  
                      nn.BatchNorm1d(256), 
                      nn.ReLU(), 
                      nn.Dropout(0.5),
                      nn.Linear(256, 64),  
                      nn.ReLU(), 
                      nn.Dropout(0.4),
                      nn.Linear(64, 1))


# In[ ]:


# ResNet50 model training

best_model,val_score2,_=train_model(model=model2,epochs_1=30,epochs_2=20,criterion=criterion)
model2.load_state_dict(best_model)


# In[ ]:


# ResNet50 model evaluation  

preds2,f_labels,scores2 = test(model2,criterion)


# ## DenseNet169

# In[ ]:


# DensetNet169 model definition

model3=models.densenet169(pretrained=True)

for param in model3.parameters():
    param.requires_grad = False   
input_d=model3.classifier.in_features

model3.classifier = nn.Sequential(
                      nn.Linear(input_d, 256),  
                      nn.BatchNorm1d(256), 
                      nn.ReLU(), 
                      nn.Dropout(0.5),
                      nn.Linear(256, 64),  
                      nn.ReLU(), 
                      nn.Dropout(0.4),
                      nn.Linear(64, 1))


# In[ ]:


# DensetNet169 model training

best_model,val_score3,_=train_model(model=model3,epochs_1=30,epochs_2=20,criterion=criterion)
model3.load_state_dict(best_model)


# In[ ]:


# DensetNet169 model evaluation

preds3,f_labels,scores3 = test(model3,criterion)


# ## Inception v3

# In[ ]:


# Inception v3 model definition

model4 = models.inception_v3(pretrained=True)
model4.aux_logits=False

for param in model4.parameters():
    param.requires_grad = False  

num_ftrs = model4.fc.in_features

model4.fc = nn.Sequential(
                      nn.Linear(num_ftrs, 256),  
                      nn.BatchNorm1d(256), 
                      nn.ReLU(), 
                      nn.Dropout(0.5),
                      nn.Linear(256, 64),  
                      nn.ReLU(), 
                      nn.Dropout(0.4),
                      nn.Linear(64, 1))


# In[ ]:


# Inception v3 model training

best_model,val_score4,_=train_model(model=model4,epochs_1=30,epochs_2=20,criterion=criterion)
model4.load_state_dict(best_model)


# In[ ]:


# Inception v3 model evaluation

preds4,_,scores4 = test(model4,criterion)


# ## DenseNet161

# In[ ]:


# DensetNet161 model definition

model5=models.densenet161(pretrained=True)

for param in model5.parameters():
    param.requires_grad = False   
input_d=model5.classifier.in_features

model5.classifier = nn.Sequential(
                      nn.Linear(input_d, 512),  
                      nn.BatchNorm1d(512), 
                      nn.ReLU(), 
                      nn.Dropout(0.5),
                      nn.Linear(512, 64),  
                      nn.ReLU(), 
                      nn.Dropout(0.4),
                      nn.Linear(64, 1))


# In[ ]:


# DenseNet161 model training

best_model,val_score5,_=train_model(model=model5,epochs_1=30,epochs_2=20,criterion=criterion)
model5.load_state_dict(best_model)


# In[ ]:


# DenseNet161 model evaluation

preds5,_,scores5 = test(model5,criterion)


# ## Wide ResNet50-2

# In[ ]:


# Wide ResNet50-2 model definition

model6=models.wide_resnet50_2(pretrained=True)

for param in model6.parameters():
    param.requires_grad = False   
input_d=model6.fc.in_features

model6.fc = nn.Sequential(
                      nn.Linear(input_d, 256),  
                      nn.BatchNorm1d(256), 
                      nn.ReLU(), 
                      nn.Dropout(0.5),
                      nn.Linear(256, 64),  
                      nn.ReLU(), 
                      nn.Dropout(0.4),
                      nn.Linear(64, 1))


# In[ ]:


# Wide ResNet50-2 model training

best_model,val_score6,_=train_model(model=model6,epochs_1=30,epochs_2=20,criterion=criterion)
model6.load_state_dict(best_model)


# In[ ]:


# Wide ResNet50-2 model evaluation

preds6,_,scores6 = test(model6,criterion)


# ## Ensemble preparation

# In[ ]:


# Test scores from the 6 pretrained models saving to evaluate voting ensemble model

scores=[scores1,scores2,scores3,scores4,scores5,scores6]
scores_f=[]
for score in scores:
    scores_f.append([e.item() for e in score])
    
df_test_score=pd.DataFrame({'VGG16':scores_f[0],'ResNet50':scores_f[1],'DenseNet169':scores_f[2],
              'InceptionV3':scores_f[3],'DenseNet161':scores_f[4],'WResNet50-2':scores_f[5]})


# In[ ]:


# Validation scores from the 6 pretrained models saving to build voting ensemble model


scores=[val_score1,val_score2,val_score3,val_score4,val_score5,val_score6]
scores_v=[]
for score in scores:
    scores_v.append([e.item() for e in score])

df_val_score=pd.DataFrame({'VGG16':scores_v[0],'ResNet50':scores_v[1],'DenseNet169':scores_v[2],
              'InceptionV3':scores_v[3],'DenseNet161':scores_v[4],'WResNet50-2':scores_v[5]})


# In[ ]:


# Validation and test set true labels for models evaluation

v_labels=[e.item() for e in v_labels]
f_labels=[e.item() for e in f_labels]

vallabels=pd.DataFrame({'ValLabels':v_labels})
testlabels=pd.DataFrame({'TestLabels':f_labels})


# In[ ]:


# Save labels and scores as CSV

df_test_score.to_csv('TestScore.csv')
df_val_score.to_csv('ValScore.csv')
vallabels.to_csv('ValLabels.csv')
testlabels.to_csv('TestLabels.csv')


# ## Proposed Ensemble Model
# 
# MultCovNet Model

# In[ ]:


# Creating list of initial models

models=[model1,model2,model3,model4,model5,model6]
for model in models:
    for param in model.parameters():
        param.requires_grad = False   


# In[ ]:


# Defining custom ensemble model

class MultCovNet(nn.Module):
    def __init__(self,model1,model2,model3,model4,model5,model6):
        super(MultCovNet, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(6, 1))
                                        
        self.model1=model1
        self.model2=model2
        self.model3=model3
        self.model4=model4
        self.model5=model5
        self.model6=model6
        

    def forward(self, m1):
        x1 = self.model1(m1)
        x2 = self.model2(m1)
        x3 = self.model3(m1)
        x4 = self.model4(m1)
        x5 = self.model5(m1)
        x6 = self.model6(m1)
        x = torch.cat((x1, x2,x3,x4,x5,x6), 1)
        x=self.classifier(x)
        return x


# In[ ]:


# Creating ensemble model

MultCovNet_model=MultCovNet(model1,model2,model3,model4,model5,model6)


# In[ ]:


# Freezing all model except for final classifier neuron

for param in MultCovNet_model.model1.parameters():
    param.requireds_grad = False
    
for param in MultCovNet_model.model2.parameters():
    param.requireds_grad = False
    
for param in MultCovNet_model.model3.parameters():
    param.requireds_grad = False
    
for param in MultCovNet_model.model4.parameters():
    param.requireds_grad = False
    
for param in MultCovNet_model.model5.parameters():
    param.requireds_grad = False
    
for param in MultCovNet_model.model6.parameters():
    param.requireds_grad = False
    
for param in MultCovNet_model.classifier.parameters():
    param.requireds_grad = True


# In[ ]:


# Uncomment to load trained model as trained in the research paper

#PATH='/models/MultCovNet-dict'
#MultCovNet_model.load_state_dict(torch.load(PATH))


# In[ ]:


# Train model

best_model,val_score,_=train_model(model=MultCovNet_model,epochs_1=15,epochs_2=0,criterion=criterion)
MultCovNet_model.load_state_dict(best_model)


# In[ ]:


# Evaluate model

MultCovNet_model.cuda()
preds,f_labels ,scores= test(MultCovNet_model,criterion)


# In[ ]:


# Save scores for complete models evaluation and comparison

score=[e.item() for e in scores]
df_test=pd.DataFrame({'MultCovNet':score})
df_test.to_csv('TestScoresModel.csv')


# In[ ]:




