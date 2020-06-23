#!/usr/bin/env python
# coding: utf-8

# In[47]:


#!pip install pytorch-gradcam


# In[1]:


#Load libraries

import os
from tqdm import tqdm
from PIL import Image

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from gradcam.utils import visualize_cam,find_resnet_layer
from gradcam import GradCAM, GradCAMpp

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


# In[30]:


# Load images previously picked at random

image_dir='/ImageDataSet'

covid_im1='2020.03.01.20029769-p21-73_1%2.png'
covid_im2='2020.02.27.20027557-p25-137%2.png'
covid_im3='2020.03.03.20030775-p11-91.png'
covid_im4='2020.03.04.20031047-p12-81%3.png'


# Put images in a list
covid_im_list=[covid_im1,covid_im2,covid_im3,covid_im4]

paths=[os.path.join(image_dir,path) for path in covid_im_list]  
covid_im_list=[e for e in paths]
img_list=[Image.open(img).convert('RGB') for img in covid_im_list]


# In[31]:


#Set transforms

torch_img=[transforms.Compose([transforms.Resize([224,224]),
                               transforms.ToTensor()])(raw_img).to(device) for raw_img in img_list]


# In[32]:


# Set normalized transform
norm_torch_img=[transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])(t_img)[None] for t_img in torch_img]


# In[53]:


#Define model class

class MultCovNet(nn.Module):
    def __init__(self,model,model2,model3,model4,model5,model6):
        super(MultCovNet, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(6, 1))
                                        
        self.model1=model
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


# In[54]:


#Load model

model_path='/models/MultCovNet'
MultCovNet_model = torch.load(model_path)


# In[55]:


for param in MultCovNet_model.parameters():
    param.requires_grad = True  
    


# In[56]:


vgg=MultCovNet_model.model1
resnet50=MultCovNet_model.model2
dense169=MultCovNet_model.model3
dense161=MultCovNet_model.model5
w_res50=MultCovNet_model.model6


configs = [
    
    dict(model_type='vgg', arch=vgg, layer_name='features_42'),
    dict(model_type='resnet', arch=resnet50, layer_name='layer4'),
    dict(model_type='resnet', arch=w_res50, layer_name='layer4'),
    dict(model_type='densenet', arch=dense161, layer_name='features_norm5'),
    dict(model_type='densenet', arch=dense169, layer_name='features_norm5')
    
]


# In[57]:


for config in configs:
    config['arch'].to(device).eval()

cams = [
    [cls.from_config(**config) for cls in (GradCAM, GradCAMpp)]
    for config in configs
]


# In[58]:


images = []
img_index=1

for gradcam,gradcam_pp in cams:

    mask_pp, _ = gradcam_pp(norm_torch_img[img_index])
    heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img[img_index])
    
    images.extend([torch_img[img_index].cpu(),  heatmap_pp,  result_pp])
    
grid_image = make_grid(images, nrow=3)


# In[59]:


transforms.ToPILImage()(grid_image)


# In[ ]:




