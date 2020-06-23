# COVID19Net
COVID19 CT Scan Visual Recognition Project.

This repository consists of three .py files:

**Model_Training.py** shows the code developed by this repository's author for his paper titled "An Ensemble Approach for Multi-Stage Transfer Learning Models for COVID-19 Detection from Chest CT Scans". Preprint soon available. Model training is shown from the data preprocessing to the final ensemble model evaluation.

**Results_Evaluation.py** shows the evaluation metrics for the baseline ensemble method and the final ensemble model. This script uses the validation and test set scores and true labels obtained as CSVs from Model_Training.py script.

**Grad-CAM.py** constructs the influence representation image for a series of randomly picked COVID19 CT Scan images under the proposed model.

This work relies on the COVID19 dataset collected by:

Zhao, Jinyu and Zhang, Yichen and He, Xuehai and Xie, Pengtao <br/>
COVID-CT-Dataset: a CT scan dataset about COVID-19 <br/>
arXiv preprint arXiv:2003.13865 <br/>
2020

## Requirements

1. PyTorch
1. Torchvision
1. tqdm
1. Pillow
1. Pandas
1. Numpy
1. Scikit-image
1. Scikit-learn
1. pytorch-gradcam

## Environment

Development is based on Python 3.6.6 and was run on a NVIDIA K80 GPU with 12GB of internal memory.

The final model parameters dictionary will be made available soon.

For instructions on dataset handling and loading please refer to https://github.com/UCSD-AI4H/COVID-CT
