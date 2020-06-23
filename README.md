# COVID19Net
COVID19 CT scan visual recognition project

This repository consists of three .py files:

Model_Training.py shows the code developed by this repository's author for his paper titled "An Ensemble Approach for Multi-Stage Transfer Learning Models for COVID-19 Detection from Chest CT Scans". Preprint soon available. Model training is shown from the data preprocessing to the final ensemble model evaluation.

Results_Evaluation.py shows the evaluation metrics for the baseline ensemble method and the final ensemble model. This script uses the validation and test set scores and true labels obtained as CSVs from Model_Training.py script.

Grad-CAM.py constructs the influence representation image for a series of randomly picked COVID19 CT Scan images under the proposed model.

This work relies on the COVID19 dataset collected by:

@article{zhao2020COVID-CT-Dataset,
  title={COVID-CT-Dataset: a CT scan dataset about COVID-19},
  author={Zhao, Jinyu and Zhang, Yichen and He, Xuehai and Xie, Pengtao},
  journal={arXiv preprint arXiv:2003.13865}, 
  year={2020}
}
