#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[268]:


# Import libraries

import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

import itertools

from sklearn.metrics import f1_score, roc_auc_score, precision_score,recall_score
from sklearn.metrics import roc_curve, confusion_matrix, auc


# In[269]:


# Load previously saved pandas dataframes with labels and scores from trained models

root='/Results'
scores_path=['TestScore.csv','ValScore.csv',
             'TestScoresModel.csv','TestLabels.csv','ValLabels.csv']


df2_test=pd.read_csv(os.path.join(root,scores_path[1]),index_col=0)
df2_val=pd.read_csv(os.path.join(root,scores_path[2]),index_col=0)
df7_test=pd.read_csv(os.path.join(root,scores_path[3]),index_col=0)
testlabels=pd.read_csv(os.path.join(root,scores_path[4]),index_col=0)
vallabels=pd.read_csv(os.path.join(root,scores_path[5]),index_col=0)


# In[270]:


###### INVERT LABELS #######
# Originally 0 stands for COVID and 1 for NON-COVID.
# To represent the relevant variable, run this cell to invert labels:
# 0 for NON-COVID and 1 for COVID

df2_test=1-df2_test
df2_val=1-df2_val
df7_test=1-df7_test
testlabels=1-testlabels
vallabels=1-vallabels


# In[271]:


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    # source: https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
    # small modifications were made to adjust for current use
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=0)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.ylim([1.5, -0.5])
    plt.show()


# In[272]:


def accuracy(pred,labels):
    return np.mean(pred==labels)


# In[273]:


def vote(scores,method,threshold):
    """
    Function performs ensemble given by scores matrix and
    method specified and returns score and prediction labels
    
    Args
    ------
    scores:    scores to be used for estimating ensemble result
               [n x d] matrix where n = number of samples and
               d = number of models to ensemble
               
    method:    method for building voting classifier: 
               'soft' or 'hard'. Refer to research paper
               for further detail.
               
    threshold: cutoff value for predicition labels boundary
    
    
    Returns
    -------
    scores: scores obtained by voting ensemble
    
    vote:   prediction labels: 0 or 1
    
    """
    
    if method=='hard':
        scores=scores>threshold
    elif method!='soft':
        raise ValueError('Please introduce a valid method.')
    
    scores=np.mean(scores,axis=1)
    
    
    vote=scores>=0.5
    vote=vote.astype(int)
    
    
    return scores,vote
    


# In[274]:


def roc_curve_plot(test,pred):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(test, pred)
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    x=np.linspace(0,1,100)
    plt.figure(figsize=(8,6))
    plt.plot(fpr[1], tpr[1],color='blue')
    plt.plot(x,x,color='red')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.show()


# In[275]:


def voting_classifier(scores,labels,method,scores_test,labels_test):
    """
    Function builds a baseline ensemble model by first sorting models
    from highest to lowest by their validation accuracy. Then starts 
    constructing an ensemble starting from 2 models to 6, in the order 
    that models were sorted. Metrics on validation set are plot and
    highest validation accuracy ensemble is chosen to finally evaluate
    on test set.
    
    Args
    ------
    scores:      validation set scores dataframe.
    
    labels:      validation set true labels dataframe.
    
    method:      method for building voting classifier: 
                 soft or hard. Refer to research paper
                 for further detail.
                 
    scores_test: test set scores dataframe
    
    labels_test: test set true labels dataframe
    
    
    Returns
    -------
    scores: ensemble model test set scores
    
    """
    model_names=scores.columns
    scores_v=scores.values
    scores_t=scores_test.values
    labels=labels.values
    labels_test=labels_test.values
    labels=np.reshape(labels,(labels.shape[0]))
    labels_test=np.reshape(labels_test,(labels_test.shape[0]))
    
    # Dictionary of models for sorting by validation accuracy
    acc={}
    
    for i,name in enumerate(model_names):
        score_aux=scores_v[:,i]>0.5
        score_aux=score_aux.astype(int)
        acc[name]=(accuracy(score_aux,labels))
    
    sorted_acc={k: acc[k] for k in sorted(acc, key=acc.get,reverse=True)}
    order=sorted_acc.keys()
    
    scores=scores[order]
    scores_v=scores.values
    
    accs=[]
    f1_list=[]
    auc_list=[]
    prec_list=[]
    rec_list=[]
    
    for i in range(len(order)-1):
        score,preds=vote(scores_v[:,:i+2],method,0.5)
        acc=accuracy(preds,labels)
        
        f1=f1_score(labels,preds)
        if method=="soft":
            auc=roc_auc_score(labels,score)
            auc_list.append(auc)
        prec=precision_score(labels,preds)
        rec=recall_score(labels,preds)
        
        accs.append(acc)
        f1_list.append(f1)
        
        prec_list.append(prec)
        rec_list.append(rec)
        
    fig=plt.figure(figsize=(8,6))
    x_axis = [i for i in range(2, len(order)+1)]
    
    plt.plot(x_axis, accs, marker='o',label='Accuracy')
    plt.plot(x_axis, f1_list, marker='o',label='F1 Score')
    if method=="soft":
        plt.plot(x_axis,auc_list, marker='o',label='AUC')
    plt.plot(x_axis,prec_list, marker='o',label='Precision')
    plt.plot(x_axis,rec_list, marker='o',label='Recall')
    plt.legend()
    plt.show()
    
    n_models=np.argmax(accs)+2
    
    print("Order of models: ",order)
    print('\nBest number of models to ensemble by accuracy: ',n_models)
    print('\nStarting ensemble and testing...')
    
    score,preds=vote(scores_t[:,:n_models],method,0.5)
    acc=accuracy(preds,labels_test)
    f1=f1_score(labels_test,preds)
    auc=roc_auc_score(labels_test,score)
    prec=precision_score(labels_test,preds)
    rec=recall_score(labels_test,preds)
    
    TN=sum((preds==0)&(labels_test==0))
    FP=sum((preds==1)&(labels_test==0))

    specificity=TN/(TN+FP)
    
    print("\n Accuracy: {:.4f} \n F1 Score: {:.4f} \n AUC: {:.4f} \n Precision: {:.4f} \n Recall: {:.4f} \n Specificity: {:.4f}".format(acc,f1,auc,prec,rec,specificity))
    
    roc_curve_plot(labels_test,score)
    cm=confusion_matrix(labels_test,preds)
    
    plot_confusion_matrix(cm,target_names=['NON-COVID','COVID'],normalize=False)
    
    return(score)
    


# In[276]:


vote_score=voting_classifier(df2_val,vallabels,'soft',df2_test,testlabels)


# In[277]:


def MultMetrics(scores,labels,threshold):
    """
    Function to evaluate proposed ensemble model
    
    Args
    ------
    scores:    scores from test set
    
    labels:    test set true labels
    
    threshold: cutoff value for decision boundary
    
    """
    scores_t=scores.values
    labels_test=labels.values
    labels_test=np.reshape(labels_test,(labels_test.shape[0]))
    scores_t=np.reshape(scores_t,(len(scores_t)))
    preds=scores_t>threshold
    preds=preds.astype(int)
    
    acc=accuracy(preds,labels_test)
    f1=f1_score(labels_test,preds)
    auc=roc_auc_score(labels_test,scores_t)
    prec=precision_score(labels_test,preds)
    rec=recall_score(labels_test,preds)
    TN=sum((preds==0)&(labels_test==0))
        
    FP=sum((preds==1)&(labels_test==0))

    specificity=TN/(TN+FP)
    print("\n Accuracy: {:.4f} \n F1 Score: {:.4f} \n AUC: {:.4f} \n Precision: {:.4f} \n Recall: {:.4f} \n Specificity: {:.4f}".format(acc,f1,auc,prec,rec,specificity))
    
    roc_curve_plot(labels_test,scores_t)
    cm=confusion_matrix(labels_test,preds)
    
    plot_confusion_matrix(cm,target_names=['NON-COVID','COVID'],normalize=False)
    


# In[278]:


MultMetrics(df7_test,testlabels,0.5)


# In[279]:



def macro_f1(test):
    """
    Function to estimate macro average F1 score
    
    Args
    ------
    test: test set scores in a numpy array or pandas dataframe
    
    """

    try:
        test=np.reshape(test.values,(len(test),))
    except:
        test=np.reshape(test,(len(test),))
    
    new_test=np.zeros((len(test),2))
    for i,val in enumerate(test):
        new_test[i,0]=1-val
        new_test[i,1]=val
    labels_test=testlabels.values
    labels_test=np.reshape(labels_test,(labels_test.shape[0]))

    new_labels=np.zeros((len(test),2))
    for i,val in enumerate(labels_test):
        if val>0.5:
            new_labels[i,1]=1
        else:

            new_labels[i,0]=1


    new_test=new_test>0.5
    print('Macro-averaged F1 score: {:.4f}'.format(f1_score(new_labels,new_test,average='macro')))


# In[280]:


macro_f1(df7_test)
        


# In[281]:


def metrics_models(labels,score):
    """
    Function to define each initial models evaluation metrics
    
    Args
    ------
    labels: test set true labels in a pandas dataframe
    
    score:  test set score in a pandas dataframe
    
    """
    for model in score.columns:
        auc_score=score[model]
        preds=score[model]>=0.5
        true_lab=labels.values[:,0]
        acc=accuracy(preds,true_lab)
        precision=precision_score(true_lab,preds)
        recall=recall_score(true_lab,preds)
        auc_=roc_auc_score(true_lab,auc_score)
        TN=sum((preds==0)&(true_lab==0))
        
        FP=sum((preds==1)&(true_lab==0))
        
        specificity=TN/(TN+FP)
        f1=f1_score(true_lab,preds)
        print("\n\n{}".format(model))
        print('\nAccuracy: {:.4f}\nPrecision: {:.4f}\nRecall: {:.4f}\nSpecificity: {:.4f}\nF1 Score: {:.4f}\nAUC: {:.4f}'.format(acc,precision,recall,specificity,f1,auc_))
        


# In[282]:


metrics_models(testlabels,df2_test)


# In[ ]:




