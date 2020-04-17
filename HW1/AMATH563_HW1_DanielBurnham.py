#!/usr/bin/env python
# coding: utf-8

# In[1]:


from mnist import MNIST
from os.path import expanduser
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import fmin
from sklearn import preprocessing
from matplotlib import rcParams
from sklearn import linear_model
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.ticker as plticker
from matplotlib.patches import Patch
from tqdm.notebook import tqdm
from copy import deepcopy
import math
from matplotlib import gridspec
from PIL import Image
import pandas as pd
import random
import itertools
from scipy import interp
import matplotlib.cm as cm
from sklearn.metrics import *
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import resample
from sklearn.model_selection import KFold


# In[8]:


'''boxPlotter takes in a 4D array X containing cross validated model coefficients for various model fit parameter 
 values (e.g. alpha, or l1 ratio) and plots box plots for the cross validated data for eah coefficient value
 and colors these box plots by which column of x they come from. This serves to indcate not only which model
 coefficients have the highest values, but which of the multiple outputs they contribute to modeling.
 
    INPUTS:
    X: numpy.ndarray of data to analyze. Expected shape is (m,p,c,a) where m is the number of variables modelled,
        p is the number of outputs to regress on, c is the number of cross validation iterations, 
        and a is the number of model fit parameters (e.g. alpha, l1 ratio) varied.
    categoreis: list of scalar values indicating model fit parameters to plot box plots for.
    top_perc: scalar value indicating the percent of largest coefficient values to plot. Default of 1%.
    medianbar: scalar value indicating the line width of the median bar in each box plot. Default of 3.
    fontsize: scalar value indicating the font size for each generated plot. Default of 12.
    figsize: list of two scalar values [w,h] indicating the width (w) and height (h) of each plot. As 
            the number of top coefficients to plot increase the width of the plot should be adjusted
            accordingly. Default is [25, 5].
    class_title: string value that allows control over the title above each plot. Title appears as,
                Top %'+ str(top_perc)+ ' Model Coefficients by Digit Class, ' + class_title + str(categories[i])
    
    OUTPUTS:
    plots: sequence of box plots.
    
    NOTES:
    The color sequece used to discriminate columns of X for each coefficient can discriminate
    up to max of 10 categories.
'''
def boxPlotter(X, categories, top_perc=1.0, medianbar = 3, fontsize = 12, figsize = [25, 5], class_title='Var = '):
    m = X.shape[0]
    p = X.shape[1]
    multiple = len(categories) > 1
    for i in tqdm(range(0, len(categories))):
        if multiple:
            x = np.average(X[:,:,:,i], axis=2)
        else:
            x = np.average(X[:,:,:], axis=2)
        xflat = x.flatten()
        top = int(((m*p)*(top_perc/100)))
        ind = np.argpartition(xflat, -top, axis=0)[-top:]
        ind_sort = ind[np.argsort(xflat[ind])[::-1]]
        (pix, dig) = np.unravel_index(ind_sort, (m,p))
        if multiple:
            data = X[:,:,:,i][pix,dig,:]
        else:
            data = X[:,:,:][pix,dig,:]

        plt.rcParams['figure.figsize'] = figsize
        rcParams.update({'font.size': fontsize})
        medianprops = dict(linestyle='-.', linewidth=medianbar)
        fig, ax = plt.subplots()

        cols = []
        legend_elements = []

        for j in range(0, len(ind)):
            bp = ax.boxplot(data[j].T, positions = [j], medianprops=medianprops, patch_artist=True)
            color = 'C' + str(dig[j])

            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(bp[element], color=color)
            for patch in bp['boxes']:
                patch.set(facecolor=color)

            if dig[j] not in cols:
                cols.append(dig[j])


        col_array = np.array(cols)
        cols_sorted = col_array[np.argsort(col_array)]
        for c in cols_sorted:
            color = 'C' + str(c)
            legend_elements.append(Patch(facecolor=color, edgecolor=color, label='coeff col ' + str(c)))

        ax.legend(handles=legend_elements, loc='best')
        ax.set_xticklabels(pix)
        ax.set_title('Top %'+ str(top_perc)+ ' Model Coefficients by Digit Class, ' + class_title + str(categories[i]) )



"""
This function prints and plots the confusion matrix.
Normalization can be applied by setting `normalize=True`.
Adapted from code at: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
"""
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    bottom, top = plt.ylim() 
    #plt.ylim(bottom + 0.5, top - 0.5)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
'''Takes in a one hout encoded array where each column is a different class, and returns a 1D array with
the class value for each row represented by the index of the 1 hot encoding in that row.'''
def class_decoder(b):
    rows = b.shape[0]
    cols = b.shape[1]
    decoded = []
    for i in range(0, rows):
        maxValue = 0
        maxInd = 0
        for j in range(0, cols):
            if b[i, j] > maxValue:
                maxValue = b[i, j]
                maxInd = j
        decoded.append(maxInd)
    return decoded

'''Takes in an array of model coefficients finds the top terms up to the number of terms indicated (num_terms), keeps
those values, and zeros out the rest of the coefficient values. This allows for testing of model sparsity. If X is 3D
the third axis is interpreted as cross validated coefficient values. The coefficient to compare to find maximal
values is the average of coefficients across cross validation. The CV data should be along the 3rd data axis.'''
def zero_out(X, num_terms):
    dims = X.shape
    if len(dims) > 2:
        x = np.average(X, axis=2)
    else:
        x = X
    xflat = x.flatten()
    ind = np.argpartition(xflat,-(int(num_terms)))[-(int(num_terms)):]
    ind_sort = ind[np.argsort(xflat[ind])[::-1]]
    mask = np.zeros(len(xflat))
    mask[ind_sort] = 1
    xflat_z = xflat*mask
    x_zeroed = np.reshape(xflat_z, dims)
    return x_zeroed


# In[3]:


file = '/Documents/AMATH Masters/AMATH563/HW1/dataset/MNIST'
home = expanduser("~") + file
mnist = MNIST(home)
x_train_raw, y_train_raw = mnist.load_training() #60000 samples
x_test_raw, y_test_raw = mnist.load_testing()    #10000 samples

#convert lists to numpy arrays
x_train = np.asarray(x_train_raw) #60000 x 784 where 784 is flattened 28x28 pixel data
y_train = np.asarray(y_train_raw) #60000 x 1 where each entry is the digit class
x_test = np.asarray(x_test_raw) #60000 x 784 where 784 is flattened 28x28 pixel data
y_test = np.asarray(y_test_raw) #60000 x 1 where each entry is the digit class

#label binarize the class arrays, i.e. each column reprents a digit class 
#with binary values representing whether or not the specific image is of that digit class
lb = preprocessing.LabelBinarizer()
lb.fit(y_train)
y_train = lb.transform(y_train)# 60000 x 10 where ach column is a digit class (0 through 9)

lb = preprocessing.LabelBinarizer()
lb.fit(y_test)
y_test = lb.transform(y_test)# 10000 x 10 where each column is a digit class (0 through 9)

n = x_train.shape[0] #number of images in data set (training set has 60000 flattened images)
m = x_train.shape[1] #number of pixels (28x28 yields 784 pixels total)
p = y_train.shape[1] #number of classes (10 classes, 0 through 9)
A = x_train
b = y_train


# In[4]:


lam = 0.1
cv_folds = 5
kf = KFold(n_splits = cv_folds, random_state = 17, shuffle = True)

E1 = np.zeros(cv_folds)
E2 = np.zeros(cv_folds)
E3 = np.zeros(cv_folds)
E4 = np.zeros(cv_folds)
E5 = np.zeros(cv_folds)
E6 = np.zeros(cv_folds)

X1 = np.zeros((m,p,cv_folds))
X2 = np.zeros((m,p,cv_folds))
X3 = np.zeros((m,p,cv_folds))
X4 = np.zeros((m,p,cv_folds))
X5 = np.zeros((m,p,cv_folds))
X6 = np.zeros((m,p,cv_folds))
jj = 0
for train_index, test_index in kf.split(A):
    print("Progress {:2.1%}".format(jj / cv_folds), end="\r")#print progress of paint can position acquisition
    #print("Progress {:2.1%}".format(jj / cv_folds), end="\r")#print progress
    x1 = np.linalg.pinv(A[train_index,:]) @ b[train_index,:]
    b1 = A[train_index,:] @ x1
    E1[jj] = np.linalg.norm(b[train_index,:] - b1,ord=2)/np.linalg.norm(b[train_index,:],ord=2)

    x2 = np.linalg.lstsq(A[train_index,:],b[train_index,:],rcond=None)[0]
    b2 = A[train_index,:] @ x2
    E2[jj] = np.linalg.norm(b[train_index,:]-b2,ord=2)/np.linalg.norm(b[train_index,:],ord=2)

    regr3 = linear_model.ElasticNet(alpha=1.0, copy_X=True, l1_ratio=lam, max_iter=10**5,random_state=0)
    regr3.fit(A[train_index,:], b[train_index,:])  
    x3 = np.transpose(regr3.coef_)
    b3 = A[train_index,:] @ x3
    E3[jj] = np.linalg.norm(b[train_index,:]-b3,ord=2)/np.linalg.norm(b[train_index,:],ord=2)

    regr4 = linear_model.ElasticNet(alpha=0.8, copy_X=True, l1_ratio=lam, max_iter=10**5,random_state=0)
    regr4.fit(A[train_index,:], b[train_index,:])  
    x4 = np.transpose(regr4.coef_)
    b4 = A[train_index,:] @ x4
    E4[jj] = np.linalg.norm(b[train_index,:]-b4,ord=2)/np.linalg.norm(b[train_index,:],ord=2)

    regr5 = MultiOutputRegressor(linear_model.HuberRegressor(), n_jobs=-1)
    huber = regr5.fit(A[train_index,:], b[train_index,:]) # matlab's robustfit() does not have an exact sklearn analogue

    x5 = np.empty([m, p])
    for i in range(0, len(huber.estimators_)):
        x5[:, i] = huber.estimators_[i].coef_

    b5 = A[train_index,:] @ x5
    E5[jj] = np.linalg.norm(b[train_index,:]-b5,ord=2)/np.linalg.norm(b[train_index,:],ord=2)

    ridge = linear_model.Ridge(alpha=1.0).fit(A[train_index,:],b[train_index,:])
    x6 = np.transpose(ridge.coef_)
    b6 = A[train_index,:] @ x6
    E6[jj] = np.linalg.norm(b[train_index,:] - b6,ord=2)/np.linalg.norm(b[train_index,:],ord=2)
    
    X1[:,:,jj] = x1
    X2[:,:,jj] = x2
    X3[:,:,jj] = x3
    X4[:,:,jj] = x4
    X5[:,:,jj] = x5
    X6[:,:,jj] = x6
    
    jj = jj + 1

    
Err = np.column_stack((E1,E2,E3,E4,E5,E6))

reg_styles = ['pinv', 'lstsq', 'elastic (alpha=1)', 'elastic (alpha=0.8)', 'huber', 'ridge']#types of Ax=b solvers used
Xdict = [X1, X2, X3, X4, X5, X6]#list of model coefficients for plotting

plt.rcParams['figure.figsize'] = [15, 21]
rcParams.update({'font.size': 12})
fig,axs = plt.subplots(len(reg_styles),1, sharex=True)
axs = axs.reshape(-1)

for j in range(0, len(reg_styles)):
    x = np.average(Xdict[j], axis=2)#average across cross validated coefficient values
    x_pcolor = axs[j].pcolor(x.T,cmap='Greys', vmin=-0.06, vmax=0.16)
    fig.colorbar(x_pcolor, ax=axs[j])
    axs[j].set_ylabel('Digit Class')
    axs[j].set_title(reg_styles[j])
    plt.setp(axs[j].get_xticklabels(), visible=False)

plt.rcParams['figure.figsize'] = [8, 8]

plt.figure()
plt.boxplot(Err)
plt.xticks([1, 2, 3, 4, 5, 6], ['pinv', 'np.linalg.solve', 'Elastic 1.0', 'Elastic 0.8', 'Huber', 'Ridge'], rotation=40)
plt.show()


# In[5]:


top_perc = 0.5 #top percent of coefficients to extract
Xdict4D = np.transpose(np.array(Xdict), (1, 2, 3, 0))#change Xdict from list of 3D arrays to 4D array

boxPlotter(Xdict4D, reg_styles, top_perc=0.5, medianbar=3, fontsize=12, figsize=[20, 5], class_title='Regression Style: ')


# In[7]:


#Work with elastic net to test out different l1_ratios to promote sparsity
cv_folds = 5
kf = KFold(n_splits = cv_folds, random_state = 17, shuffle = True)
l1_ratios = [1.0, 0.75, 0.5, 0.25]
Esparse = np.zeros((cv_folds, len(l1_ratios)))
Xsparse = np.zeros((m,p,cv_folds,len(l1_ratios)))

i = 0
for train_index, test_index in kf.split(A):
    print("Progress {:2.1%}".format(i / cv_folds), end="\r")#print progress of paint can position acquisition
    for j in range(0, len(l1_ratios)):
        regr_sparse = linear_model.ElasticNet(alpha=1.0, copy_X=True, l1_ratio=l1_ratios[j], max_iter=10**5,random_state=0)
        regr_sparse.fit(A[train_index,:], b[train_index,:])  
        xsparse = np.transpose(regr_sparse.coef_)
        bsparse = A[train_index,:] @ xsparse
        Esparse[i, j] = np.linalg.norm(b[train_index,:]-bsparse,ord=2)/np.linalg.norm(b[train_index,:],ord=2)
        Xsparse[:,:,i,j] = xsparse
    i = i + 1#iterate cross validation counter
        
boxPlotter(Xsparse, l1_ratios, top_perc=0.5, medianbar=3, fontsize=12, figsize=[20, 5], class_title='l1 ratio: ')   


# In[11]:


#Work with elastic net to test out different alpha values to promote sparsity
cv_folds = 5
kf = KFold(n_splits = cv_folds, random_state = 17, shuffle = True)
Elasso = np.zeros((cv_folds, len(l1_ratios)))
Xlasso = np.zeros((m,p,cv_folds,len(l1_ratios)))
alphas = [0.25, 0.5, 0.75, 1.0]

i = 0
for train_index, test_index in kf.split(A):
    print("Progress {:2.1%}".format(i / cv_folds), end="\r")#print progress of paint can position acquisition
    for j in range(0, len(alphas)):
        lasso = linear_model.ElasticNet(alpha=alphas[j], copy_X=True, l1_ratio=1.0, max_iter=10**5,random_state=0)
        lasso.fit(A[train_index,:], b[train_index,:])  
        xlasso = np.transpose(lasso.coef_)
        blasso = A[train_index,:] @ xlasso
        Elasso[i, j] = np.linalg.norm(b[train_index,:]-blasso,ord=2)/np.linalg.norm(b[train_index,:],ord=2)
        Xlasso[:,:,i,j] = xlasso
    i = i + 1

boxPlotter(Xlasso, alphas, top_perc=0.5, medianbar=3, fontsize=12, figsize=[20, 5], class_title='Alpha: ')


# In[7]:


#Apply your most important pixels to the test data set to see how accurate you are with as few pixels as possible.
#Best l1 ratio appears to be 1.0 and best alpha value appears to be 0.25. Try a model employing both values.
#Work with elastic net to test out different alpha values to promote sparsity
cv_folds = 5
kf = KFold(n_splits = cv_folds, random_state = 17, shuffle = True)
l1_ratio = 1.0
Elasso_best = np.zeros((cv_folds))
Xlasso_best = np.zeros((m,p,cv_folds))
alpha = 0.25

i = 0
for train_index, test_index in kf.split(A):
    print("Progress {:2.1%}".format(i / cv_folds), end="\r")#print progress of paint can position acquisition
    lasso_best = linear_model.ElasticNet(alpha=alpha, copy_X=True, l1_ratio=l1_ratio, max_iter=10**5,random_state=0)
    lasso_best.fit(A[train_index,:], b[train_index,:])  
    xlasso_best = np.transpose(lasso_best.coef_)
    blasso_best = A[train_index,:] @ xlasso_best
    Elasso_best[i] = np.linalg.norm(b[train_index,:]-blasso_best,ord=2)/np.linalg.norm(b[train_index,:],ord=2)
    Xlasso_best[:,:,i] = xlasso_best
    i = i + 1

boxPlotter(Xlasso_best, [alpha], top_perc=0.5, medianbar=3, fontsize=12, figsize=[20, 5], class_title='Alpha: ')

Atest = x_test
btest = y_test
classes = np.unique(y_test_raw)
preds = lasso_best.predict(Atest)#test the model
preds_1D = np. array(class_decoder(preds))
btest_1D = np. array(class_decoder(btest))

CM = confusion_matrix(btest_1D, preds_1D)# Confusion Matrix
plot_confusion_matrix(CM, classes)#plot of the multiclass confusion matrix

aScore = accuracy_score(btest_1D, preds_1D)#accuracy score
P = precision_score(btest_1D, preds_1D, average='weighted')#precision score
R = recall_score(btest_1D, preds_1D, average='weighted')#recall score
F1 = f1_score(btest_1D, preds_1D, average='weighted')#F1 score
a = {'Results': [aScore, P, R, F1]}#series of evaluation results
aFrame_k = pd.DataFrame(a, index = ['Accuracy', 'Precision', 'Recall', 'F1'])
print(aFrame_k)


# In[151]:


#perform analysis for degree of sparsity for best lasso model
num_terms = [1, 5, 10, 25, 50, 150, 300, 600]
xlasso_bz_plot = np.empty((m, p, cv_folds, len(num_terms)))
xlasso_bz = np.empty((m, p,len(num_terms)))
for i in range(0, len(num_terms)):

    xlasso_bz[:,:,i] = zero_out(np.average(Xlasso_best[:,:,:], axis=2), num_terms[i])#zero out coefficients for average cv lasso model
    mask = np.ones(xlasso_bz[:,:,i].shape)*xlasso_bz[:,:,i]
    mask = np.repeat(mask[:, :, np.newaxis], 5, axis=2)
    
    xlasso_bz_plot[:,:,:,i] = mask*Xlasso_best[:,:,:]

boxPlotter(xlasso_bz_plot, num_terms, top_perc=0.5, medianbar=3, fontsize=12, figsize=[20, 5], class_title='Number of Terms: ')

for i in range(0, len(num_terms)):
    
    Atest = x_test
    btest = y_test
    classes = np.unique(y_test_raw)
    preds =  Atest @ xlasso_bz[:,:,i]#test the model
    preds_1D = np. array(class_decoder(preds))
    btest_1D = np. array(class_decoder(btest))
    
    CM = confusion_matrix(btest_1D, preds_1D)# Confusion Matrix
    plot_confusion_matrix(CM, classes, title='Confusion matrix: ' + str(num_terms[i]) + ' Model Terms')#plot of the multiclass confusion matrix


    aScore = accuracy_score(btest_1D, preds_1D)#accuracy score
    P = precision_score(btest_1D, preds_1D, average='weighted')#precision score
    R = recall_score(btest_1D, preds_1D, average='weighted')#recall score
    F1 = f1_score(btest_1D, preds_1D, average='weighted')#F1 score
    a = {'Results': [aScore, P, R, F1]}#series of evaluation results
    aFrame_k = pd.DataFrame(a, index = ['Accuracy', 'Precision', 'Recall', 'F1'])
    print(aFrame_k)


# In[37]:


#Redo analysis with each digit individually
cv_folds = 5
kf = KFold(n_splits = cv_folds, random_state = 17, shuffle = True)
classes = np.unique(y_test_raw)
l1_ratio = 1.0
Elasso_dig = np.zeros((len(classes), cv_folds))
Xlasso_dig = np.zeros((m,1,len(classes), cv_folds))
alpha = 0.25
top_perc=5
medianbar=3
fontsize=12
figsize=[20, 5]
class_title='Alpha: '

for i in tqdm(range(0, len(classes))):
    #Need to up sample minority digit class to resolve class imbalance problem
    btrain = b[:, i]
    dig_ind = np.where(btrain == 1)[0]
    other_ind = np.where(btrain == 0)[0]
    upsampled_b = resample(btrain[dig_ind], replace=True,n_samples=len(other_ind),random_state=123)
    upsampled_A = resample(A[dig_ind], replace=True,n_samples=len(other_ind),random_state=123)
    btrain = np.append(btrain[other_ind], upsampled_b, axis=0)
    Atrain = np.append(A[other_ind,:], upsampled_A, axis=0)
    
    j = 0
    for train_index, test_index in kf.split(Atrain):
    #perform cross validation per digit class
        lasso_dig = linear_model.ElasticNet(alpha=alpha, copy_X=True, l1_ratio=l1_ratio, max_iter=10**5,random_state=0)
        lasso_dig.fit(Atrain[train_index,:], btrain[train_index]) 
        xlasso_dig = np.transpose(lasso_dig.coef_)
        blasso_dig = Atrain[train_index,:] @ xlasso_dig
        Elasso_dig[i, j] = np.linalg.norm(btrain[train_index]-blasso_dig,ord=2)/np.linalg.norm(btrain[train_index],ord=2)
        Xlasso_dig[:,:,i,j] = np.expand_dims(xlasso_dig, axis=1)
        j = j + 1

for i in tqdm(range(0, len(classes))):
    m = Xlasso_dig.shape[0]
    p = Xlasso_dig.shape[1]
    xav = np.average(Xlasso_dig[:,:,i,:], axis=2)
    xflat = xav.flatten()
    '''
    top = int(((m*p)*(top_perc/100)))
    top_inds = np.argpartition(xflat, -top, axis=0)[-top:]
    top_vals = xflat[top_inds]
    data = top_vals[np.argsort(top_vals)[::-1]]
    '''
    ind = np.argpartition(xflat, -top, axis=0)[-top:]
    ind_sort = ind[np.argsort(xflat[ind])[::-1]]
    (pix, dig) = np.unravel_index(ind_sort, (m,p))
    
    data = Xlasso_dig[:,:,i,:][pix,dig,:]
    plt.rcParams['figure.figsize'] = figsize
    rcParams.update({'font.size': fontsize})
    medianprops = dict(linestyle='-.', linewidth=medianbar)
    fig, ax = plt.subplots()

    cols = []
    legend_elements = []

    for j in range(0, len(top_vals)):
        bp = ax.boxplot(data[j], positions = [j], medianprops=medianprops, patch_artist=True)
        color = 'C' + str(i)

        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color=color)
        for patch in bp['boxes']:
            patch.set(facecolor=color)

    ax.set_xticklabels(top_inds[np.argsort(top_vals)[::-1]])
    ax.set_title('Top %'+ str(top_perc)+ ' Model Coefficients for Digit Class, ' + str(classes[i]) )


    #upsample minority class in test data
    Atest = x_test
    btest = y_test[:, i]
    dig_ind = np.where(btest == 1)[0]
    other_ind = np.where(btest == 0)[0]
    upsampled_b = resample(btest[dig_ind], replace=True,n_samples=len(other_ind),random_state=123)
    upsampled_A = resample(Atest[dig_ind], replace=True,n_samples=len(other_ind),random_state=123)
    btest_itr = np.append(btest[other_ind], upsampled_b, axis=0)
    Atest_itr = np.append(Atest[other_ind,:], upsampled_A, axis=0)

    preds_itr =  Atest_itr @ xav#test the model
    for k in range(0, len(preds_itr)):
        if preds_itr[k] >=0.55:
            preds_itr[k] = 1
        else:
            preds_itr[k] = 0
    
    preds_itr = preds_itr.astype(int)
    CM = confusion_matrix(btest_itr, preds_itr)# Confusion Matrix
    plot_confusion_matrix(CM, [0, 1])#plot of the multiclass confusion matrix


    aScore = accuracy_score(btest_itr, preds_itr)#accuracy score
    P = precision_score(btest_itr, preds_itr, average='weighted')#precision score
    R = recall_score(btest_itr, preds_itr, average='weighted')#recall score
    F1 = f1_score(btest_itr, preds_itr, average='weighted')#F1 score
    a = {'Results': [aScore, P, R, F1]}#series of evaluation results
    aFrame_k = pd.DataFrame(a, index = ['Accuracy', 'Precision', 'Recall', 'F1'])
    print(aFrame_k)

