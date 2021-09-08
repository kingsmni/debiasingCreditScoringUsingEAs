import os,sys,time

import numpy as np
from scipy.optimize import minimize
from copy import deepcopy

from utility_functions import *
from random import seed, shuffle
     

def disparateTreatmentScore(w,X,sensitiveAttributeIndex=0):
    n,d = X.shape
    
    # create features for non-protected group (z=1)
    Xz1 = X.copy()
    Xz1[:,sensitiveAttributeIndex]=np.ones(n)
    
    # create features for protected group (z=0)
    Xz0 = X.copy()
    Xz0[:,sensitiveAttributeIndex]=np.zeros(n)
    
    # predictions for non-protected group (z=1) and protected group (z=0)
    yHatZ1 = np.sign(Xz1@w)
    yHatZ0 = np.sign(Xz0@w)
    
    # fraction of samples for which predictions deviate
    score = np.sum(yHatZ1!=yHatZ0) / n
    
    return score


def differenceDisparateImpact(X,y,sensitiveAttributeIndex=0):
    n,d = X.shape
    
    # number of instances of non-protected group (z=1) and protected group (z=0)
    n_z1 = np.sum(np.where(X[:,sensitiveAttributeIndex]==1,1,0))
    n_z0 = n - n_z1
    
    # number of instances of +ve class for non-protected and protected group
    n_y1_z1 = np.sum(np.where(y[X[:,sensitiveAttributeIndex]==1]==1,1,0))
    n_y1_z0 = np.sum(np.where(y[X[:,sensitiveAttributeIndex]==0]==1,1,0))
    
    # percentage +ve class for non-protected and protected groups
    p_y1_z1 = n_y1_z1 / n_z1
    p_y1_z0 = n_y1_z0 / n_z0
    
#     print('p_y1_z1: {:.4%}'.format(p_y1_z1))
#     print('p_y1_z0: {:.4%}'.format(p_y1_z0))
    
    return p_y1_z1, p_y1_z0


def differenceDisparateImpactModel(w,X,sensitiveAttributeIndex=0):
    yHat = np.sign(X@w)
    return differenceDisparateImpact(X,yHat,sensitiveAttributeIndex)


def differenceDisparateMistreatment(w,X,y,sensitiveAttributeIndex=0,type='OMR'):
    n,d = X.shape    
    yHat = np.sign(X@w)
    type = type.upper()

    if type=='OMR':        
        z1_rows = X[:,sensitiveAttributeIndex]==1
        z0_rows = X[:,sensitiveAttributeIndex]==0
    
    elif type=='FPR':
        z1_rows = (X[:,sensitiveAttributeIndex]==1) * (y==-1)
        z0_rows = (X[:,sensitiveAttributeIndex]==0) * (y==-1)
    
    elif type=='FNR':
        z1_rows = (X[:,sensitiveAttributeIndex]==1) * (y==1)
        z0_rows = (X[:,sensitiveAttributeIndex]==0) * (y==1)
        
    elif type=='FOR':
        z1_rows = (X[:,sensitiveAttributeIndex]==1) * (yHat==-1)
        z0_rows = (X[:,sensitiveAttributeIndex]==0) * (yHat==-1)
        
    elif type=='FDR':
        z1_rows = (X[:,sensitiveAttributeIndex]==1) * (yHat==1)
        z0_rows = (X[:,sensitiveAttributeIndex]==0) * (yHat==1)
        
    else:
        print('incorrect mistreatment type provided')
    
    n_z1 = np.sum(z1_rows)
    n_z0 = np.sum(z0_rows)

    n_yHatNotY_z1 = np.sum(y[z1_rows] != yHat[z1_rows])
    n_yHatNotY_z0 = np.sum(y[z0_rows] != yHat[z0_rows])
    
    # percentage misclassification for non-protected and protected groups
    p_yHatNotY_z1 = n_yHatNotY_z1 / n_z1
    p_yHatNotY_z0 = n_yHatNotY_z0 / n_z0
    
    return p_yHatNotY_z1, p_yHatNotY_z0


def differenceEqualOpportunity(w,X,y,sensitiveAttributeIndex=0):
    n,d = X.shape    
    yHat = np.sign(X@w)

    z1_rows = (X[:,sensitiveAttributeIndex]==1) * (y==1)
    z0_rows = (X[:,sensitiveAttributeIndex]==0) * (y==1)
       
    n_z1 = np.sum(z1_rows)
    n_z0 = np.sum(z0_rows)

    n_yHatEqualY_z1 = np.sum(y[z1_rows] == yHat[z1_rows])
    n_yHatEqualY_z0 = np.sum(y[z0_rows] == yHat[z0_rows])
    
    # percentage correct classification for non-protected and protected groups
    p_yHatEqualY_z1 = n_yHatEqualY_z1 / n_z1
    p_yHatEqualY_z0 = n_yHatEqualY_z0 / n_z0
    
    return p_yHatEqualY_z1, p_yHatEqualY_z0


def businessNecessityConstraint(w,X,y, loss, gamma):
    
    # note that this function can consume loss/gamma as scalars (when constraining total loss) or arrays (when constraining loss for each sample)
    
    if str(type(loss))=="<class 'numpy.float64'>" or type(gamma)==int or type(gamma)==float:
        assert str(type(loss))=="<class 'numpy.float64'>" and (type(gamma)==int or type(gamma)==float), "one of loss or gamma is an array with the other a scalar"
    else:
        assert len(loss)==len(gamma), "loss and gamma arrays have different lengths"
        
    n,d = X.shape    
    yHat = np.sign(X@w)
    
    if str(type(loss))=="<class 'numpy.float64'>":
        return logisticLoss(w,X,y) - (1+gamma) * loss
    else:
        return list(logisticLoss(w,X,y, returnIndividualLosses=True) - (1+gamma) * loss)