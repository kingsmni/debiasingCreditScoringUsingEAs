import os,sys,time

import numpy as np
from scipy.optimize import minimize
from copy import deepcopy

from random import seed, shuffle


# NB I have included the 'returnIndividualLosses' flag to faciliate the use of individual losses by the business necessity clause

def logisticLoss(w,X,y, returnIndividualLosses=False):
    n,d = X.shape
#     print('len w: {}, x shape: {} y shape: {}'.format(len(w), X.shape, y.shape))
    assert len(w)==d, "dimension of data does not match weight vector dimension"
    assert len(y)==n, "number of data items does not match number of target values"
    
    # use an interim term z=(X@w)*-y in order to facilitate conditional statement
    z = (X@w)*-y
    
    if returnIndividualLosses==False:
        result = 0
        zNegatives = z[z<0]
        zPositives = z[z>=0]
        result += np.sum(np.log(1+np.exp(zNegatives)))
        result += np.sum(zPositives+np.log(1+np.exp(-zPositives)))
        
        return result
    
    else:
        return np.log(1+np.exp(z))
    
# helper function to return error rate from a model

def errorRate(w,X,y):
    n,d = X.shape
    yHat = np.sign(X@w)
    errorRate = np.sum(yHat!=y)/n
    
    return errorRate
    
    
# helper function to compute the losses associated with the unconstrained problem (i.e. with no fairness objectives)

def computeUnconstrainedLoss(trainX, trainY, testX, testY, max_iter=100000, returnIndividualLosses=False):
    w = minimize(fun = logisticLoss,
            x0 = np.random.rand(trainX.shape[1],),
            args = (trainX, trainY),
            method = 'SLSQP',
            options = {"maxiter":max_iter},
            constraints = []
            )
    return logisticLoss(w.x,testX,testY,returnIndividualLosses)
   
    
# helper function to compute the minimum and maximum objective function values achieved by an algorithm

def resultsMinMax(algorithm, numberStd=2):
    numberObjectives = len(algorithm.result[0].objectives)
    resultsSummaryArray = np.zeros((6,numberObjectives)) # rows: min, max, mean, std, adj. min, adj. max
    
    for objective in range(numberObjectives):
        objectiveResults = []
        for solution in algorithm.result:
            objectiveResults.append(solution.objectives[objective])
        resultsSummaryArray[0,objective] = min(objectiveResults)
        resultsSummaryArray[1,objective] = max(objectiveResults)
        resultsSummaryArray[2,objective] = np.mean(objectiveResults)
        resultsSummaryArray[3,objective] = np.std(objectiveResults)
        resultsSummaryArray[4,objective] = resultsSummaryArray[2,objective] - (numberStd * resultsSummaryArray[3,objective])
        resultsSummaryArray[5,objective] = resultsSummaryArray[2,objective] + (numberStd * resultsSummaryArray[3,objective])
    
    return resultsSummaryArray


# helper function to return the range for each variable that underpins the results

def findRange(algorithm):
    numberOfVariables = len(algorithm.result[0].variables)
    variableSummaryArray = np.zeros((2,numberOfVariables))
    
    for variable in range(numberOfVariables):
        variableResults = []
        for solution in algorithm.result:
            variableResults.append(solution.variables[variable])
        variableSummaryArray[0,variable] = min(variableResults)
        variableSummaryArray[1,variable] = max(variableResults)
    
    return variableSummaryArray


# helper function to convert the switch from loss value to error rate on a trained algorithm
# note that this means that we convert that objective into a figure in [0,1].

def convertLossToErrorRate(algorithm,X,y,lossObjectiveIndex=0): 
    n,d = X.shape
    copyAlgorithm = deepcopy(algorithm)
    
    for resultNumber, result in enumerate(copyAlgorithm.result):
        w = result.variables
#         n,d = X.shape    
        yHat = np.sign(X@w)
        singleErrorRate = np.sum(yHat!=y)/n
#         print(singleErrorRate)
        copyAlgorithm.result[resultNumber].objectives[lossObjectiveIndex]=singleErrorRate
        
    return copyAlgorithm


def findExtremes_averagePoint(algorithm):
    numberObjectives = len(algorithm.result[0].objectives)
    numberResults = len(algorithm.result)
    
    singleRunExtremesArray = np.zeros((numberObjectives,numberObjectives))
    
    objectiveResults = np.zeros((numberResults,numberObjectives))
    
    for i, solution in enumerate(algorithm.result):
        objectiveResults[i,:] = list(solution.objectives)
    
    ind = np.argsort(objectiveResults, axis=0)
    for objective in range(numberObjectives):
        singleRunExtremesArray[objective, :] = objectiveResults[ind[0,objective],:]
        
    averagePoint = np.mean(objectiveResults,axis=0)
    
    return singleRunExtremesArray, averagePoint 

def sortSolutions(algorithm):
    numberObjectives = len(algorithm.result[0].objectives)
    numberResults = len(algorithm.result)
    
    accuracyList = np.zeros(numberResults)
    
    for i, solution in enumerate(algorithm.result):
        accuracyList[i] = solution.objectives[0]
    
    ind = np.argsort(accuracyList, axis=0)
    accList = accuracyList[ind]
        
    return ind, accList

def returnDIlist(algorithm, X, sensitiveAttributeIndex=0):
    # returns 4 columns: impact for z1, impact for z0, z1-z0 impact, misclassification rate
    ind, accList = sortSolutions(algorithm)
    
    numberResults = len(algorithm.result)
    DIarray = np.zeros((numberResults,4))
    
    for i, indexNumber in enumerate(ind):
        w = algorithm.result[indexNumber].variables
        DIarray[i,:2] = differenceDisparateImpactModel(w,X,sensitiveAttributeIndex)
        
    DIarray[:,2] = np.abs(DIarray[:,0]-DIarray[:,1])
    DIarray[:,3] = accList
    
    return DIarray

def returnDMlist(algorithm, X, y, sensitiveAttributeIndex=0, type='OMR'):
    # returns 4 columns: error rate for z1, error rate for z0, z1-z0 error rate, misclassification rate
    ind, accList = sortSolutions(algorithm)
    
    numberResults = len(algorithm.result)
    DMarray = np.zeros((numberResults,4))
    
    for i, indexNumber in enumerate(ind):
        w = algorithm.result[indexNumber].variables
        DMarray[i,:2] = differenceDisparateMistreatment(w,X,y,sensitiveAttributeIndex,type)
        
    DMarray[:,2] = np.abs(DMarray[:,0]-DMarray[:,1])
    DMarray[:,3] = accList
    
    return DMarray

def returnEOlist(algorithm, X, y, sensitiveAttributeIndex=0):
    # returns 4 columns: opportunity for z1, opportunity for z0, z1-z0 opportunity, misclassification rate
    ind, accList = sortSolutions(algorithm)
    
    numberResults = len(algorithm.result)
    EOarray = np.zeros((numberResults,4))
    
    for i, indexNumber in enumerate(ind):
        w = algorithm.result[indexNumber].variables
        EOarray[i,:2] = differenceEqualOpportunity(w,X,y,sensitiveAttributeIndex)
        
    EOarray[:,2] = np.abs(EOarray[:,0]-EOarray[:,1])
    EOarray[:,3] = accList
    
    return EOarray


# Functions to return lowest error rate and associated weight vector obtainable given fairness values

def minErrorRate_given_targetMeasures(algorithm, targetMeasures):

    numberVariables = len(algorithm.result[0].variables)
    numberObjectives = len(algorithm.result[0].objectives)
    numberResults = len(algorithm.result)

    solutionsArray = np.zeros((numberResults,1+numberObjectives+numberVariables))
    solutionsArray[:,0] = range(numberResults)

    for i, solution in enumerate(algorithm.result):
        solutionsArray[i,1:] = list(solution.objectives) + list(solution.variables)

    filteredSolutionsArray = solutionsArray

    for i, target in enumerate(targetMeasures):
        filteredSolutionsArray = filteredSolutionsArray[filteredSolutionsArray[:,i+2]<=target]
        
    if len(filteredSolutionsArray) == 0:
        minErrorRate = 1.0 # placeholder
        minErrorRate_resultIndex = 0.0 # placeholder
    else:
        minErrorRate = min(filteredSolutionsArray[:,1])
        minErrorRate_resultIndex = filteredSolutionsArray[filteredSolutionsArray[:,1].argmin(),0]
    
    return minErrorRate, minErrorRate_resultIndex


def minErrorRateSet_given_targetMeasuresArray(algorithm, targetMeasureArray):
    
    minErrorRateSet = np.zeros((len(targetMeasureArray),2))
    
    for i, targetMeasures in enumerate(targetMeasureArray):
        minErrorRate, minErrorRate_resultIndex = minErrorRate_given_targetMeasures(algorithm,targetMeasures)
        minErrorRateSet[i,0] = minErrorRate
        minErrorRateSet[i,1] = minErrorRate_resultIndex
        
    return minErrorRateSet


# FPR and FNR computations for a model

def FPRcompute(w,X,y):
    n,d = X.shape
    yHat = np.sign(X@w)
    
    yNeg_rows = (y==-1)
    n_yNeg = np.sum(yNeg_rows)
    
    n_yHatPos_given_yNeg = np.sum(y[yNeg_rows] != yHat[yNeg_rows])
    p_yHatPos_given_yNeg = n_yHatPos_given_yNeg / n_yNeg
    
    return p_yHatPos_given_yNeg


def FNRcompute(w,X,y):
    n,d = X.shape
    yHat = np.sign(X@w)
    
    yPos_rows = (y==1)
    n_yPos = np.sum(yPos_rows)
    
    n_yHatNeg_given_yPos = np.sum(y[yPos_rows] != yHat[yPos_rows])
    p_yHatNeg_given_yPos = n_yHatNeg_given_yPos / n_yPos
    
    return p_yHatNeg_given_yPos