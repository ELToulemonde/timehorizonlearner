# -*- coding: utf-8 -*-
"""
Created on Monday November 28th 2016
@author: Alexis BONDU and Emmanuel-Lin TOULEMONDE
"""
import random
import matplotlib.pyplot as plt
import sklearn
import numpy as np
import pandas as pd
import progressbar

# Sample a dataFrame will putting every occurence of the same id in the same set
# ------------------------------------------------------------------------------
# To-do supprimer cette fonction car elle coute de la RAM!
def samplingDisjoinedIds(X, y, rateVal, varId, random_state=None):
    """Sample X_train and y_train according to varId, every occurence of an id will be in the same set. """
    # Return splited set
    ## Initialiszation
    fit_index, val_index = samplingDisjoinedIdsReturnIndex(X, rateVal, varId, random_state)

    ## Wrap-up
    return X.loc[fit_index], X.loc[val_index], y.loc[fit_index], y.loc[val_index]

# Sample a dataFrame will putting every occurence of the same id in the same set
# ------------------------------------------------------------------------------
def samplingDisjoinedIdsReturnIndex(X, rateVal, varId, random_state=None):
    """Sample X according to varId, every occurence of an id will be in the same set. """
    # Return splited indexes
    ## Initialiszation
    if random_state != None:
        random.seed(random_state)       
    ## Split ids
    list_ids = set(X[varId].unique())
    list_ids_fit = set(random.sample(list_ids, int(len(list_ids) * (1 - rateVal))))
    list_ids_val = list_ids.difference(list_ids_fit)
    
    ## From splited ids build  fit and val set
    fit_index = X[X[varId].isin(list_ids_fit)].index.tolist()
    val_index = X[X[varId].isin(list_ids_val)].index.tolist()
    
    ## Wrap-up
    return fit_index, val_index

# Indexing a dataFrame
# --------------------
def indexDataFrame(df, varId, varPeriod, afterNPeriods=1):
    """Add a column to df with index of the line for each individual at next period or None 
    Handle int period or list of periods otherwise it doesn't do anything."""
    # Initialization
    list_unique_periods = sorted(df[varPeriod].unique())
    
    # Sort data frame because algo will compare line i with line i+1, we want to have it ordered
    df = df.sort_values(by=[varId, varPeriod], axis=0)
    df.index = range(df.shape[0]) # Reindexaction after sort
    
    # Compute
    if isinstance(afterNPeriods, int):
        df = _indexDataFrame(df, list_unique_periods, varId, varPeriod, afterNPeriod = afterNPeriods)
    if isinstance(afterNPeriods, list):
        bar = progressbar.ProgressBar()
        for period in bar(afterNPeriods):
            df = _indexDataFrame(df, list_unique_periods, varId, varPeriod, afterNPeriod = period)
    
    return df


def _indexDataFrame(df, list_unique_periods, varId, varPeriod, afterNPeriod):
    # Initialization
    var_next_period = 'idNextPeriod' + str(afterNPeriod)
    # Get All the available periods
    
    if afterNPeriod > len(list_unique_periods):
        print ("Warning: indexDataFrame: afterNPeriod shoould be strictly lower than the number of unique elements in column " + str(varPeriod))
        return df
    
    # Build an index of next period
    index_next_period = {}
    for i in range(len(list_unique_periods) - afterNPeriod):
        index_next_period[list_unique_periods[i]] = list_unique_periods[i + afterNPeriod]
    # The last afterNPeriod period don't have a next period so we put None
    for i in range(len(list_unique_periods) - afterNPeriod, len(list_unique_periods)):
        index_next_period[list_unique_periods[i]] = None
        
    # We want to check if the following line has the same varId and the next varPeriod,
    # if it's true then next period exists otherwise it doesn't.
    idperiod_at_t = df[[varId, varPeriod]].copy()
    idperiod_at_tplus1 = df[[varId, varPeriod]].copy()
    
    idperiod_at_t = idperiod_at_t.drop(range(len(idperiod_at_t) - afterNPeriod, 
                                             len(idperiod_at_t)
                                            ), axis=0).reset_index(drop=True)
    idperiod_at_tplus1 = idperiod_at_tplus1.drop(range(afterNPeriod), axis=0).reset_index(drop=True)
    
    idperiod_at_t[varPeriod] = idperiod_at_t[varPeriod].apply(lambda x: index_next_period[x])
    
    # Check that next period exist on both id and period
    next_period_exist = (idperiod_at_t == idperiod_at_tplus1).all(axis=1) 
    for i in range(len(df) - afterNPeriod, len(df)):
        next_period_exist = next_period_exist.append(pd.Series(False))
    next_period_exist.index = range(next_period_exist.shape[0]) 
    # Add new column that will contain the index of the indivual at the next period
    # (if it exists) otherwise None
    df['idNextPeriod' + str(afterNPeriod)] = None
    df.loc[next_period_exist, var_next_period] = df.loc[next_period_exist, var_next_period].index + afterNPeriod
    
    return df
    
## Plot ROC curve and compute average AUC over classes
# -----------------------------------------------------
def aucAndRocCurve(y_truth, prediction, listOfClasses):
    """ To plot ROC curve and get AUC"""
    # Initialization
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    avg_auc = 0.
    nbr_classes = len(listOfClasses)
    
    # Computation
    for i in range(len(listOfClasses)):
        y_one_vs_all = y_truth == listOfClasses[i]
        if len(y_one_vs_all.unique()) == 2:
            fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y_one_vs_all, prediction[:, i])
            roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])
            auc = sklearn.metrics.roc_auc_score(y_one_vs_all, prediction[:, i])
            print "For class " + str(listOfClasses[i]) + " AUC is of " +str(auc)
            avg_auc += auc
        else:
            nbr_classes = nbr_classes - 1
    # Plot if possible
    if nbr_classes != 0:
        avg_auc = avg_auc / nbr_classes      
        i = 0 # To plot first ROC curve
        plt.figure()
        plt.plot(fpr[i], tpr[i], color='darkorange',
                 lw=2, label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
    else:
        # This the case where in next period every one is tagged to the same thing,
        # we can't evaluate ourself
        avg_auc = 0.5
    print avg_auc


def auc(y_truth, prediction, listOfClasses):
    """ To plot ROC curve and get AUC"""
    # Initialization
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    avg_auc = 0.
    nbr_classes = len(listOfClasses)
    
    # Computation
    for i in range(len(listOfClasses)):
        y_one_vs_all = y_truth == listOfClasses[i]
        if len(y_one_vs_all.unique()) == 2:
            fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y_one_vs_all, prediction[:, i])
            roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])
            auc = sklearn.metrics.roc_auc_score(y_one_vs_all, prediction[:, i])
            #print "For class " + str(listOfClasses[i]) + " AUC is of " +str(auc)
            avg_auc += auc
        else:
            nbr_classes = nbr_classes - 1
    # Plot if possible
    if nbr_classes != 0:
        avg_auc = avg_auc / nbr_classes    
    else:
        # This the case where in next period every one is tagged to the same thing,
        # we can't evaluate ourself
        avg_auc = 0.5
    #print avg_auc
    return avg_auc
    
def scaler(result):
    """Scale results by keeping mean and var"""
    result2 = result.copy()
    for i in range(result2.shape[1]):
        storeMean = np.mean(result2[:, i, 0])
        storeVar = np.var(result2[:, i, 0])
        for j in range(1, result2.shape[2]):
            result2[:, i, j] = (result2[:, i, j] -  np.mean(result2[:, i, j]))/  np.sqrt(np.var(result2[:, i, j])) * np.sqrt(storeVar) + storeMean
            
    return result2
    
    
	
	
## To use sklearn paralelisation
#It's a copy paste of a private function:
 ## Problem, not well imported? to check i had to import them by hands
def partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(get_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = (n_estimators // n_jobs) * np.ones(n_jobs,
                                                              dtype=np.int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)
    
    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()
    
from sklearn.externals.joblib import cpu_count
def get_n_jobs(n_jobs):
    """Get number of jobs for the computation.
    This function reimplements the logic of joblib to determine the actual
    number of jobs depending on the cpu count. If -1 all CPUs are used.
    If 1 is given, no parallel computing code is used at all, which is useful
    for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
    Thus for n_jobs = -2, all CPUs but one are used.
    Parameters
    ----------
    n_jobs : int
        Number of jobs stated in joblib convention.
    Returns
    -------
    n_jobs : int
        The actual number of jobs as positive integer.
    Examples
    --------
    >>> from sklearn.utils import _get_n_jobs
    >>> _get_n_jobs(4)
    4
    >>> jobs = _get_n_jobs(-2)
    >>> assert jobs == max(cpu_count() - 1, 1)
    >>> _get_n_jobs(0)
    Traceback (most recent call last):
    ...
    ValueError: Parameter n_jobs == 0 has no meaning.
    """
    if n_jobs < 0:
        return max(cpu_count() + 1 + n_jobs, 1)
    elif n_jobs == 0:
        raise ValueError('Parameter n_jobs == 0 has no meaning.')
    else:
        return n_jobs
        

##  Computing min and max according to a key (groups)
# Adapted from: https://stackoverflow.com/questions/8623047/group-by-max-or-min-in-a-numpy-array
def group_min(groups, data, varGroup, varData):
    if len(groups) == 0:
        return  pd.DataFrame({varGroup:np.array([]), varData:np.array([])}) 
    # sort with major key groups, minor key data
    order = np.lexsort((data, groups))
    groups = groups[order] # this is only needed if groups is unsorted
    data = data[order]
    # construct an index which marks borders between groups
    index = np.empty(len(groups), 'bool')
    index[0] = True
    index[1:] = groups[1:] != groups[:-1]
    return  pd.DataFrame({varGroup:np.unique(groups), varData:data[index]}) 
    
    
#max is very similar
def group_max(groups, data, varGroup, varData):
    if len(groups) == 0:
        return  pd.DataFrame({varGroup:np.array([]), varData:np.array([])}) 
    order = np.lexsort((data, groups))
    groups = groups[order] #this is only needed if groups is unsorted
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    return pd.DataFrame({varGroup:np.unique(groups), varData:data[index]}) 
	
	
## all list
# ---------
__all__ = ["indexDataFrame", "samplingDisjoinedIds", "samplingDisjoinedIdsReturnIndex", "aucAndRocCurve", "auc"]