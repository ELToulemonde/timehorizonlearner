# -*- coding: utf-8 -*-
"""
Created on Monday November 28th 2016
@author: Emmanuel-Lin TOULEMONDE and Alexis BONDU
"""

import math
import random
import time

import numpy as np
import pandas as pd
from sklearn import tree

from .genericFunctions import auc, group_min, group_max

__all__ = ["timeHorizonLearner"]


class timeHorizonLearner(tree.DecisionTreeClassifier):
    """Small classifier, build on a decision tree and a markov chain"""
    ## To-do
    # Prunning
    def __init__(self, varPeriod, varId, varNextPeriod, max_depth=None, min_samples_split=2):
        ## Public attributes
        self.clf = tree.DecisionTreeClassifier(max_features="sqrt", 
                                               max_depth=max_depth, 
                                               min_samples_split=min_samples_split)
        self.varPeriod = varPeriod               # Column name containing the periods
        self.varId = varId                       # Column name containing individuals ids
        self.varNextPeriod = varNextPeriod       # Column name containing the index for the line with the same individuals at next period (built by generic function indexDataFrame)
        ## Private attributes
        self.__oob_score = 0.5                   # quality of the classifier estimated by evaluator function (default to 0.5)
        self.__transitions = None                # Matrix with in each cell i, j number of individuals making a transition from state i to state j
        self.__states_index = {}                 #
        self.__states_prob = []                # probability of y in each state
        self.features_ = []                      # list of variable names used to learn the decision tree
        self.classes_ = []                       # exhaustive list of classes to predict 
        # (We have to store it because sampling into fit and val may cause the 
        # disappearance of a class in y_fit. 
        # It has more chance to happen when the len of y is low)
        self.__usable_idperiod = None            # dataFrame of two columns. vFor each varId, first usable varPeriod to learn more transitions
        

    
    ## fit function
    # -------------
    def fit(self, X, y, verbose=True, lines_for_tree=None):
        """Fit timeHorizonLearner
        Parameters
        ----------
        X: pandas.core.frame.DataFrame
            DataFrame with explicative variables
        y: pandas.core.series.Series
            Column to predict
        lines_for_tree: list, optional
            list of index labels to use to learn tree
            
        Returns
        -------
        self : object
            Returns self.
        """
        ## Store informations
        self.classes_ = sorted(y.unique().tolist())        
        self.__build_features_(X)    

        
        ## Find elt with next period, we learn treee only on them
        if lines_for_tree is None:
            lines_to_use = X[X[self.varNextPeriod].notnull()].index.tolist()
        else: 
            next_period = X[self.varNextPeriod].copy()
            next_period = next_period.loc[lines_for_tree]
            lines_to_use = next_period[next_period.notnull()].index.tolist()

        ## Build tree
        if verbose == "debug":
            print("timeHorizonLearner.fit: distribution of value in y: " + str(y.loc[lines_to_use].value_counts()))
        self.clf.fit(X.loc[lines_to_use][self.features_], y.loc[lines_to_use]) 
        
        # Once, tree is built, we set the list of leafs        
        self.__build_states_index(verbose)
        self.__build_states_prob(verbose) 

        ## built markov chain
        self.complete_transitions(X, verbose=verbose)
        
    ## Build or complete transitions matrix (markov chain part of the THL)
    # --------------------------------------------------------------------
    def complete_transitions(self, X, verbose=True):
        """Function to complete previously buiold transition matrix with new transitions 
        Parameters
        ----------
        X: pandas.core.frame.DataFrame
            DataFrame with explicative variables
        verbose: bool
            Should the algorithm log
            
        Returns
        -------
        self : object
            Returns self.        
        """
        ## If you have a same data set on an other period you can feed it to make transition proba better
        # Keep lines with same id as in learning and periods bigger or equal than previously seen periods
        if self.__usable_idperiod is not None:
            result = pd.merge(X[[self.varId, self.varPeriod]], self.__usable_idperiod, how="left", on=[self.varId])
            usable_lines = np.where(result[self.varPeriod] >= result["firstUsablePeriod"])[0] # [0] to handle specific shape of np.where results
        else:
            usable_lines = range(X.shape[0])
        # reduce list to those which have a next period
        # Here we are using iloc instead of loc so that we don't modify X, see here: https://github.com/pandas-dev/pandas/issues/11502
        next_period = X[self.varNextPeriod].copy()
        next_period = next_period.iloc[usable_lines]
        next_period = next_period[next_period.notnull()] # contain next period that are not null
        lines_to_use = next_period.index.tolist() # index of lines that have a next period that isn't null
        # perform matrix improvement
        if len(lines_to_use) > 0:
            # Compute starting and ending states
            start_states = self.__predict_tree(X.loc[lines_to_use]) 
            end_states = self.__predict_tree(X.loc[next_period])             
            # Add it to transition matrix
            self.__build_transitions(start_states, end_states, verbose)
        else:
            print("timeHorizonLearner.complete_transitions: Sorry there where no usable lines in what you fed me.")
        # To-do reevalute classifier???
        if verbose is True:
            print("timeHorizonLearner.complete_transitions: transition has: " + str(sum(sum(self.__transitions))) + " elements in it.")
        ## Update __usable_idperiod, 
        # it's a bit heavy in terms of computational time: to-do see how to optimize it        
        # We need to compute max of used before min of usable to handle IDs that doesn't have every period ex: 201601 201603 201604
        x_reduced = X.loc[lines_to_use][[self.varId, self.varPeriod, self.varNextPeriod]]    
        local_used_idperiod = group_max(x_reduced[self.varId].values, x_reduced[self.varPeriod].values, self.varId, "lastUsedPeriod")
        # merge it
        result = pd.merge(X.iloc[usable_lines][[self.varId, self.varPeriod, self.varNextPeriod]]  , 
                          local_used_idperiod, 
                          how="left", on=[self.varId])
        # select lines that weren't used
        result = result[result[self.varPeriod] > result["lastUsedPeriod"]]
        # Build local usable periods
        local_usable_idperiod = group_min(result[self.varId].values, result[self.varPeriod].values, varGroup=self.varId, varData="firstUsablePeriod")
        # Integrate local_usable into stored usable
        if self.__usable_idperiod is not None:
            local_usable_idperiod = local_usable_idperiod.append(self.__usable_idperiod, ignore_index=True)
            local_usable_idperiod = group_max(local_usable_idperiod[self.varId].values, local_usable_idperiod["firstUsablePeriod"].values, self.varId, "firstUsablePeriod")
        # Integrate it 
        self.__usable_idperiod = local_usable_idperiod
        del local_used_idperiod, result, local_usable_idperiod
        
    ## Evaluate classifier quality
    # ----------------------------
    def evaluator(self, X, y, verbose=True):
        """Function to evaluate the quality of previously build classifier"""
        ## eval classifieur
        # Identify lines with "next" period
        # This should be useless but we are never too safe
        lines_to_use = X[X[self.varNextPeriod].notnull()].index.tolist()
        # Predict result
        result_val = self.predict_proba(X.loc[lines_to_use], jumps=1, verbose=verbose)

        # evalute it by average AUC on all classes (if n_classes_ > 2)
        self.__oob_score = auc(y[X.loc[lines_to_use][self.varNextPeriod]], result_val, self.classes_)
        if self.__oob_score == 0.5:
            print("timeHorizonLearner.evaluator: WARNING: This the case where in next period every one has the same class, we can't evaluate ourself")
        if verbose is True:
            print("timeHorizonLearner.evaluator: The quality weight as been calculated to: " + str(self.__oob_score))
    
    ## Predict proba
    # --------------       
    def predict_proba(self, X_deploy, jumps=1, verbose=True):
        """Function to predict the classifier on a new data set"""
        # Idea: Si il manque une colone, utiliser uniquement les experts qui 
        # ne se servent pas de cette colonne pour les autres retourner la moyenne
        
        # get the final leaf id for each example 
        if verbose is True:
            print("timeHorizonLearner.predict_proba: I will predict the tree. It is " + str(time.time()))
        
        if isinstance(X_deploy, np.ndarray):
            if X_deploy.shape[1] == len(self.features_):
                states_deploy = self.clf.tree_.apply(X_deploy)
            else:
                raise ValueError("timeHorizonLearner.predict_proba: error you gave a numpy.ndarray but not with the same number of columns as in train. You can try to feed a pandas.core.frame.DataFrame.")
        if isinstance(X_deploy, pd.core.frame.DataFrame):
            states_deploy = self.__predict_tree(X_deploy)
            #clf.tree_.apply(np.array(X_deploy[self.features_], dtype='float32'))
        

        if verbose is True:
            print("timeHorizonLearner.predict_proba: I finished to predict the tree. It is " + str(time.time()))
        
        
        M_deploy = self.__build_states_matrix(states_deploy)
        # To-do si on a pas vu la periode en train, on peut dire a 
        #       user de completer en feedant t-1 et t
        
        # Predict
        # Take transition matrix, devided by row sum in order to have proba
        # multiply it jumps times (or 0 if jumps=0)
        # Multiply it by the vector of proba in each state
        transitions_prob = self.__get_transitions_prob(verbose)        
        if isinstance(jumps, int):
            predicted_prob = np.dot(M_deploy, 
                                    np.dot(np.linalg.matrix_power(transitions_prob,
                                                                  jumps),
                                           self.__states_prob))
        if isinstance(jumps, list):
            predicted_prob = np.zeros((len(M_deploy), len(self.classes_), len(jumps)))
            for i in range(len(jumps)):
                predicted_prob[:, :, i] = np.dot(M_deploy, 
                                                 np.dot(np.linalg.matrix_power(transitions_prob,
                                                                               jumps[i]),
                                                        self.__states_prob))
        ## Wrapp-up
        return predicted_prob
        

        
    
        
    # ------------------------------------------------------------------------
    ## private function functions
    # ------------------------------------------------------------------------ 
        
    ## Clean transtion matrix and compute transitions_prob
    # ----------------------------------------------------
    def __get_transitions_prob(self, verbose):
        min_n_per_transition = 1
        # Get transitions and round
        transitions_prob = self.__transitions.copy()
        transitions_prob = np.round(transitions_prob)
        # drop poorly estimated transitions
        if verbose is True:
            n_transitions = transitions_prob.shape[0] * transitions_prob.shape[1]
            n_poor_transitions = np.sum(transitions_prob < min_n_per_transition, axis=(0, 1)) - np.sum(transitions_prob == 0, axis=(0, 1))
            print("timeHorizonLearner.__get_transitions_prob: there are " + str(n_poor_transitions) + " poorly estimated transition (" + str(round(float(n_poor_transitions) / n_transitions * 100)) + "%), i delete them.")
            n_transitions = sum(sum(transitions_prob))
        transitions_prob[transitions_prob < min_n_per_transition] = 0.
        if verbose is True:
            print("timeHorizonLearner.__get_transitions_prob: they represented: " + str(n_transitions -  sum(sum(transitions_prob))) + " transtions (" + str(round(float(n_transitions -  sum(sum(transitions_prob))) / sum(sum(self.__transitions)) * 100)) + "%).")
        # Normalize to have probabilities (prob)
        transitions_prob = transitions_prob / transitions_prob.sum(axis=1, keepdims=True)
        
        # Set 0 for lines that didn't have any transitions
        # This case occures when: 
        # - all transitions add less than min_n_per_transition elt
        transitions_prob[np.isnan(transitions_prob)] = 0.
        # Wrapp-up
        return transitions_prob

    ## prediction function predict result of tree
    # -------------------------------------------
    def __predict_tree(self, X):
        # To-do control that all self.features_ are in X
        return  self.clf.tree_.apply(np.array(X[self.features_], dtype='float32')) 
    
    # identify in which state a line is
    # ----------------------------------
    def __build_states_matrix(self, states):
        ## From the state name build a matrix 
        # with 1 observation per line 
        # and as mean columns as there are states. 
        # 1 in the column where the observation
        # Convert states using a loop
        converted_states = np.array([self.__states_index[x] for x in states])
        
        # Fill matrix    
        M = np.zeros((states.size, len(self.__states_index)))    
        for i in self.__states_index.values():
            M[converted_states == i, i] = 1
        
        # Return
        return M

    ## Build transitions matrix from sart and end states
    # --------------------------------------------------
    def __build_transitions(self, start_states, end_states, verbose=True):
        ## From start_states and end_states
        # build a matrix with
        # number of unique states columns
        # number of unique states rows
        # In T[i, j] there is the probability to go from state uniqueStatesId[i] to uniqueStatesId[j]
        
        ## Initialization  
        n_states = len(self.__states_index)
        T = np.zeros((n_states, n_states))
        if verbose == "debug":
            print("Debug: it is " + str(time.time()) + " Identify this starting states: " + str(np.unique(start_states)))
            print("Debug: it is " + str(time.time()) + " Identify this ending states: " + str(np.unique(end_states)))
        
        ## Computation
        for i in range(len(start_states)):
            T[self.__states_index[start_states[i]], self.__states_index[end_states[i]]] += 1


        ## wrapp-up
        if self.__transitions is None:
            self.__transitions = T
        else:
            # T will have the same size as previous transition matrix
            if verbose is True:#== "debug":
                n_transition_bef = np.sum(self.__transitions, axis=(0, 1))
                
            self.__transitions = self.__transitions + T 
            
            if verbose is True:#== "debug":
                n_transition_aft = np.sum(self.__transitions, axis=(0, 1))
                print("timeHorizonLearner.__build_transitions: we added: " + str(n_transition_aft - n_transition_bef) + " transitions.")
        
    # identify list of terminal nodes in builded tree
    # -----------------------------------------------
    def __build_states_index(self, verbose=True):
        ## Description 
        ## Input
        # A tree classifier
        ## Output
        # a list of int (ID of leafs)
        # To-do if depth == 0 => ça plante
        # To-do: degorifier
        states = np.where(self.clf.tree_.children_left == -1)[0].tolist()  # Filter to get only leafs (nodes that haven't a left child)
        ## Build indexOfStates
        for i in range(len(states)):
            self.__states_index[states[i]] = i
        
        if verbose is True:
            print("timeHorizonLearner.__buildListInStates: there are " + str(len(states)) + " final leafs in the classifier.")
    # for each terminal node of the tree compute proba in it and store it 
    # -------------------------------------------------------------------
    def __build_states_prob(self, verbose=True):
        # For each State effectif per class
        effectif = self.clf.tree_.value[self.__states_index.keys(), 0, :] 
        
        # If not all classes from classes_ are present in classifier we 
        # have to add prob to 0 for this classes in states
        if len(self.classes_) != len(self.clf.classes_.tolist()):
            # Identify index of missing classes
            local_classes_ = self.clf.classes_.tolist()
            missing_classes_ = []    
            for i in range(len(self.classes_)):
                if i >= len(local_classes_) or self.classes_[i] != local_classes_[i]:
                    missing_classes_.append(i)
                    local_classes_.insert(i, self.classes_[i])
            if verbose == "debug":
                print("Debug: timeHorizonLearner.__build_states_prob: list of missing classes indexes: " + str(missing_classes_))
            
            # Add column with 0 at those indexes. 
            for i in missing_classes_:
                effectif = np.insert(effectif, i, values=0, axis=1)
                
        # Normalize result (in order to have proba)   
        self.__states_prob = (effectif.transpose() / np.sum(effectif, axis=1)).transpose()
        
    # Construct list of var to learn the tree
    # ----------------------------------------
    def __build_features_(self, X):
        ## Identify columns used to leran
        for var in X.columns:
            if not var in [self.varId, self.varPeriod, self.varNextPeriod] + ['idNextPeriod' + str(i) for i in range(100)]:
                self.features_.append(var)
        # sample features_
        self.features_ = [self.features_[i] for i in sorted(random.sample(xrange(len(self.features_)), int(math.sqrt(len(self.features_)))))]
        
        
    # ------------------------------------------------------------------------
    ## Getter functions
    # ------------------------------------------------------------------------
    def get_n_transitions(self):
        """Control function"""
        print("timeHorizonLearner.complete_transitions: transition has: " + str(np.sum(self.__transitions, axis=(0, 1))) + " elements in it.")
    def get_transitions(self):
        return self.__transitions
        
    ## get quality weight: a getter function for private attibute
    # -----------------------------------------------------------
    def get_oob_score(self):
        return self.__oob_score
    

    # ------------------------------------------------------------------------
    ## Compute results functions
    # ------------------------------------------------------------------------    
    
    ## Compute feature importance    
    # ---------------------------
    def feature_importances_(self):
        """ Get dictionnary of feature importance"""
        feature_imp = {}
        for i in range(len(self.features_)):
            feature_imp[self.features_[i]] = self.clf.feature_importances_[i]
        return feature_imp
        
    ## Compute infinite distribution
    # ------------------------------
    def infinite_distrib(self, verbose=True):
        """Get distribution after an infinite number of jumps"""
        # Trick python power 10 000 is infinite http://stackoverflow.com/questions/28259768/how-can-i-use-numpy-linalg-matrix-power-in-python-to-raise-a-matrix-to-a-large-p
        converged_transitions_prob = np.linalg.matrix_power(self.__get_transitions_prob(verbose), 10000) 
        if verbose == "debug":
            print("debug: proba matrix " + str(self.__get_transitions_prob(verbose)))
        return np.dot(self.__transitions.sum(axis=1) / sum(sum(self.__transitions)), np.dot(converged_transitions_prob, self.__states_prob))