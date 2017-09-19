# -*- coding: utf-8 -*-
"""
Created on Monday November 28th 2016
@author: Alexis BONDU and Emmanuel-Lin TOULEMONDE
"""

import os  
import gc
import shutil
import glob
import pandas as pd
import random
import time
import pickle       # NEW
import numpy as np
from genericFunctions import partition_estimators, samplingDisjoinedIdsReturnIndex, indexDataFrame
from timeHorizonLearner import timeHorizonLearner

import itertools
from sklearn.externals.joblib import Parallel, delayed
import warnings 
# For sklearn error log. 
import sys  
stdout = sys.stdout
reload(sys)  
sys.setdefaultencoding('utf8')
sys.stdout = stdout # To make sure that printing is still working


###################################################################################
#                     Preductive model with arbitrary time horizon
###################################################################################
class timeHorizonLearners:
    """Time horizon learners is bagging of classifier at variable echeances 
    based on a tree and markov chain"""
    # constructor
    # -----------
    def __init__(self, varCible=None, varPeriod=None, varId=None, varNextPeriod=None, n_estimators=100, 
                 max_depth=None, min_samples_split=2, class_weight=None, sampleSize=None):

        # Public attributes
        self.n_estimators = n_estimators	        # Number of weak learners
        self.max_depth = max_depth                 # parameters for tree: max_depth
        self.min_samples_split = min_samples_split # parameter for tree: min sample split
        self.varPeriod = varPeriod			   # Temporal variable name which encodes the differents time periods (string)
        self.varNextPeriod = varNextPeriod         # var containing the id for next period
        self.varId = varId				        # Name of the variable that represents the identifiers of costumers (string)
        self.varCible = varCible                  # Name of the target variable
        self.class_weight = class_weight		# The weigh of each class value for re-sampling the dataset (list of float values)
        self.sampleSize = sampleSize		# Size of each sample (int)
        self.classes_ = []                 # List of classes that where found in fit
        # Private attributes
        self.estimators = []        	# The collection of weak classifiers (list of objects)
        self.__features_ = []
    
    


    # ------------------------------------------------------------------------
    ## Public functions
    # ------------------------------------------------------------------------
    ## fit function
    # -------------
    def fit(self, X, y, rateVal=0.32, n_jobs=1, verbose=True, n_weaklearner=None):
        """Fit model
        
        Parameters
        ----------
         X: pandas.core.frame.DataFrame
            DataFrame with explicative variables
        y: pandas.core.series.Series
            Column to predict
        rateVal: float
            rate of lines to keep for evaluation of built estimators
        n_weaklearner: int, optional
            number of estimator to learn with this X, y. 
            So that one can provide new data to learn remaining estimators
        verbose: bool
            should the algorithm log (default to True)
        n_jobs: int
            number of core to use (1 for no parallelization)
        
        Returns
        -------
        self : object
            Returns self.        
        """
        ## Sanity check
        if sum(X[0:10].index == X[0:10][self.varNextPeriod]) == 10:
            # Small check before check onfull dataSet to pass it fast
            if sum(X.index == X[self.varNextPeriod]) == len(X):
                raise ValueError('Your variable varNextPeriod is not well defined, it is pointing to the same period.')
        ## Initialization
        if verbose is True:
            print "timeHorizonLearners.fit: I start by checking your parameters and initializing stuf. It is: " + str(time.time())
            
        # Identifiy the number of weak learner to build
        if n_weaklearner is None:
            n_weaklearner = self.n_estimators
        
        # Get the id of the first weak learner to build
        idFirstEstimator = len(self.estimators) # on retient l'id du 1er classifier ajoute dans cette fonction	

        # Store and control list of classes
        self.classes_ = [elt for elt in y.unique() if elt is not None and elt is not np.nan]
        self.classes_.sort()          # We sort in order to have the same order as prediction (from sklearn trees)
        # Split fit - val according to rateVal       
        I_fit, I_val = samplingDisjoinedIdsReturnIndex(X=X, rateVal=rateVal, varId=self.varId)

        # Control and update class_weight and SampleSize
        self.__build_class_weight(y.loc[I_fit])
        self.__buildAndControlSampleSize(y.loc[I_fit]) 
        #  RQ: chaque batch a un sample size different ? PB ??? 
        ## Perform fitting using Parallel from sklearn library
        if verbose is True:
            print "timeHorizonLearners.fit: I start to fit. It is: " + str(time.time())
        n_jobs, n_estimators, _ = partition_estimators(n_weaklearner, n_jobs)     
        total_n_estimators = sum(n_estimators)
        all_results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_parallel_build_estimators)(
                n_estimators[i],
                X.loc[I_fit],
                y.loc[I_fit],
                self.varId, self.varPeriod, self.varNextPeriod, 
                self.max_depth, self.min_samples_split, 
                total_n_estimators,
                self.class_weight,
                self.sampleSize,
                verbose=verbose)
            for i in range(n_jobs))
        # Reduce results
        self.estimators += list(itertools.chain.from_iterable(t for t in all_results))

        # Eval experts
        if rateVal > 0:
            if verbose is True:
                print "timeHorizonLearners.fit: I eval your experts. It is: " + str(time.time())
            self.evalutor(X.loc[I_val], y.loc[I_val], idFirstEstimator, n_weaklearner, n_jobs, verbose) 
        
    ## fit from a file
    # ----------------
    def fit_fromFile(self, file_path, separator, batch_size, jump=1, verbose=True, n_jobs=1, rateVal=0.32, features_=None):
        """Fit model from unloaded file
        To use when data is to lagre for your machine. It will load data by batch.
        Be carefull, file should have every lines for an varId successive. 
        One should also make sure that varIds are randomly placed in your file 
        in order to be sure that there is no bias in the dataset.
        
        Parameters
        ----------
        file_path: string
            path to data file
        separator: string
            column separator in file
        batch_size: int
            number of lines on which to learn a batch (approximatly)
        jump: int
            size of jump (in terms of periods)
        verbose: bool
            should the algorithm log (default to True)
        n_jobs: int
            number of core to use (1 for no parallelization)
        rateVal: float
            rate of lines to keep for evaluation of built estimators
        features_: list, optional
            which variable should be used to learn (in order not to read all columns)
        
        Returns
        -------
        self : object
            Returns self.        
        """
        # initialisation :
        self.varNextPeriod = 'idNextPeriod'+str(jump)
        
        # control parameter         
        if not isinstance(jump, int):
            raise ValueError('ERROR :: fit_fromFile :: the parameter \'jump\' must be an interger. ')

        #### first pass on the input datafile ####
        # ----------------------------------------
        if verbose is True:
            print "timeHorizonLearner.fit_fromFile: I start by reading a few coluns to build my batches"
        ## The data summary only contains the target variable and the IDs of the individuals
        data_summary = pd.read_csv(file_path, sep=separator, usecols=[self.varId, self.varCible])
        data_summary.sort(columns=self.varId)

        ## list of all possible class values 
        classes_ = data_summary[self.varCible].unique().tolist()
        
        ## total number of lines
        n_lines = data_summary.shape[0]

        # free memory
        data_summary.drop([self.varCible], axis=1) 
                            
	
        #### loop over the input file, reading by batch ####
        # --------------------------------------------------
        if verbose is True:
            print "timeHorizonLearner.fit_fromFile: Now i start to learner from batches"
        n_batches = n_lines / batch_size + 1
        batch_size = n_lines / n_batches + 1 # Update batch size so that they are about the same shape
        batch_index = []
        first_line = 0

        for i in range(n_batches):            
            last_line = first_line+batch_size-1 

            if last_line >= n_lines:              # index control
                last_line = n_lines-1

            instanceID = data_summary[self.varId][last_line] # valeur de l'ID client pour le dernier element du batch
            
            while last_line < n_lines and data_summary[self.varId][last_line] == instanceID:      # jump to change of individual
                last_line = last_line+1
	          
            batch_index.append([first_line, last_line-1])
            first_line = last_line
            
        if n_batches != len(batch_index):
            raise ValueError('DEBUG TODO : wrong batch management :-/ ')

        del data_summary

	
        ## LOOP over batches
        #  -----------------
        n_estimators = partition_estimators(self.n_estimators, n_batches)[1]      
        for i in range(n_batches):
            ## Loging
            if verbose is True:
                print "Compliting batch number %d of %d " % (i + 1, n_batches)
            
            ## Initialization
            IDs = batch_index[i]
            batch_is_good = True
            # load the part of the input file which corresponds to the batch
            df = pd.read_csv(file_path, sep=separator, skiprows=range(1, (IDs[0]+1)), nrows=int(IDs[1]-IDs[0]+1))
            
            # Load control
            local_classes_ = df[self.varCible].unique().tolist()
            for elt in classes_:
                if elt not in local_classes_:
                    warnings.warn("WARINING :: fit_fromFile :: ignored batch, because a class value is missing !! ")
                    batch_is_good = False
                    
            if batch_is_good:
                # 1 - DATA PREPARATION

                # 1.1 - index all df (line nexct periode)
                df = indexDataFrame(df, varId=self.varId, varPeriod=self.varPeriod, afterNPeriods=jump)
            
                # 1.2 - split X and y 
                X = df.drop([self.varCible], axis=1)                                   # explicative variables
                y = df[self.varCible]                                                    # target variable

                del df                                                              # memory saving
                gc.collect()                                                        # garbadge collector 

                # 1.3 - var selection by param
                if features_ is not None:
                    # control that X includes all variable to keep
                    features_ = features_ + [self.varNextPeriod, self.varId, self.varPeriod]
                    features_ = list(set(features_))
                    for col in features_:
                        if col not in X.columns:
                            raise ValueError('ERROR :: fit_fromFile :: some variable to keep are not in the dataframe X.')
                        
                    X = X[features_] 

                # 1.4 Drop columns that will cause issues
                # Forbid it to drop id,s, varNextPeriod, period....
                dropable_features_ = [column for column in X.columns if column not in [self.varId, self.varPeriod, self.varCible] + ['idNextPeriod'+str(jump)]]
                n_col_bef = X.shape[1]
                X = X.drop(X[dropable_features_].columns[X[dropable_features_].dtypes == 'object'], axis=1)
                
                # Warning if some variables are droped
                if X.shape[1] != n_col_bef:
                    warnings.warn("WARRNING :: fit_fromFile :: Some categorical variables are droped from the X dataset !! You should recoded it before :) ")
                 
                # 2 - Fit of weak learners 
                self.fit(X, y, n_jobs=n_jobs, rateVal=rateVal, verbose=verbose, n_weaklearner=n_estimators[i])
                
                # 3 - Finish
                gc.collect() # Clean memory
                                   
    # ------------------------------------------------------------------------
    ## Complete transition matrix
    # ------------------------------------------------------------------------
    def complete_transitions(self, X, n_jobs=1, verbose=True):
        """Add transtiotions to markov chains
        Usefull when you have observations without there targets but X mooves can still be used
        """
        ## Prepare parallelisation
        n_jobs, n_estimators, starts = partition_estimators(self.n_estimators, n_jobs)     
        total_n_estimators = sum(n_estimators)
        
        ## Make computation
        all_results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_parallel_complete_transitions)(
                self.estimators[starts[i]:starts[i+1]],
                X,
                total_n_estimators,
                verbose=verbose)
            for i in range(n_jobs))
        
        ## De-serialize result and put it back where it belong
        self.estimators = [] # Reset it to empty in order to truly change the classifiers
        self.estimators += list(itertools.chain.from_iterable(t for t in all_results))        

            
    ## Evaluator - compute oob_score
    # ------------------------------
    def evalutor(self, X, y, idFirstEstimator, n_weaklearner, n_jobs=1, verbose=True):
        """Compute oob_score given a set of X and y"""
        # Drop those that don't have a next period
        indexWithNextperiod = X[X[self.varNextPeriod].notnull()].index.tolist()
        # Update sample size
        sample_size_store = self.sampleSize
        self.sampleSize = None
        self.__buildAndControlSampleSize(y.loc[indexWithNextperiod])
        # Build a sample to rebalabnce evaluation to have it the same way as in fit
        lines_to_use = _build_sample(y.loc[indexWithNextperiod], self.sampleSize, self.class_weight)
        # Drop those that haven't a target at nextPeriod
        y_truth = y.loc[X.loc[lines_to_use, self.varNextPeriod]]
        
        # since list can't be index by np.array, set it as np.arrray then get list
        lines_to_use = np.array(lines_to_use)[np.where(y_truth.notnull())[0]].tolist()

        # Split jobs
        n_jobs, n_estimators, starts = partition_estimators(n_weaklearner, n_jobs)     
        total_n_estimators = sum(n_estimators)
        all_results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_parallel_eval)(
                self.estimators[(idFirstEstimator + starts[i]):(idFirstEstimator + starts[i+1])],
                X.loc[lines_to_use],
                y, # feed all y in order to be able to find nextPeriod y for eval
                total_n_estimators,
                verbose=verbose)
            for i in range(n_jobs))       

        # Reduce results
        estimators_evaluated = []
        estimators_evaluated += list(itertools.chain.from_iterable(t for t in all_results))

        # Udpate list of classifiers
        self.estimators[idFirstEstimator:(idFirstEstimator+n_weaklearner)] = estimators_evaluated
        
        ## Wrapp-up
        # reset sampleSize
        self.sampleSize = sample_size_store

     
    ## predict p(y|X) 
    # ---------------
        
    def predict_proba(self, X_deploy, jumps=1, n_jobs=1, verbose=True):
        """Predicting probability of each class at various horizon for new elments"""       
        # Input:
        # X_deploy:  matrix with the same variables as X used for fit
        # jumps an int or a list of int: number of jumps you want to perform
        ## Sanity check
        print "Due to a weird parallelization, one don't need to paralellize predict_proba with n_jobs. I set it to 1."
        n_jobs = 1
        # It some times stop working (don't do anything, but don't finish either) when
        # n_jobs > 1.
        ## Initialization:
        if not self.__is_oob_score_set():
            print "timeHorizonLearners.predict_proba: Warning: oob_score is 0.5 it might not have been computed, use the evaluator function to compute it!"
        
        ## Prepare parallelisation
        n_jobs, n_estimator, starts = partition_estimators(len(self.estimators), n_jobs)     
        total_n_estimators = sum(n_estimator)

        ## Make computation
        all_results = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(_parallel_perdict_proba)(
                self.estimators[starts[i]:starts[i+1]],
                X_deploy,
                len(self.classes_), 
                jumps,
                total_n_estimators,
                verbose=verbose)
            for i in range(n_jobs))

        ## Reduce
        result = all_results[0]				# reshape the outcome predictions
        for i in range(1, len(all_results)):
            result = result + all_results[i]
        

        sum_oob_scores = self.__get_sum_oob_scores()


        result /= sum_oob_scores
        

        
        return result

    ## predict the class values
    # ------------------------------------------------------------------------

    def predict(self):
        """Predict time horizon learner"""
        # To-do
        print "Pas implémenté"

        

    # ------------------------------------------------------------------------
    ## private function functions
    # ------------------------------------------------------------------------ 
    ## Construct features list
    # ------------------------
    def __build_features_(self):
        for estimator in self.estimators:
            self.__features_ = self.__features_ + estimator.features_
        self.__features_ = list(set(self.__features_))
    
    ## set parameters for sampling the rows of the dataset 
    # ----------------------------------------------------
    def __buildAndControlSampleSize(self, y):
        logWarnings = True
        ## If sampleSize is not defined.
        if self.sampleSize == None or self.sampleSize > len(y):
            self.sampleSize = len(y)
            logWarnings = False
        
        ## If The less reprsented class in class_weight doesn't even have 
        # 1 element in a sample, we have to make it bigger
        if self.sampleSize * min(self.class_weight.itervalues()) <= 1:
            self.sampleSize = int(1. / min(self.class_weight.itervalues()))
            
        ## If sampleSize is two big considering class_weight
        valueCounts = pd.value_counts(y)
        for elt in self.class_weight:
            # For each elt in class_weight, we control that we have more element
            # in y of class elt than sampleSize * class_weight[elt]
            if elt in valueCounts:
                tempSampleSize = int(float(valueCounts[elt]) / self.class_weight[elt])
                if tempSampleSize < self.sampleSize:
                    self.sampleSize = tempSampleSize
                    if logWarnings:
                        print "Warning: timeHorizonLearners.__buildAndControlSampleSize I had to lower sampleSize to: " + str(self.sampleSize)
            else:
                print "timeHorizonLearners.__buildAndControlSampleSize: WARNING: class" + str(elt) + "is present in class_weight but is missing in this y (fit or val)." 
                # Warning: if this exception was raised, the real sampleSize will be smaller than self.sampleSizes
    
    ## set parameters for re-balancing the dataset by class value
    # -----------------------------------------------------------
    def __build_class_weight(self, y):
        # Si pas defini, reprendre la distrib de y
        if self.class_weight is None:
            self.class_weight = pd.value_counts(y).to_dict()
        
        # Type control
        if not isinstance(self.class_weight, dict):
            raise Exception("timeHorizonLearners.__build_class_weight: Error class_weight should be a dictionnarys")
        # To-do important! Control that we have two periods for enough individuals
        
        # Check that every class are indeed present
        for elt in y.unique():
            if elt is not None and elt is not np.nan:
                if not elt in self.class_weight:
                    raise Exception("timeHorizonLearners.__build_class_weight: Error class  " +
                                    str(elt) +
                                    "  is missing from class_weight or you can leave it to None")
                if self.class_weight[elt] == 0:
                    raise Exception("timeHorizonLearners.__build_class_weight: Error class_weight must not contain 0 ")
        
                 
        # Normaliser par1
        total = sum(self.class_weight.itervalues(), 0.0) + 1
        self.class_weight = {k: v / total for k, v in self.class_weight.iteritems()}    
    ## Control that oob_score weight is set
    # -------------------------------------
    def __is_oob_score_set(self):
        is_set = False        
        for estimator in self.estimators:
            if estimator.get_oob_score() != 0.5:
                is_set = True
        return is_set
        
    ## get sum of oob_score (for normalization)
    # -----------------------------------------
    def __get_sum_oob_scores(self):
        sum_oob_scores = 0.
        for estimator in self.estimators:
            sum_oob_scores += estimator.get_oob_score()
        return sum_oob_scores
    # ------------------------------------------------------------------------
    ## Getter functions
    # ------------------------------------------------------------------------        
    def get_features_(self):
        """Get list of used vars"""
        if len(self.__features_ ) == 0:
            self.__build_features_()
            return self.__features_

    # ------------------------------------------------------------------------
    ## Compute results functions
    # ------------------------------------------------------------------------    
    ## Compute feature importance    
    # ---------------------------    
    def feature_importances_(self):
        """get importance per variables"""
        ## Sanity check
        if len(self.estimators) == 0:
            raise ValueError("timeHorizonLearner.feature_importances_: cannot get feature_importances_ if it hasn't been fit")
            
        ## Some lamba function
        dict_sum = lambda x, y: {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}
        dict_prod = lambda x, y: {k: x.get(k, 0) * y for k in set(x)}
        
        # Computation with the rest
        sum_oob_scores = self.__get_sum_oob_scores()
        for i in range(len(self.estimators)):
            if i == 0: # special case to initial it correctly
                feature_imp = dict_prod(self.estimators[0].feature_importances_(),
                                        self.estimators[0].get_oob_score())
            else:
                feature_imp = dict_sum(feature_imp,
                                       dict_prod(self.estimators[i].feature_importances_(),
                                                 self.estimators[i].get_oob_score()))
        # Normalize by sum of quality weight
        feature_imp = dict_prod(feature_imp, 1. / sum_oob_scores)
        
        return feature_imp
     
    # Util function: Get infitinit distribution function 
    # ---------------------------------------------------
    def infinite_distrib(self, verbose=True):
        """Converged distribution after infinit number of jumps"""
        ## Initialization
        sum_oob_scores = self.__get_sum_oob_scores()
        infinit_matrix = np.zeros(len(self.classes_))
        
        ## Computation
        for i in range(len(self.estimators)):
            infinit_matrix += self.estimators[i].get_oob_score() * self.estimators[i].infinite_distrib(verbose)

        infinit_matrix /= sum_oob_scores

        return infinit_matrix
    # ------------------------------------------------------------------------
    # saving functions
    # ------------------------------------------------------------------------
    ## serialize
    # -----------
    def serialize(self, path, name):
        # repertoire ou est stocke le model
        folder_path = path+'/'+name      	                # construction du path du repertoire
        
        if os.path.isdir(folder_path) == True:                   # si le repertoire existe
            shutil.rmtree(folder_path)                          # on le supprime
        os.makedirs(folder_path)                                # On cree le répertoire dans lous les cas

        # serialisation de chaque model 
        for i in range(len(self.estimators)):
            fhand = open(folder_path + "/model_" + str(i) + ".obj", 'w')
            pickle.dump(self.estimators[i], fhand, pickle.HIGHEST_PROTOCOL)
            fhand.close()
        # autres parametres	
        param = [self.n_estimators, self.max_depth, self.varId, self.varPeriod, self.varNextPeriod, self.class_weight, self.sampleSize, self.classes_]
        
        fhand = open(folder_path + "/param.obj", 'w')
        pickle.dump(param, fhand, pickle.HIGHEST_PROTOCOL)
        fhand.close()
	

    ## deserialisation
    # ----------------
    
    def deserialize(self, path, name):
        # repertoire ou est stocke le model
        folder_path = path+'/'+name      	                		# construction du path du repertoire

        if os.path.isdir(folder_path) is True:                   		# si le repertoire existe
            self.estimators = []					# on vide la liste de weak learner 
            filesPath = glob.glob(folder_path+'/model*')			# on recupere le liste des des 
            
            # les modeles 		
            for path in filesPath:
                pkl_model = open(path, 'rb')
                self.estimators.append(pickle.load(pkl_model))	# on deserialise
                pkl_model.close()
            
            # les parametres
            pkl_param = open(folder_path+'/param.obj', 'rb')
            params = pickle.load(pkl_param)
            if len(params) >= 8:
                self.n_estimators = params[0]
                self.max_depth = params[1]
                self.varId = params[2]
                self.varPeriod = params[3]
                self.varNextPeriod = params[4]
                self.class_weight = params[5]
                self.sampleSize = params[6]
                self.classes_ = params[7]
            else:
                raise ValueError('Warrning : input corrupted ! (not engouth parameters)')
        else:
            raise ValueError('Warrning : Wrong path ! The input model does not exist !')
##############################################################################
#           Multi-threading functions used by the parallel from sklearn
##############################################################################
# Fit 
# ---   
    
def _parallel_build_estimators(n_estimators, X, y, 
                               varId, varPeriod, varNextPeriod, 
                               max_depth, min_samples_split, total_n_estimators, 
                               class_weight, sampleSize, verbose):
    """Private function used to build a batch of estimators within a job."""
    ## Log initialization
    if verbose is True:
        print "timeHorizonLearner.fit distribution of value in each tree: " + str({elt: int(class_weight.get(elt, 0) * sampleSize) for elt in set(class_weight)})
    # Build estimators
    estimators = []
    for i in range(n_estimators):
        if verbose is True:
            print("Building estimator %d of %d for this parallel run "
                  "(total %d)..." % (i + 1, n_estimators, total_n_estimators))
        
        ## Make sampling for this round
        lines_to_use = _build_sample(y, sampleSize, class_weight)
        # Find all lines to cnsider
        store_id = X[varId]
        id_to_use = list(set(store_id.loc[lines_to_use]))
        lines_to_use_allperiods = store_id[store_id.isin(id_to_use)].index.tolist()
        
        ## Build estimator
        estimator = timeHorizonLearner(varPeriod=varPeriod, varId=varId, 
                                       varNextPeriod=varNextPeriod, 
                                       max_depth=max_depth, min_samples_split=min_samples_split)
        estimator.fit(X.loc[lines_to_use_allperiods], 
                      y.loc[lines_to_use_allperiods], verbose=verbose, lines_for_tree=lines_to_use)        
        estimators.append(estimator)

    return estimators

# Complete matrix
# ---------------
def _parallel_complete_transitions(estimators, X, total_n_estimators, verbose=True):
    n_estimators = len(estimators)
    for i in range(n_estimators):
        if verbose is True:
            print("Compliting estimator %d of %d for this parallel run "
                  "(total %d)..." % (i + 1, n_estimators, total_n_estimators))
        estimators[i].complete_transitions(X, verbose=verbose)
        
    return estimators
        
# Predict P(y|X)
# --------------
def _parallel_perdict_proba(estimators, X_deploy, nbrClasses, jumps, total_n_estimators, verbose=True):
    ## Initialization
    # Get params:
    n_estimators = len(estimators)
    
    # Set object
    if isinstance(jumps, int):
        result = np.zeros((X_deploy.shape[0], nbrClasses))
    if isinstance(jumps, list):
        result = np.zeros((X_deploy.shape[0], nbrClasses, len(jumps)))
    
    # Data preparation
    if isinstance(X_deploy, pd.core.frame.DataFrame):
        wasPandas = True
        # Drop unmanted columns
        n_col_bef = X_deploy.shape[1]
        X_deploy = X_deploy.drop(X_deploy.columns[X_deploy.dtypes == 'object'], axis=1)
        # Warning if some variables are droped
        if X_deploy.shape[1] != n_col_bef:
            warnings.warn("WARRNING :: predict_proba :: Some categorical variables are droped from the X dataset !! You should recoded it before")
        # Transform it to numpy array in float 32
        features_ = X_deploy.columns
        X_deploy = np.array(X_deploy, dtype='float32')
    else:
        wasPandas = False
         
    ## Looping over classifiers    
    for i in range(n_estimators):
        if verbose is True:
            print("Compliting estimator %d of %d for this parallel run "
                  "(total %d)..." % (i + 1, n_estimators, total_n_estimators))
        
        if wasPandas:
            # If it was a pandas dataFrame we ensure that we feed the write amount of variables for each classifiers. 
            # It is done inline in order to save ram
            result += estimators[i].get_oob_score() * estimators[i].predict_proba(X_deploy[:, np.where([x in estimators[i].features_ for x in features_])[0]], jumps=jumps, verbose=verbose)
            
        else:
            # No control on columns is performed, it might fail
            result += estimators[i].get_oob_score() * estimators[i].predict_proba(X_deploy, jumps=jumps, verbose=verbose)
        
    return result


    
    
## _parallel_eval: evaluate classifiers (to set oob_score)
# ------------------------------------------------------------
def _parallel_eval(estimators, X, y, total_n_estimators, verbose=True):
    n_estimators = len(estimators)
    
    for i in range(n_estimators):
        if verbose is True:
            print("Evaluating estimator %d of %d for this parallel run "
                  "(total %d)..." % (i + 1, n_estimators, total_n_estimators))
        estimators[i].evaluator(X, y, verbose=verbose)
    
    return estimators
    
## _build_sample sample indexes according to class weight
#---------------------------------------------------------
                
def _build_sample(y, sampleSize, class_weight):
    sampled_lines = []
    for elt in class_weight:
        elt_lines = y[y == elt].index.tolist()
        if len(elt_lines) != 0:
            sampled_lines += random.sample(elt_lines, int(sampleSize * class_weight[elt]))
        else:
            print "_build_sample: Warning: class" + str(elt) + "is present is class_weight but is missing in this y."
    return sampled_lines
            
## all list
# ---------
__all__ = ["timeHorizonLearner"]