# -*- coding: utf-8 -*-
"""
Created on Monday November 28th 2016
@author: Emmanuel-Lin TOULEMONDE and Alexis BONDU
"""

import numpy as np
import pandas as pd
import time
import progressbar
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
cimport cython
np.import_array() # C'est Cédric qui met ça

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.int
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int_t DTYPE_t


from libc.math cimport sqrt
from libc.math cimport exp
from libc.math cimport log
from numpy import inf

import time
import matplotlib.pyplot as plt
##############################################################################
########################## Define class myLines ##############################
##############################################################################
# Class lines
# Is built to reprensent a line in n dimension.
# Some function are built in to project on this straight lines
cdef class myLines:
    cpdef np.ndarray a, b, ab
    cdef float norm_ab
    ## __init__
    # -------------------------    
    def __init__(self, np.ndarray[double, ndim=1] a, np.ndarray[double, ndim=1] b):
        self.a = a # Point n 1 to define the line
        self.b = b # Point n 2 to define the line
        # Next built in some stuff to save time 
        self.ab = self.b - self.a
        self.norm_ab = sqrt(cython_dot(self.ab, self.ab))
        if self.norm_ab == 0:
            raise ValueError("a and b should be two different points!")
    ## distFromOrigin_forAPoint
    # -------------------------
    @cython.boundscheck(False) 
    @cython.wraparound(False) 
    @cython.cdivision(True) 
    cpdef float distFromOrigin_forAPoint(self, np.ndarray[double, ndim=1] point):
        """Compute closest point on line defined by a, b"""
        # See how to make a scalar product: https://www.cmath.fr/1ere/produitscalaire/cours.php
        return cython_dot(self.ab,  point - self.a) / self.norm_ab

    ## distFromOrigin
    # ----------------
    @cython.boundscheck(False) 
    @cython.wraparound(False) 
    @cython.cdivision(True)     
    cpdef np.ndarray[double, ndim=1] distFromOrigin(self, np.ndarray[double, ndim=2] points_array):
        """Compute dist from origin of an array of point closest point on line"""
        cdef Py_ssize_t i, j
        cdef unsigned int n_points = (<object> points_array).shape[0]
        cdef np.ndarray[double, ndim=1] result = np.empty(n_points, dtype="double")
        
        for i in range(n_points):
            result[i] = self.distFromOrigin_forAPoint(points_array[i,:])
        return result


##############################################################################
########################## Define class generator ############################
##############################################################################           
cdef class generator(myLines):
    ## Attributes
    cdef public myLines line
    cdef public int n_dims, n_uniDir
    cdef public float changeRateX, mixRate, rateOfOne, initialGraphSize
    cdef np.ndarray Sigma
    
    ## Computation vars
    cdef np.ndarray results
    
    ## __init__
    # -----------
    def __init__(self, n_dims, changeRateX=1, mixRate=0, rateOfOne=0.5, initialGraphSize=1, rateOfUniDir = 0.5):
        if mixRate <= 0  or rateOfOne <= 0 or mixRate >= 1  or rateOfOne >= 1:
            raise ValueError("mixRate and rateOfOne should be >0 and <1")
        self.line = myLines(np.zeros(n_dims), np.ones(n_dims))          # Line on which we project to compute y
        self.n_dims = n_dims                                            # Dimension of X space
        self.Sigma = np.diag(np.full(n_dims, changeRateX))              # Matrix of variance covariance
        self.mixRate = mixRate                                  
            
        self.initialGraphSize = initialGraphSize                        # width of initial space in every dimension
        self.rateOfOne = rateOfOne                                      # rate of ones in y
        if rateOfUniDir <= 1 and rateOfUniDir >= 0: 
            self.n_uniDir = int(rateOfUniDir * n_dims)
        else: 
            raise ValueError("rateOfUniDir should be between 0 and 1.")
        # To-do: add a seed
    
    ## __genY
    # --------
    @cython.boundscheck(False) 
    @cython.wraparound(False) 
    @cython.cdivision(True) 
    cdef np.ndarray[np.int32_t, ndim=1] __genY(self, np.ndarray[double, ndim=2] X):
        # Computation vars
        cdef np.ndarray[double, ndim=1] dists       
        cdef np.ndarray[np.int32_t, ndim=1] y 
        cdef float bias, l_BefB, l_AftB                 # Sigmoid param
        cdef Py_ssize_t i        
        cdef int n_points
        cdef double prob
        # Control vars
        cdef int oneAftB = 0, oneBefB = 0
        cdef int nBefB = 0, nAftB = 0
        
        # Compute dists
        dists = self.line.distFromOrigin(X)
        n_points = len(dists)

        # In order to conserve rateOfOne computed by bias, it is necessary to have different slope 
        # if you have negative or positive bias. (because, value of dists are limited 
        # by minDist and maxDist)
        l_BefB, l_AftB, bias = paramCalculator(dists, self.rateOfOne, self.mixRate, 4)
        

        ## Make sampling
        y = np.empty(n_points, dtype=np.int32)  
        for i in range(n_points):
            # Put dist in sigmoid to have a proba
            if dists[i] > bias:
                prob = 1. / (1 + exp(-l_AftB * (dists[i] - bias)))
            else:
                prob = 1. / (1 + exp(-l_BefB * (dists[i] - bias)))
            # From proba generate random value (1 or 0)
            y[i] = choose_one([1-prob, prob], 2)
#            # Control part            
#            if dists[i] > bias:
#                oneAftB = oneAftB + y[i] 
#                nAftB = nAftB + 1
#            else:
#                oneBefB = oneBefB + y[i]
#                nBefB = nBefB + 1
#             
#        # Print to control that generated values are the expected ones       
#        print "#### Sigmoid parameters"
#        print "bias: " + str(bias)
#        print "l_AftB: " + str(l_AftB)
#        print "l_BefB: " + str(l_BefB)
#        
#        
#        print "### Errors"
#        print "rate of 1 - expected: " + str(float(oneAftB + oneBefB) / float(n_points) - self.rateOfOne)
#        print "rate of 1 over bias - expected: " + str(float(oneAftB) / nAftB  - (1 - self.mixRate))
#        print "rate of 1 before bias - expected: " + str(float(oneBefB) / nBefB  - (self.mixRate))
        return y
    
    ## initialPoints
    # ---------------    
    @cython.boundscheck(False) 
    @cython.wraparound(False) 
    @cython.cdivision(True)     
    cpdef np.ndarray[double, ndim=2] initialPoints(self, int n_points):
        """Draw an initial distribution"""
        cdef np.ndarray[double, ndim=2] results
        results = np.random.uniform(-self.initialGraphSize / 2., self.initialGraphSize / 2., size=(n_points, self.n_dims))
        return results
    
    ## mooveFunction
    # ---------------
    @cython.boundscheck(False) 
    @cython.wraparound(False) 
    @cython.cdivision(True) 
    cpdef np.ndarray[double, ndim=2] mooveFunction(self, np.ndarray[double, ndim=2] X):
        """Moove a point close to itself"""
        # def        
        cdef Py_ssize_t col
        # Computation
        X = multivariate_normal(X, self.Sigma, self.n_uniDir)
        # Handle space limits via modulo:
        for col in range(self.n_uniDir):
            X[X[:,col] > self.initialGraphSize / 2., col] = X[X[:,col] > self.initialGraphSize / 2., col] % (self.initialGraphSize / 2.) - (self.initialGraphSize / 2.)
        return X
    
    ## generate
    # ----------
    @cython.boundscheck(False) 
    @cython.wraparound(False) 
    @cython.cdivision(True) 
    cpdef generate(self, int n_points, int T, str form="dataFrame"):
        """Main function: data generator"""
        # n_points: the number of lines
        # T the number of time periods (aka mooves) you want to perform
        cdef np.ndarray[double, ndim=3] X = np.empty((n_points, self.n_dims, T))
        cdef np.ndarray[np.int32_t, ndim=2] y = np.empty((n_points, T), dtype=np.int32)  
        cdef Py_ssize_t i

        bar = progressbar.ProgressBar()
        for i in bar(range(T)):
            if i == 0:
                X[:, :, 0] = self.initialPoints(n_points)
            else:
                X[:, :, i] = self.mooveFunction(X[:, :, i-1])
                # Plot of moovement
#                plt.hist(X[:, 0, i] - X[:, 0, i-1], bins = 20, label = "Var 0 ", normed  = True)
#                plt.hist(X[:, self.n_dims - 1, i] - X[:, self.n_dims - 1 , i-1], bins = 20, label = "Var n_dims", normed  = True)
#                plt.legend(loc='upper right')
#                plt.show()
#                # Plot of positions
#                plt.hist(X[:, 0, i], bins = 20, label = "X 0 ", normed  = True)
#                plt.hist(X[:, self.n_dims - 1, i], bins = 20, label = "X n_dims", normed  = True)
#                plt.legend(loc='upper right')
#                plt.show() 
            y[:, i] = self.__genY(X[:, :, i])


        
        if form == "nparray":
            return X, y
        if form == "dataFrame":
            finalDF = pd.DataFrame()
            for i in range(T):
                df = pd.DataFrame(X[:, :, i])
                df['id'] = df.index
                df['period'] = i
                df['target'] = y[:, i]
                finalDF = finalDF.append(df)
            return finalDF 

##############################################################################
########################### Outside functions ################################
##############################################################################
## multivariate_normal
# ---------------------
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.cdivision(True)             
cpdef np.ndarray[double, ndim=2] multivariate_normal(np.ndarray[double, ndim=2] mean, np.ndarray[double, ndim=2] Sigma, int n_uniDir):
    cdef int n_dims =  (<object> mean).shape[1]
    cdef int n_points =  (<object> mean).shape[0]
    cdef np.ndarray[double, ndim=2] cholSigma = np.transpose(np.linalg.cholesky(Sigma))
    cdef Py_ssize_t i
    cdef np.ndarray[double, ndim=2] results
    # Generate randomness
    results = np.random.standard_normal(size = (n_points, n_dims))
    
    # For eazch point, add center (the point) and randomness following a N(0, Sigma)
    for i in range(n_points):
        results[i,:] = mean[i,:] + speciaAbs(np.dot(cholSigma, results[i,:]), n_uniDir)
        
    ## Control step
    #plt.hist(results[:, 0] - mean[:, 0], bins = 20, label = "Var 0 ", normed  = True)
    #plt.hist(results[:, n_dims - 1] - mean[:, n_dims - 1], bins = 20, label = "Var n_dims", normed  = True)
    #plt.legend(loc='upper right')
    #plt.show()
    
    return results
    
## Random choice functions
# ------------------------
cdef extern from "stdlib.h":
    double drand48()
    void srand48(long int seedval)

@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.cdivision(True)    
cdef Py_ssize_t choose_one(double weights[2], Py_ssize_t length):
    cdef Py_ssize_t idx, i
    cdef double cs
    cdef double random
    random = drand48()
    cs = 0.0
    i = 0
    while cs < random and i < length:
        cs += weights[i]
        i += 1
    return i - 1

## specialAbs
# -----------
# To take absolute values on n_abs first values of the vector. 
# Will be applied on mooves in order to have unidirectionnal mooves... 
#@cython.boundscheck(False) 
#@cython.wraparound(False) 
#@cython.cdivision(True)     
cpdef np.ndarray[double, ndim=1] speciaAbs(np.ndarray[double, ndim=1] x, int n_abs):
    x[:n_abs] = np.abs(x[:n_abs])
    return x

## Computing sigmoid paramters
# -----------------------------
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.cdivision(True)      
cpdef paramCalculator(np.ndarray[double, ndim=1] points, double rateOfOne, double mixRate, int maxIter):
    cdef double optL1 = 0., optL2 = 0.
    cdef np.ndarray[double, ndim=1] pointsBefB
    cdef np.ndarray[double, ndim=1] pointsAftB
    cdef double step
    cdef int n_p_bef_b, n_points
    cdef np.ndarray[double, ndim=1]  points_tmp
    cdef double maxVal, minVal, width, b
    cdef double rangeL1[2]
    cdef double rangeL2[2]
    cdef int it
    cdef unsigned int n_points_bef, n_points_aft
    # Compute n_points
    n_points = len(points)
    # Compute bias
    points_tmp = np.sort(points)
    
    maxVal = points[n_points - 1]
    minVal = points[0]
    width = maxVal - minVal     
    # Compute bias
    # bias is computed so that E(p>b) * (1 - mixRate) + E(p< b) * mixRate = rateOfOne
    n_p_bef_b = n_points - int((rateOfOne - mixRate) / ( 1 - 2 * mixRate) * n_points)
    b = points_tmp[n_p_bef_b]
    
    # Split points
    pointsBefB = points_tmp[:n_p_bef_b] # Points before b
    pointsAftB = points_tmp[n_p_bef_b:] # points after b
    n_points_bef = len(pointsBefB)
    n_points_aft = len(pointsAftB)



    # Control:
    if (n_points_bef== 0 or n_points_aft == 0):
        print "len(pointsBefB): " + str(n_points_bef)
        print "len(pointsAftB): " + str(n_points_aft)
        print "bias: " + str(b)
        raise ValueError("An error occured while computing bias")
        
    # Compute lambdas
    rangeL1 = [0.1, 100]
    rangeL2 = [0.1, 100]
    for it in range(maxIter):
        step = pow(10, -it)
        optL1 = optimByRangeLambda(pointsBefB, n_points_bef,
                                   rangeL1[0], rangeL1[1], step,
                                   rateOfOne, mixRate, b)
        
        optL2 = optimByRangeLambda(pointsAftB, n_points_aft, 
                                   rangeL2[0], rangeL2[1], step,
                                   rateOfOne, 1 - mixRate, b)
        #print "With range: " + str(rangeL1) + " i found optL1: " + str(optL1)
        #print "With range: " + str(rangeL2) + " i found optL2: " + str(optL2)
        rangeL1 = [max(0.1, optL1 - step), optL1 + step]
        rangeL2 = [max(0.1, optL2 - step), optL2 + step]

    return optL1, optL2, b


## Optimize lambda by step
# ------------------------
# Search by range of optimal lambda given a bunch of points
@cython.cdivision(True) # disable check for 0 division because we are sur of dising by more than 0
cdef double optimByRangeLambda(np.ndarray[double, ndim=1] points, int n_points,
                               double dl, double al, double step,
                               double rateOfOne, double mixRate, double b):
    # def
    cdef double minDelta = float('inf')
    cdef double point, l, delta, optL
    cdef Py_ssize_t i
    # computation
    l = dl
    while l < al:
        l = l + step
        delta = 0.
        for i in range(n_points): # Looping on index is 3x time faster that looping on points
            delta = delta + 1. / ( 1. + exp( - l * ( points[i] - b))) 
        delta = delta / n_points
        delta = (mixRate - delta) * (mixRate - delta)
        
        
        if delta < minDelta:
            minDelta = delta
            optL = l
                
    return optL

cdef double cython_dot(double[:] a, double[:] b):
    cdef int N = a.shape[0]
    cdef Py_ssize_t i
    cdef double result = 0.
    for i in range(N):
        result = result + a[i] * b[i]
    return result
    