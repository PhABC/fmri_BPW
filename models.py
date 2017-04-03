import numpy as np
import scipy as sp

from fmri_BPW.functions import *
from lmfit              import minimize, Parameters, report_fit


class model(object):
    
    def __init__(self, name):
        'Add the order of parameter'
        
        #Order of input argument for each model

        if isinstance(name, list):
            #If multiple moels

            self._modelList = name
            self.cHat       = self.multiModel

        else :

            if name == 'euclidC':
                self.cHat = self.euclidC

            else:
                raise Exception('Name of model' + '< ' + str(name) + ' >' + \
                            'does not refer to any known model')
    
    '''
                        DEFINING MODELS
    
    Each model must contain as a first argument the input (e.g. the distance matrix)
    and the other argument should be *P, a list of argument. The ordered list of argument
    should be added in the __init__ function for each model.
            
    '''
    
    def euclidC(self, X, P):
    
        '''
        Euclidean distance covariance estimation
        
        D : Distance matrix between channels.
        P : Dictionnary of variables
        
                'sd'   : Standard deviation controlling strength.
                'lamb' : Correlation floor.

        '''
        
        #If each voxel is standardize, then correlation == cov
        #since cov_ij = corr_ij * sqrt(sd_i * sd_j)
        
        #Unpacking dictionnary (changing locals() doesn't work inside functions ...)
        sd = P['sd_euc']; lamb = P['lamb_euc']

        
        #Correlation modelling
        C = np.minimum(np.exp(-X / (2 * sd)) + lamb, 1)

        return C


    def multiModel(self, X, P):
        '''Combine models '''

        for m in range(len(self._modelList)):

            curModel = getattr(self, self._modelList[m])

            if not m:
                C = curModel(X, P)
            else:
                C = C + curModel(X, P)

        return C
    

    
def costFunction(P, modelName, X, Y):

    '''
    Cost function based on the least-squared errors. Need to calculate the residuals.

    P : Dictionnary containing parameters. Name of fields have to match name of 
        variables in the model.
    X : Input data
    Y : Actual data

    modelName : Name of model to use, which will take P parameter as input.
    '''

    #Building model
    m = model(modelName)

    #Calculating predicted Y (stacked)
    Yhat = m.cHat(X, P)

    return (Y - Yhat) + np.sum(Yhat[abs(Yhat) > 1])*100




def crossValFit(X, D, P, modelName):
    ''' 
    Will cross-validate a model (or a list of model)
    
    X     : Data with shape (nvox,npoints,nruns)
    D     : Distance (or other feature) matrix
    P     : Parameter dictionnary for minimize function
    model : Model name
    '''
    
    #Dimensions
    nvox, ntime, nruns = X.shape
    
    #Initializing
    Res  = np.zeros((nruns, nvox**2))
    MSE  = np.zeros(nruns)
    Phat = [None]*nruns 

    #Stacking D
    D = np.hstack(D.copy())
    
    #Loop over all run with 1 being test set
    for r in np.arange(nruns):
        
        #Mask for indexing (excluding test)
        M = np.ones(nruns); M[r] = 0; M = M == 1
        
        #Test sample-covariance
        Ctest = abs(np.hstack(np.corrcoef(X[:,:,r])))
        
        #Arguments for minimize function
        args = (modelName,  # Model name
                D,          # Distance (or input to model) 
                Ctest)      # Sample correlation/covariance matrix
        
        #Fitting model on current run
        out = minimize(costFunction, P, args = args)
        
        #Extracting estimated parameters
        Phat[r] = {k : out.params[k].value for k in out.params.keys()}
        
        #Building model
        m = model(modelName)
        
        #Covariance modelling to test fit
        cHat = m.cHat(D, Phat[r]) #Will be stacked since D is stacked
        
        #Residuals
        Res[r,:] = (Ctest - cHat)
        
        #Mean-Square error
        MSE[r] = np.mean( (Res[r,:])**2)
        
        #Correlation
        corr = np.corrcoef(Ctest, cHat)
        
        print('Fold {} of {} ~ Correlation : {:.2f}'.format(r+1, nruns, corr[0,1]))
        
    return MSE, Res, Phat



def simNoise(n, D, modelParam, modelName = 'euclidC'):
    '''Noise simulation
    
    p : Number of dimensions (voxels)
    n : Number of samples
    
    modelParam : List of parameters for the model (right order), 
                 containing name of model.
    
    '''
    
    #Covariance matrix
    #print('Creating covariance matrix          ... \n\n')
    
    #Number of voxels
    p =D.shape[0]

    #Creating model
    m = model(modelName)
    
    #Covariance
    C = m.cHat(D, modelParam)

    #Simulating noise
    E = np.random.multivariate_normal([0]*p, C, n)
    
    return {'E': E, 'C': C}


