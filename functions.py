#Functions

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

def corrSpacePlot(pos, colVal, plot3D = False, title = '', target = False):
    '''
    Will plot in 2d the voxels with color related to a color value 'colVal'.
    
    pos   : Matrix (nvox x 3) containg the XYZ cordinate of each voxel.
            The columns have to be representing X Y Z. 
    
    colVal: Value between -1 and 1 that will determine the color of each voxel 
            (from blue to red).
    
    '''
    
    #Initializing figure
    fig = plt.figure( figsize = (15,10))
    
    if not plot3D:
        plt.subplot(2,2,1)
        xx = plt.scatter(pos[:,0],pos[:,1], c = colVal, cmap = 'jet')
        plt.xlabel('X', fontsize = 20)
        plt.ylabel('Y', fontsize = 20)
        plt.colorbar(xx)

        if target:
            plt.scatter(pos[target,0],pos[target,1], 
            c = 'deeppink', marker = '*', s = 500)

        plt.subplot(2,2,2)
        yy = plt.scatter(pos[:,0],pos[:,2], c = colVal, cmap = 'jet')
        plt.xlabel('X', fontsize = 20)
        plt.ylabel('Z', fontsize = 20)
        plt.colorbar(yy)

        if target:
            plt.scatter(pos[target,0], pos[target,2], 
                c = 'deeppink', marker = '*', s = 500)

        plt.subplot(2,2,3)
        zz = plt.scatter(pos[:,1],pos[:,2], c = colVal, cmap = 'jet')
        plt.xlabel('Y', fontsize = 20)
        plt.ylabel('Z', fontsize = 20)
        plt.colorbar(zz)

        if target:
            plt.scatter(pos[target,1], pos[target,2], 
                c = 'deeppink', marker = '*', s = 500)

    else:
        ax  = fig.gca(projection="3d")

        ax.scatter(pos[:,0],pos[:,1],pos[:,2], c = colVal, s = 12, cmap = 'jet')
        ax.set_xlabel('X', fontsize = 20)
        ax.set_ylabel('Y', fontsize = 20)
        ax.set_zlabel('Z', fontsize = 20)

        #Putting target as pink X
        if target:
            ax.scatter(pos[target,0],pos[target,1],pos[target,2], 
                c = 'deeppink', marker = '*', s = 500)

    
    plt.suptitle(title, fontsize = 30)




def dictUnpack(model, P):
    'Will unpack parameter dictionnary to pass in model in correct order'
    
    #Model arguments
    args = model.order
    keys = list(P.keys())
    
    #Will hold correct index for keys wrt to args
    idx = [0]*len(args)
    
    #Finding indexes to order keys
    for i in range(len(args)):
        if args[i] in keys:
            idx[i] = keys.index(args[i])

    argList =   [P[keys[i]]for i in idx]
    
    return argList


def euclidDmat(X):
    'Calculate the euclidean distance between rows of matrix p x n'
    

    if len(X.shape) == 1:
        p = len(X)
        X = X.copy().reshape(p,1)

    else:
        #Shape
        p, n = X.shape 
    
    #Sum of squared tiled
    SSx = np.tile(np.diag(np.dot( X, X.T )), (p,1)) #sum square X
    
    #Distance
    D = np.sqrt(SSx.T - 2*np.dot( X, X.T ) + SSx)
    
    return D



def filterIdx(dat, idx):
    '''
    Will remove index of all the matrices in the dictionnary dat, by keeping idx.
    
    dat : Dictionnary with relevant datasets
    idx : Indexes of voxels to keep

    '''
    
    #Going over every data
    
    #Taking only relevant indexes
    D = {d : dat[d][idx,...] for d in dat}
    
    return D


def matrixSorting(X):
    'Will sort matrix based on projection on diagonal'
    
    #Projection on diagonal
    xDiag = np.sum(X**2, axis = 1)
    
    #Sorting indexes
    idx   = np.argsort(xDiag)
    
    #Sorting
    Xsort = X[idx,:]
    
    return Xsort, idx


def nearPSD(A, epsilon = 0.1):
    'Nearest positive semi-definite Matrix'
    
    n = A.shape[0]
    eigval, eigvec = np.linalg.eig(A)
    
    val = np.matrix(np.maximum(eigval,epsilon))
    vec = np.matrix(eigvec)
    T   = 1/(np.multiply(vec,vec) * val.T)
    T   = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)) )))
    B   = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
    out = B*B.T
    
    return(out)


def plotCovs(C,Chat):

    plt.figure(figsize = (10,5))

    plt.subplot(1,2,1)
    plt.imshow(C)
    plt.title('True Covariance', fontsize = 20)
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.imshow(Chat)
    plt.title('Estimated Covariance', fontsize = 20)
    plt.colorbar()
    plt.show()
    
    
def standardizeMat(M):
    'Will standardize matrix M so that variance of each row is 1.'
    
    #Shape (number of voxels X timepoints)
    p, n = M.shape

    #Stanard deviation
    sd = np.std(M, axis = 1, ddof = 1)

    #Tiled SD
    tiled = np.tile(np.std(M, axis = 1, ddof = 1), (n,1)).T

    #Standardizing
    mStand = M / tiled

    return mStand


def scaleMod(Y, a, b):
    '''
    Fitting Yhat positive and negative overshoot
    
    Y : Predicted Y
    a : scalar for positive scaling
    b : scalar for negative scaling
    
    '''
    Yscale = Y.copy()
    
    #Scaling positive and negative
    Yscale[Y > 0] = Y[Y > 0] * a
    Yscale[Y < 0] = Y[Y < 0] * b
    
    return Yscale
    
    
def simDist(p):
    'Simulate a distance matrix'

    
    #XYZ position of voxels
    XYZ = np.random.randn(p, 3) + 7; 
    
    #Sorting based on projection on diagonal
    XYZ, _ = matrixSorting(XYZ)
    
    #Euclidean distance between voxels
    D = euclidDmat(XYZ)
    
    return D
    

