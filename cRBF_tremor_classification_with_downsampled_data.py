import numpy as np
import scipy as sp
from sklearn.decomposition import PCA
from sklearn import datasets
import pickle
import math
from scipy.spatial.distance import cdist
from scipy.special import xlogy

from sklearn import preprocessing


from scipy.special import gammaln
from numpy.linalg import slogdet

from numpy.matlib import repmat
from numpy.linalg import inv


import scipy as sp
from sklearn.decomposition import PCA
from sklearn import datasets
import pickle

from scipy.spatial.distance import cdist
from scipy.special import xlogy

from sklearn import preprocessing
#import GPy

def mapdp_nw(X, N0, m0, a0, c0, B0, epsilon=1e-6, maxiter=100, fDebug=False):
    '''
    MAP-DP for Normal-Wishart data.

    Inputs:  X  - DxN matrix of data
             N0 - prior count (DP concentration parameter)
             m0 - cluster prior mean
             a0 - cluster prior scale
             c0 - cluster prior degrees of freedom
             B0 - cluster prior precision (inverse covariance)
             epsilon - convergence tolerance
             maxiter - maximum number of iterations, overrides threshold calculation
             fDebug - extra verbose input of algorithm operation

    Outputs: mu - cluster centroids
             z  - data point cluster assignments
             K  - number of clusters
             E  - objective function value for each iteration

    CC BY-SA 3.0 Attribution-Sharealike 3.0, Max A. Little. If you use this
    code in your research, please cite:
    Yordan P. Raykov, Alexis Boukouvalas, Fahd Baig, Max A. Little (2016)
    "What to do when K-means clustering fails: a simple yet principled alternative algorithm",
    PLoS One, (11)9:e0162259
    This implementation follows the description in that paper.
    '''
   
    # Initialization (Alg. 3 line 1)
    (D, N) = X.shape
    assert(D > 0)
    assert(N > 0)
    K = 1
    z = np.zeros((N), dtype=int)  # everybody assigned to first cluster
    Enew = np.inf
    dE = np.inf
    ic = 0  # iteration coung
    E = list()
    # Convergence test (Alg. 3 line 14 and Appendix B)
    while (abs(dE) > epsilon and ic < maxiter):
        Eold = Enew
        dik = np.ones((N, 1)) * np.inf
        for i in range(N):
            dk = np.ones((K+1, 1)) * np.inf
            f = np.empty(K+1)
            Nki = np.ones((K), dtype=int)
            xi = np.atleast_2d(X[:, i]).T  # current data point
            for k in range(K):
                zki = (z == k)
                zki[i] = False
                Nki[k] = zki.sum()
                # Updates meaningless for Nki=0
                if (Nki[k] == 0):
                    continue
                # Update NW cluster hyper parameters (Alg. 3 line 7)
                mki, aki, cki, Bki = nwupd(Nki[k], X[:, zki], m0, a0, c0, B0)

                # Compute Student-t NLL, existing clusters (Alg. 3 line 8)
                dk[k] = stnll(xi, mki, aki, cki, Bki, D)
                # Avoid reinforcement effect at initialization (Appendix B)
                if (ic == 0):
                    Nki[0] = 1
                f[k] = dk[k]-np.log(Nki[k])
            # Compute Student-t NLL, new cluster (Alg. 3 line 9)
            dk[K] = stnll(xi, m0, a0, c0, B0, D)
            f[K] = dk[K]-np.log(N0)
            # Compute MAP assignment (Alg. 3 line 10)
            if(fDebug):
                print(i, 'Compute MAP assignment K=', K, 'f=', f, 'dk=', dk)

            z[i] = np.argmin(f)
            dik[i] = f[z[i]]
            # Create new cluster if required (Alg. 3 line 11-12)
            if (z[i] == K):
                K = K + 1
        # Remove any empty clusters and re-assign (Appendix B)
        Knz = 0
        for k in range(K):
            i = (z == k)
            Nk = i.sum()
            if (Nk > 0):
                z[i] = Knz
                Knz = Knz + 1
        K = Knz
        Nk, _ = np.histogram(z, range(K+1))
        # Compute updated NLL (Alg. 3 line 13)
        Enew = dik.sum()-K*np.log(N0)-np.sum(gammaln(Nk))
        dE = Eold - Enew
        ic += 1
        E.append(Enew)
        print('Iteration %d: K=%d, E=%f, dE=%f\n' % (ic, K, Enew, dE))

    # Compute cluster centroids (Appendix D)
    mu = np.ones((D, K))
    sigma = np.ones((D, K))
    for k in range(K):
        xk = X[:, z == k]
        mu[:, k] = xk.mean(1)
        sigma[:,k] = xk.std(1)
    return mu, sigma, z, K, E


def stnll(x, m, a, c, B, D):
    '''
    Compute Student-t negative log likelihood (Appendix A, eqn. (20))
    '''
    mu = m
    nu = a-D+1
    Lambda = c*float(nu)/(c+1)*B
    S = np.dot(np.dot((x-mu).T, Lambda), (x-mu))
    _, logdetL = slogdet(Lambda)
    return float(nu+D)/2.*np.log(1.+S/float(nu))\
        - 0.5*logdetL+gammaln(nu/2.)\
        - gammaln((float(nu)+D)/2.)+D/2.*np.log(float(nu)*np.pi)


def nwupd(Nki, xki, m0, a0, c0, B0):
    '''
    Update Normal-Wishart hyper parameters (Appendix A, eqns. (18-19))
    '''
    xmki = xki.mean(1)[:, None]
    xmcki = xki-repmat(xmki, 1, Nki)
    Ski = np.dot(xmcki, xmcki.T)
    cki = c0+Nki
    mki = (c0*m0+Nki*xmki)/cki
    xm0cki = xmki-m0
    Bki = inv(inv(B0)+Ski+c0*Nki/cki*np.dot(xm0cki, xm0cki.T))
    aki = a0+Nki
    return mki, aki, cki, Bki


def tpsrbf(x, mu, beta, ax = None):
    #Add epsilon to avoid zero value for log
    dist = np.divide((x - mu)**2, beta)
    dist = np.sum(dist, axis=0)
    r = dist    
    out = np.exp(-r)
    if r.any()==0:#r == 0):
        out[np.where(r==0)] = 1
    return out

# def compute_phi(X, C, Beta):    
#     T = X.shape[1]
#     h = C.shape[0]      
#     r = C.shape[1]
#     d = C.shape[1]
#     phi = np.zeros([T, h])    
#     for i in range(h):
#         phi[:,i] = tpsrbf(X, C[i,:].reshape(d,1), Beta[i,:].reshape(d,1), ax = 0)
#     return phi

def compute_phi(function, X, C, Beta = None, C_sum = None, N = None, theta = None):
    
    #C_sum, N, and theta only needed if function == 'invquad'
    if function == 'gaussian':
        X = X.T
        T = X.shape[1]
        h = C.shape[0]
        d = C.shape[1]
        phi = np.zeros([T, h]) 
        for i in range(h):
            phi[:,i] = tpsrbf(X, C[i,:].reshape(d,1), Beta[i,:].reshape(d,1), ax = 0)
    
    if function == 'invquad':
        T = X.shape[0]
        phi = 1/(theta * np.dot(np.sum(X**2, 1).reshape(-1,1), np.ones([1,N])) + \
         np.dot(X,C) + np.dot(np.ones([T,1]), C_sum))
        
    elif function == 'z2logz':
        z = cdist(X, C, 'euclidean')
        phi = xlogy(z**2,z)
        
    elif function == 'zlogz':
        z = cdist(X, C, 'euclidean')
        phi = xlogy(z,z)
    return phi

def compute_grad(function, X, W, C, N, class_labels, nl, l, d, Beta = None):
    
    T = X[nl].shape[0]
    grad = X[nl] - class_labels
    for k in np.arange(nl-1,l-1,-1):
        if function == 'zlogz':
            z = cdist(X[k], C[k], 'euclidean')
            G = -np.multiply(np.dot(grad, W[k].T), np.log(z) + 1)
            grad = np.zeros([T, d[k-1]])
            for ind1 in range(T):
                grad[ind1,:] = np.dot(G[ind1,:].reshape(1,-1), (np.dot(np.ones([N[k],1]), X[k][ind1,:].reshape(1,-1)) - C[k]))
        elif function == 'gaussian':
            G = -np.multiply(np.dot(grad, W[k].T), compute_phi('gaussian',X[k], C[k], Beta[k]))#tpsrbf(X[k], C[k], Beta[k]))
            grad = np.zeros([T, d[k-1]])
            for ind1 in range(T):
                grad[ind1,:] = np.dot(G[ind1,:].reshape(1,-1), (np.dot(np.ones([N[k],1]), X[k][ind1,:].reshape(1,-1)) - C[k]))
    grad[np.isinf(grad)] = 0
    return grad

from sklearn import preprocessing

def CrossEntropyLoss(yHat, y):
    T = y.shape[0]
    K = y.shape[1]
    E = 0

    for t in range(T):
        for k in range(K):
            E = E - y[t,k]*math.log2(yHat[t,k])
    return E
    
  

def NS(X, W, phi, phi_inv, niter):
#Inputs:
#    X: T x D matrix of the layers input data
#    W: dictionary of the wieghts at each layer
#    phi: output of layer before weight multiplication
#    niter: maximum number of training iterations
    
    T = X.shape[0]
    d = W.shape[1]
    
    Y = np.dot(phi,W)
    
    Dx = sp.spatial.distance.cdist(X, X)
    Dy = sp.spatial.distance.cdist(Y, Y)
    
    error = 0.5*np.linalg.norm((Dx - Dy)/(Dx + (Dx == 0)))
    
    eta = 1e-3
    kup = 2.5
    kdown = 0.1
    success = 1
    c = 0
        
    for epoch in range(niter):
        if success == 1:
            grad = np.zeros([T, d])
            for i in range(T):
                grad[i,:] = - np.dot(((Dx[i,:] - Dy[i, :])/((Dy[i, :] + (Dy[i, :] == 0))*(Dx[i, :] + (Dx[i, :] == 0)))), \
                    (Y[i,:] - Y))
           
        Y_new = Y - eta*grad
        W_new = np.dot(phi_inv, Y_new)
        Y_new = np.dot(phi, W_new)
        
        Dy = sp.spatial.distance.cdist(Y_new, Y_new)           
        error_new = 0.5*np.linalg.norm((Dx - Dy)/(Dx + (Dx == 0)))           
    
        if error > error_new:
            success = 1
            eta = eta*kup
            error = error_new
            W = W_new.copy()
            Y = Y_new.copy()
            c = 0
        else:
            success = 0
            eta = eta*kdown
            c += 1
            
        if c > 20:
            break
        
        print('Iter: ', epoch, 'Error: ', error)
        
    return Y, W


def CRBFtrain(data, class_labels, nl, niter, eta_s, N, d, basis_shape = 'gaussian', basis_parameters = None, use_pca = True, basis_learning = None):  
    #Inputs:
    #    data: T x D matrix of the input data
    #    target: T x D_target matrix of the output targets
    #    nl: number of layers for the netwrok
    #    niter: number of iterations for the post training step
    #    eta: starting learning rate of the post training step
    #    N: a list of the number of centers at each layer of the RBF
    #    d: a list of number of the dimension of the data projection after each layer in the RBF
    #    use_pca: if True initializes each layer with PCA for pre-training
    #    basis_learning: learn basis parameters by either fixing them to domain informed values, random subselection or clustering 
      
    X = {}
    W = {}
    C_sum = {}
    C = {}
    Beta = {}
    
    X[0] = data
    T = X[0].shape[0]
    
    for i in range(nl):                
        if i < nl - 1:
            if basis_learning == None:    
                if basis_shape == 'zlogz':
                    centres = np.random.choice(np.arange(T), N[i], replace = False)
                    C[i] = (X[i][centres,:])
                if basis_shape == 'gaussian':
                    centres = np.random.choice(np.arange(T), N[i], replace = False)
                    C[i] = (X[i][centres,:])
                    Beta[i] = np.random.uniform(0,1,(N[i],C[i].shape[1]))*epsilon
                  #  Beta[i] = np.std(C[i],0)*(1/N[i]) # Make basis proportionate to the variance 
                if basis_shape == 'invquad':
                    theta = 1e12; #global scaling parameter for inv quadratic basis function
                    fl_theta = 1e12
                    fl_theta2 = -2*fl_theta
                    theta2 = -2*theta; #for use with centres
                    C_sum[i] = (theta * np.sum(X[i][centres,:]**2, 1) + 1).reshape(1,-1)
                    C[i] = (theta2 * X[i][centres,:]).T
            elif basis_learning == 'clustering':
                N0 = 3
                m0 = X[i].mean(1)[:, None]    # Normal-Wishart prior mean
                a0 = 100             # Normal-Wishart prior scale
                c0 = 10/float(10000)    # Normal-Wishart prior degrees of freedom
                B0 = np.diag(1/(0.05*X[i].var(1)))  # Normal-Wishart prior precision
                # # Run MAPDP to convergence
                mu_basis = np.zeros((d[i],1))
                sigma_basis = np.zeros((d[i],1))
                mu, z, K, E = mapdp_nw(X[i].T, N0, m0, a0, c0, B0)
                mu_basis = np.hstack((mu_basis, mu))
                for k in np.unique(z):
                    sigma_basis = np.hstack((sigma_basis, np.std(X[i][k,:].T)))
                C[i] = mu_basis
                Beta[i] = sigma_basis                
            elif basis_learning == 'fixed':
                C[i] = basis_parameters.mu[i]
                Beta[i] = basis_parameters.sigma[i]                
            
            if basis_shape =='zlogz':
                phi = compute_phi(basis_shape, X[i], C[i])
            if basis_shape == 'gaussian':
                phi = compute_phi(basis_shape, X[i], C[i], Beta[i])
                
            if use_pca:
                if d[i] > X[i].shape[1]:
                    raise SyntaxError('use_pca should be set to false when projecting from lower dimensions to higher dimensions')
                X[i+1] = PCA(n_components = d[i]).fit_transform(X[i])
                phi_inv = np.linalg.pinv(phi)
                W[i] = np.dot(phi_inv, X[i+1])
                X[i+1] = np.dot(phi, W[i])
              
            else:
                phi_inv = np.linalg.pinv(phi)
                W[i] = np.random.randn(N[i],d[i])
                X[i+1] = np.dot(phi, W[i])

            print('Pretrain layer: ', i)
            X[i+1], W[i] = NS(X[i], W[i], phi, phi_inv, 400)
                   
        elif i == nl - 1:
            l = 1
            if basis_shape =='zlogz':
                phi = compute_phi(basis_shape, X[0], C[0])
            if basis_shape == 'gaussian':
                phi = compute_phi(basis_shape, X[0], C[0], Beta[0])
         #   phi = compute_phi(basis_shape, X[0], C[0], Beta[0])
            
            
            W[0] = np.dot(np.linalg.pinv(phi), X[l])
            X[l] = np.dot(phi, W[0])
            for ind2 in range(l,nl-1):
                if basis_shape =='zlogz':
                    phi = compute_phi(basis_shape, X[ind2], C[ind2])
                if basis_shape == 'gaussian':
                    phi = compute_phi(basis_shape, X[ind2], C[ind2], Beta[ind2])
                X[ind2+1] = np.dot(phi, W[ind2])
            
            ## Update centers for the following layer if they are not known
            
            if basis_learning == None:
              #  centres = np.random.choice(np.arange(T), N[i],replace = False)
              #  C[i] = (X[i][centres,:])
                if basis_shape == 'zlogz':
                    centres = np.random.choice(np.arange(T), N[i], replace = False)
                    C[i] = (X[i][centres,:])
                if basis_shape == 'gaussian':
                    centres = np.random.choice(np.arange(T), N[i], replace = False)
                    C[i] = (X[i][centres,:])
                    Beta[i] = Beta[i] = np.random.uniform(0,1,(N[i],C[i].shape[1]))*epsilon # Make basis proportionate to the variance 
                if basis_shape == 'invquad':
                    theta = 1e12; #global scaling parameter for inv quadratic basis function
                    fl_theta = 1e12
                    fl_theta2 = -2*fl_theta
                    theta2 = -2*theta; #for use with centres
                    C_sum[i] = (theta * np.sum(X[i][centres,:]**2, 1) + 1).reshape(1,-1)
                    C[i] = (theta2 * X[i][centres,:]).T
            
            if basis_shape =='zlogz':
                phi = compute_phi(basis_shape, X[i], C[i])
            if basis_shape == 'gaussian':
                phi = compute_phi(basis_shape, X[i], C[i], Beta[i])
            #phi = compute_phi(basis_shape, X[i], C[i], Beta[i])
            
            # Last layer needs a softmax (logistic link function) to map binary targets
            W[i] = np.dot(np.linalg.pinv(phi), class_labels)
            X[i+1] = np.dot(phi, W[i])
            
    print('pretraining done')
    
    k_up = 2.5; #learning increase rate (1.2 standard)
    k_down = 0.1; #learning decrease rate
    
    for l in range(1,nl):
        if basis_shape =='zlogz':
            phi = compute_phi(basis_shape, X[l-1], C[l-1])
        if basis_shape == 'gaussian':
            phi = compute_phi(basis_shape, X[l-1], C[l-1], Beta[l-1])   
        W[l-1] = np.dot(np.linalg.pinv(phi), X[l])
        X[l] = np.dot(phi, W[l-1])
        for ind2 in range(l,nl):
            if basis_shape =='zlogz':
                phi = compute_phi(basis_shape, X[ind2], C[ind2])
            if basis_shape == 'gaussian':
                phi = compute_phi(basis_shape, X[ind2], C[ind2], Beta[ind2])
            X[ind2+1] = np.dot(phi, W[ind2]) 
        print('shape of X[nl]')
        print(np.shape(X[nl]))
        X[nl] = np.exp(X[nl])/np.sum(np.exp(X[nl]), axis = 1).reshape(-1,+1)
        print('shape of X[nl] after normalizing')
        print(np.shape(X[nl]))
        error = CrossEntropyLoss(X[nl], class_labels)
       # error = np.sqrt(np.sum((target - X[nl])**2))
        
        success = 1
        eta = eta_s
        for it in range(niter):
            if success:
                #grad = X[nl] - class_labels
                grad = compute_grad(basis_shape, X, W, C, N, class_labels, nl, l, d, Beta) 
                
            #    for k in np.arange(nl-1,l-1,-1):
            #        G = -np.multiply(np.dot(grad, W[k].T), (compute_phi(X[k].T,C[k],Beta[k]))**2)
            #        grad = np.zeros([T, d[k-1]])
            #        for t in range(T):
            #            grad[t,:] = np.dot(G[t,:].reshape(1,-1),(np.dot(np.ones([N[k],1]),X[k][t,:].reshape(1,-1)) - C[k].T/theta2))##### WHAT TO put here

            X_old = X.copy()
            W_old = W.copy()
            X[l] = X[l] - eta*grad
            if basis_shape == 'zlogz':            
                phi = compute_phi(basis_shape, X[l-1],C[l-1])
            if basis_shape == 'gaussian':
                phi = compute_phi(basis_shape, X[l-1], C[l-1], Beta[l-1])
            
            W[l-1] = np.dot(np.linalg.pinv(phi), X[l])
            X[l] = np.dot(phi, W[l-1])
            for ind2 in range(l,nl):
                if basis_shape =='zlogz':
                    phi = compute_phi(basis_shape, X[ind2], C[ind2])
                if basis_shape == 'gaussian':
                    phi = compute_phi(basis_shape, X[ind2], C[ind2], Beta[ind2])
                X[ind2+1] = np.dot(phi, W[ind2])                        
            
            X[nl] = np.exp(X[nl])/np.sum(np.exp(X[nl]), axis = 1).reshape(-1,+1)
            error_new = CrossEntropyLoss(X[nl], class_labels)
                                    
            if error > error_new:
                success = 1
                eta = eta*k_up
                error = error_new
            else:
                success = 0
                X = X_old.copy()
                W = W_old.copy()
                eta = eta*k_down
               
            if not it %5:
                print(error_new)
                print(eta)
                print('Layer: ', l, 'Iteration: ', it, 'Error: ', error)
    
    l = nl - 1
    if basis_shape == 'zlogz':
        phi = compute_phi(basis_shape, X[l], C[l])        
    if basis_shape == 'gaussian':
        phi = compute_phi(basis_shape, X[l], C[l], Beta[l])   
    W[l] = np.dot(np.linalg.pinv(phi), class_labels)
    X[nl] = np.dot(phi, W[l])
    X[nl] = np.exp(X[nl])/np.sum(np.exp(X[nl]), axis = 1).reshape(-1,+1)
    error_new = CrossEntropyLoss(X[nl], class_labels)
    print('Final Error: ', error_new)
    
    return X, W, C, Beta, N
#CRBFtrain(data, class_labels, nl, niter, eta_s, N, d, basis_shape = 'gaussian', basis_parameters = None, use_pca = True, basis_learning = None):  

def CRBFtest(data_test, W, C, nl, basis_shape = 'gaussian', Beta = None, C_sum = None):  
#Inputs:
#    data: T x D matrix of the test data
#    W: dictionary of the wieghts at each layer
#    C: dictionary of the centers at each layer
#    C_sum: sum of squared centers for faster calculation
#    nl: number of layers for the netwrok
       
    X_test = {}
    X_test[0] = data_test  
    T_test = X_test[0].shape[0]
    
    if basis_shape == 'zlogz':
        for i in range(nl):
            phi = compute_phi('zlogz', X_test[i], C[i])
            X_test[i+1] = np.dot(phi, W[i])
    if basis_shape == 'gaussian':
        for i in range(nl):
            phi = compute_phi('gaussian', X_test[i], C[i], Beta[i])
            X_test[i+1] = np.dot(phi, W[i])
    if basis_shape == 'z2logz':
        for i in range(nl):
            phi = compute_phi('z2logz', X_test[i], C[i])
            X_test[i+1] = np.dot(phi, W[i])
    if basis_shape == 'invquad':
        for i in range(nl):
            phi = compute_phi('invquad', X_test[i], C[i], C_sum[i])
            X_test[i+1] = np.dot(phi, W[i]) 
    return X_test


# =============================================================================
# Import and prepare data data
# =============================================================================
    
import numpy as np
from matplotlib import pyplot as plt
#import GPy # import GPy package
np.random.seed(12345)


data_reduced = data_features.sample(frac=0.03, replace=False, random_state=1)
Y_features = data_reduced.to_numpy()

indices = np.arange(6597)
np.random.shuffle(indices)

data_train = Y_features[indices,1:46]
temp = Y_features[indices,48].reshape(-1,1)
target_train = np.zeros([6597,2])
for i in range(6597):
    target_train[i, int(temp[i])] = 1
    
    

    
#data_test = data_tot[120:,:]
#target_test = target_tot[120:,:]

# =============================================================================
# Specify and train model
# =============================================================================

nl = 3
niter = 20
eta = 0.1
N = [6597,6597,70]
d = [17,2,2]
use_pca = True
epsilon = 10

#CRBFtrain(data, class_labels, nl, niter, eta_s, N, d, basis_shape = 'gaussian', basis_parameters = None, use_pca = True, basis_learning = None):  

X, W, C, Beta, N = CRBFtrain(data_train, target_train, nl, niter, eta, N, d, basis_shape = 'gaussian', use_pca = use_pca)
Embeddings_and_classes = np.hstack((X[2], Y_features[:,0].reshape(-1,+1), Y_features[:,46:51]))

data_features_reduced_d = pd.DataFrame(data=Embeddings_and_classes, columns=["Axis_1", "Axis_2","ID","Medication_Intake","Prototype_ID","Non-tremor/Tremor","Activity_label","Non_tremor_activity_labels"])
import seaborn as sns
sns.kdeplot(data=data_features_reduced_d, x="Axis_1", y="Axis_2", hue="Non-tremor/Tremor")
sns.scatterplot(data=data_features_reduced_d, x="Axis_1", y="Axis_2")

clas1 = sp.spatial.distance.cdist(X[3], np.array([[1,0,0],[0,1,0],[0,0,1]]))
pred_train = np.argmin(clas1, axis = 1)



# =============================================================================
# Test trained model out of sample
# =============================================================================

X_test = CRBFtest(data_test, W, C, nl, 'gaussian', Beta)
X_test[nl] = np.exp(X_test[nl])/np.sum(np.exp(X_test[nl]), axis = 1).reshape(-1,+1)
clas2 = sp.spatial.distance.cdist(X_test[3], np.array([[1,0,0],[0,1,0],[0,0,1]]))
pred_test = np.argmin(clas2, axis = 1)

print('misclustered training points: ', np.where(pred_train != temp[:120,0])[0])
print('misclustered test points: ', np.where(pred_test != temp[120:,0])[0])

### Inducing points

### Gaussian kernel checks


