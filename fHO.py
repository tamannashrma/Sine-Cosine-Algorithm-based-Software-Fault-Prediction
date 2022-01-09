import numpy as np
from sklearn.neighbors import KNeighborsClassifier


# error rate
def error_rate(x_train, y_train, x, opts):
    # parameters
    k     = opts['k']
    fold  = opts['fold']
    xt    = fold['xt']
    yt    = fold['yt']
    xv    = fold['xv']
    yv    = fold['yv']
    
    # Number of instances
    num_train = np.size(xt, 0)
    num_valid = np.size(xv, 0)
    # Define selected features
    x_train  = xt[:, x == 1]
    y_train  = yt.reshape(num_train)  # Solve bug
    x_valid  = xv[:, x == 1]
    y_valid  = yv.reshape(num_valid)  # Solve bug   
    # Training
    mdl     = KNeighborsClassifier(n_neighbors = k)
    mdl.fit(x_train, y_train)
    # Prediction
    y_pred   = mdl.predict(x_valid)
    acc     = np.sum(y_valid == y_pred) / num_valid
    error   = 1 - acc
    
    return error


# Error rate & Feature size
def Fnc(x_train, y_train, x, opts):
    # Parameters
    alpha    = 0.99
    beta     = 1 - alpha
    # Original feature size
    max_feat = len(x)
    # Number of selected features
    num_feat = np.sum(x == 1)
    # Solve if no feature selected
    if num_feat == 0:
        cost  = 1
    else:
        # Get error rate
        error = error_rate(x_train, y_train, x, opts)
        # Objective function
        cost  = alpha * error + beta * (num_feat / max_feat)
        
    return cost

