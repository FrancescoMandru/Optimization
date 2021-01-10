import numpy as np

def squared_hinge_loss(x, w, y, l):
    """Squared hinge loss function + L2 Regularization. 
    Input:  
        x: Sample
        w: Weights
        y: True label
        l: Lambda regularization value
    Return:
        Squared hinge loss + L2 Regularization
    """
    return np.sum( np.max( 0, 1 - y*np.dot(x,w) )**2 ) + (l/2)*np.sum(w*w)