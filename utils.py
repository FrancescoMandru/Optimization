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
    # Single point
    if(len(x.shape)==1):
        return np.sum( np.max(( 0, 1 - y*np.dot(x,w) ))**2 ) + (l/2)*np.sum(w*w)
    # Batch of points 
    else:
        preds = 1 - y*np.dot(x,w)
        # Compute the maximum between 0 and the result for a batch of points
        preds = [np.max(( 0, pred ))**2 for pred in preds]
        return np.sum( preds ) + (l/2)*np.sum(w*w)
    
    
def grad(x, w, y, l):
    """
    Compute the gradient of the squared hinge loss + L2 regularization
    Input:  
        x: Sample
        w: Weights
        y: True label
        l: Lambda regularization value
    Return:
        Squared hinge loss + L2 Regularization
    """
    # Basic reshape for pairwise multiplication
    y = y.reshape(-1,1)
    # Condition to be satisfied wrt to the gradient of the loss to be non zero
    cond = y*np.dot(x,w)
    cond = np.array(cond > 0).astype(int)
    # Compute final gradient 
    grad = np.sum( -cond*x + l*w.T, axis=1)
        
    return grad