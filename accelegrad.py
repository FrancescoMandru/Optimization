from base_optimizer import BaseOptimizer


class AcceleGrad(BaseOptimizer):
    
    """ Implementation of Accelegrad Algorithm according with the
    official paper"""
    
    def __init__(self, loss, grad, step_size, **kwargs):
        super().__init__(loss, grad, step_size, **kwargs):
            
        self.alpha_weights = set_alpha_weights(n_iters)
        self.diameter = 10**4
        self.x = 0
        self.y = x
        self.z = x
        
        
    def set_alpha_weights(self.n_iters):
        alpha_weights = np.zeros(n_iters)
        for t in range(3,n_iters):
            alpha_weights[t] = 1/4*(t+1)
             
        return alpha_weights