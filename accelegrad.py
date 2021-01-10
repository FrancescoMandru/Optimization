from base_optimizer import BaseOptimizer
from utils import squared_hinge_loss
import numpy as np


class AcceleGrad(BaseOptimizer):
    
    """ Implementation of Accelegrad Algorithm according with the
    official paper"""
    
    def __init__(self, loss, grad, step_size, **kwargs):
        super().__init__(loss, grad, step_size, **kwargs)
            
        self.alpha_weights = set_alpha_weights(n_iters)
        self.diameter = 10**4
        self.gradient_history = np.zeros(n_iters)
        self.x = 0
        self.y = x
        self.z = x
        
        
    def set_alpha_weights(n_iters):
        alpha_weights = np.zeros(n_iters)
        for t in range(3,n_iters):
            alpha_weights[t] = 1/4*(t+1)
             
        return alpha_weights
    
    def update_step_size(cur_iter):
        s = 0
        for i in range(cur_iter):
            s += np.sqrt( alpha_weights[i]**2 * np.linalg.norm(self.grad[:i])**2 )
        
        return 2 * self.diameter * s
    
    def update_x_val(cur_iter):
        tau_t = 1/alpha_weights[cur_iter]
        tau_t * z_t + (1 - tau_t) * y_t
        
    def update_z_val(cur_iter):
        z_t = z_t_one - alpha_weights[cur_iter] * self.update_step_size()
        
    def update_y_val(cur_iter):
        return 0
        
    def gradient_projection(grad):
        return 0