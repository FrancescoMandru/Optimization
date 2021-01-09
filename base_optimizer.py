class BaseOptimizer():
    
    def __init__(self, loss, grad, step_size, **kwargs):
        self.loss = loss
        self.grad = grad
        self.step_size = step_size
        
    def get_convergence(self):
        return 0
    
    def get_step_size(self):
        return 0
        
    def solve(self):
        return 0