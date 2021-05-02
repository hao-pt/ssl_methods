from __future__ import division
from __future__ import unicode_literals

import torch

# Partially based on: https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/moving_averages.py
class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.
    """
    def __init__(self, parameters, decay, use_num_updates=True, 
        rampup_steps=20000, rampup_decay=0.99):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
          decay: The exponential decay.
          use_num_updates: Whether to use number of updates when computing
            averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.rampup_steps = rampup_steps
        self.rampup_decay = rampup_decay

        self.use_num_updates = use_num_updates
        self.num_updates = 0

        parameters = list(parameters)
        # print("Initial param:", list(parameters)[0][0][0][0])
        self.shadow_params = [p.clone().detach()
                              for p in parameters]
        # print("Initial shadow:", self.shadow_params[0][0][0][0])

        self.collected_params = []

    def set_params(self, parameters):
        """
        Set new parameters for shawow_params
        """
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]
      
    def update(self, parameters):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """

        # decay = self.decay
        # if self.num_updates is not None:
        #     self.num_updates += 1
        #     decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        # one_minus_decay = 1.0 - decay
        # with torch.no_grad():
        #     parameters = [p for p in parameters if p.requires_grad]
        #     for s_param, param in zip(self.shadow_params, parameters):
        #         s_param.sub_(one_minus_decay * (s_param - param))
        
        alpha = self.decay
        self.num_updates += 1
        if self.use_num_updates:
            alpha = min(1-1/(self.num_updates+1), alpha) # use normal average until model is better at later step
        else:
            alpha = self.decay if self.num_updates > self.rampup_steps else self.rampup_decay 
        # print(f"Step={self.num_updates}, alpha={alpha}")

        parameters = list(parameters)
        # print("s params:", self.shadow_params[0][0][0][0])
        # print("params:", parameters[0][0][0][0])
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                # theta'(t) = alpha * theta'(t-1) + (1-alpha) * theta(t)
                s_param.mul_(alpha).add_(1-alpha, param.data)
        # print("new s params:", self.shadow_params[0][0][0][0])
        
        
    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.
        Args: 
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            # if param.requires_grad:
                # param.data.copy_(s_param.data)
            param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args: 
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        parameters = list(parameters)
        self.collected_params = [param.clone()
                                 for param in parameters
                                 if param.requires_grad]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args: 
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        parameters = list(parameters)
        for c_param, param in zip(self.collected_params, parameters):
            if param.requires_grad:
                param.data.copy_(c_param.data)

if __name__ == "__main__":
	params = torch.randn((3,4))
	ema_params = params.clone()
	ema_updater = ExponentialMovingAverage(params, decay=0.999)
	
	params += torch.randn((3,4))

	ema_updater.update()