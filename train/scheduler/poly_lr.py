import math
import torch

from .scheduler import Scheduler
import numpy as np

class PolyLRScheduler(Scheduler):
    """
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial = 100,
                 power = 0.9,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)
        
        self.max_epoch = t_initial
        self.power = power
        self.t_in_epochs = t_in_epochs

    def _get_lr(self, t):
        lrs = [round(v * np.power(1-(t) / self.max_epoch, self.power), 8) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None