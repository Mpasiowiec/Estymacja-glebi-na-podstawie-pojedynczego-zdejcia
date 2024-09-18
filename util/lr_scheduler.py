# Copied from https://github.com/prs-eth/Marigold/blob/main/src/util/lr_scheduler.py
import numpy as np
import warnings
from typing import (
    List,
    Literal,
    SupportsFloat,
    Union,
)
import math
from torch import inf
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import (
    LRScheduler,
    _check_verbose_deprecated_warning,
    EPOCH_DEPRECATION_WARNING    
)



class ReduceLROnPlateau(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        mode: Literal["min", "max"] = "min",
        factor=0.1,
        patience=10,
        threshold=1e-4,
        threshold_mode: Literal["rel", "abs"] = "rel",
        cooldown=0,
        min_lr: Union[List[float], float] = 0,
        eps=1e-8,
        verbose="deprecated",
    ):
        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0.")
        self.factor = factor

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer

        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(
                    f"expected {len(optimizer.param_groups)} min_lrs, got {len(min_lr)}"
                )
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience

        self.verbose = _check_verbose_deprecated_warning(verbose)
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best: float
        self.num_bad_epochs: int
        self.mode_worse: float  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
        self._init_is_better(
            mode=mode, threshold=threshold, threshold_mode=threshold_mode
        )
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics: SupportsFloat, epoch=None):  # type: ignore[override]
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group["lr"] = new_lr

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold
            return a < best * rel_epsilon

        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold

        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.0
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")

        if mode == "min":
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def state_dict(self):
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(
            mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode
        )


class IterExponential:
    def __init__(self, total_iter_length, final_ratio, warmup_steps=0) -> None:
        """
        Customized iteration-wise exponential scheduler.
        Re-calculate for every step, to reduce error accumulation

        Args:
            total_iter_length (int): Expected total iteration number
            final_ratio (float): Expected LR ratio at n_iter = total_iter_length
        """
        self.total_length = total_iter_length
        self.effective_length = total_iter_length - warmup_steps
        self.final_ratio = final_ratio
        self.warmup_steps = warmup_steps

    def __call__(self, n_iter) -> float:
        if n_iter < self.warmup_steps:
            alpha = 1.0 # * n_iter / self.warmup_steps
        elif n_iter >= self.total_length:
            alpha = self.final_ratio
        else:
            actual_iter = n_iter - self.warmup_steps
            alpha = np.exp(
                actual_iter / self.effective_length * np.log(self.final_ratio)
            )
        return alpha


if "__main__" == __name__:
    # lr_scheduler = IterExponential(
    #     total_iter_length=50000, final_ratio=0.01, warmup_steps=2000
    # )
    # lr_scheduler = IterExponential(
    #     total_iter_length=50000, final_ratio=0.2, warmup_steps=0
    # )
    from torch.optim import SGD
    from torchvision.models import resnet50
    model = resnet50()
    opt = SGD(model.parameters(), lr=1)
    test = ReduceLROnPlateau(optimizer=opt, mode='min', factor=0.99, patience=1, min_lr=0.0000001)
    x = np.arange(150)
    loss = [-10*x*np.sin(20*i/100 + 1.3) for i in x]
    alphas = []
    for i in loss:
        test.step(i)
        alphas.append(test.get_last_lr())
    import matplotlib.pyplot as plt
    f, a = plt.subplots(2,1)
    a[0].plot(loss)
    a[1].plot(alphas)
    plt.savefig("lr_scheduler.png")