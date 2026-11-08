import math
from typing import List, Optional, Union

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


class LinearWarmupCosineAnnealingLR(LRScheduler):
    r"""Sets the learning rate of each parameter group with a linear warmup phase followed by a cosine annealing phase.

    The learning rate increases linearly from `initial_lr` to `max_lr` over the first `pct_start` of the cycle,
    and then decreases via cosine annealing to `min_lr` over the remaining steps.

    `initial_lr` is determined by `max_lr / div_factor`.
    `min_lr` is determined by `initial_lr / final_div_factor`.

    This scheduler is not chainable.

    Note also that the total number of steps in the cycle can be determined in one
    of two ways (listed in order of precedence):

    #. A value for total_steps is explicitly provided.
    #. A number of epochs (epochs) and a number of steps per epoch
       (steps_per_epoch) are provided.
       In this case, the number of total steps is inferred by
       total_steps = epochs * steps_per_epoch

    You must either provide a value for total_steps or provide a value for both
    epochs and steps_per_epoch.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group.
        total_steps (int): The total number of steps in the cycle. Note that
            if a value is not provided here, then it must be inferred by providing
            a value for epochs and steps_per_epoch.
            Default: None
        epochs (int): The number of epochs to train for. This is used along
            with steps_per_epoch in order to infer the total number of steps in the cycle
            if a value for total_steps is not provided.
            Default: None
        steps_per_epoch (int): The number of steps per epoch to train for. This is
            used along with epochs in order to infer the total number of steps in the
            cycle if a value for total_steps is not provided.
            Default: None
        pct_start (float): The percentage of the cycle (in number of steps) spent
            increasing the learning rate.
            Default: 0.3
        div_factor (float): Determines the initial learning rate via
            initial_lr = max_lr/div_factor
            Default: 25
        final_div_factor (float): Determines the minimum learning rate via
            min_lr = initial_lr/final_div_factor
            Default: 1e4
        last_epoch (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning.
            Default: -1
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: Union[float, List[float]],
        total_steps: Optional[int] = None,
        epochs: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1,
    ):
        # Validate optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")

        # Validate total_steps
        if total_steps is not None:
            if total_steps <= 0 or not isinstance(total_steps, int):
                raise ValueError(f"Expected positive integer total_steps, but got {total_steps}")
            self.total_steps = total_steps
        elif epochs is not None and steps_per_epoch is not None:
            if not isinstance(epochs, int) or epochs <= 0:
                raise ValueError(f"Expected positive integer epochs, but got {epochs}")
            if not isinstance(steps_per_epoch, int) or steps_per_epoch <= 0:
                raise ValueError(
                    f"Expected positive integer steps_per_epoch, but got {steps_per_epoch}"
                )
            self.total_steps = epochs * steps_per_epoch
        else:
            raise ValueError("You must define either total_steps OR (epochs AND steps_per_epoch)")

        self.pct_start = pct_start
        self.warmup_steps = int(self.total_steps * pct_start)

        # Initialize learning rate variables
        max_lrs = self._format_param(optimizer, max_lr)
        for idx, group in enumerate(optimizer.param_groups):
            group["max_lr"] = max_lrs[idx]
            group["initial_lr"] = max_lrs[idx] / div_factor
            group["min_lr"] = group["initial_lr"] / final_div_factor

        super().__init__(optimizer, last_epoch)

    def _format_param(self, optimizer, param):
        """Return correctly formatted parameter for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError(f"expected {len(optimizer.param_groups)} values, got {len(param)}")
            return param
        else:
            return [param] * len(optimizer.param_groups)

    def get_lr(self):
        lrs = []
        step_num = self.last_epoch

        if step_num > self.total_steps:
            raise ValueError(
                f"Tried to step {step_num} times. The specified number of total steps is {self.total_steps}"  # noqa: UP032
            )

        for group in self.optimizer.param_groups:
            if step_num < self.warmup_steps:
                # Linear warmup
                pct = step_num / self.warmup_steps
                lr = group["initial_lr"] + (group["max_lr"] - group["initial_lr"]) * pct
            else:
                # Cosine annealing
                decay_steps = self.total_steps - self.warmup_steps
                if decay_steps > 0:
                    pct = (step_num - self.warmup_steps) / decay_steps
                    cos_out = math.cos(math.pi * pct) + 1
                    lr = group["min_lr"] + (group["max_lr"] - group["min_lr"]) / 2.0 * cos_out
                else:
                    lr = group["max_lr"]
            lrs.append(lr)

        return lrs
