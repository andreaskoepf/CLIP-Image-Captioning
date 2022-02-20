from math import isfinite
from collections import deque
import torch
import numpy as np


class AutoClip:
    def __init__(self, percentile=10, max_keep_history=10000):
        self.grad_history = deque(maxlen=max_keep_history)
        self.percentile = percentile
        self.max_keep_history = max_keep_history

    @torch.no_grad()
    def compute_grad_norm(self, model):
        total_norm = 0
        for p in model.parameters():
            if p.grad is None:
                continue
            total_norm += torch.sum(p.data ** 2).item()
        total_norm = total_norm ** 0.5
        return total_norm 

    def __call__(self, model):
        grad_norm = self.compute_grad_norm(model)
        if isfinite(grad_norm):
            self.grad_history.append(grad_norm)

        if len(self.grad_history) > 0:
            clip_value = np.percentile(self.grad_history, self.percentile)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)


# TODO, try AGC, e.g. see https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/agc.py
