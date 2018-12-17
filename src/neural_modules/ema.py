import torch
import os
from collections import OrderedDict

# Default parameter decay=0.9999


class EMA():
    def __init__(self, parameters, decay=0.9999):
        self.decay = decay
        self.steps = 0
        self.shadow = OrderedDict()
        for _name, _parameter in parameters:
            if _parameter.requires_grad:
                self.shadow[_name] = _parameter.clone()

    def __call__(self, parameters):
        self.steps += 1
        decay = min((self.steps + 1) / (10 + self.steps), self.decay)

        for _name, _parameter in parameters:
            if _parameter.requires_grad:
                new_average = (1.0 - decay) * _parameter.data + decay * self.shadow[_name]
                self.shadow[_name] = new_average.clone()
        return self.shadow


def save_ema_to_file(ema_model, filename):
    torch.save(ema_model.shadow, filename)


def load_ema_to_model(model, ema_model):
    if not isinstance(ema_model, EMA):
        ema_shadow = torch.load(ema_model)
    else:
        ema_shadow = ema_model.shadow

    state_dict = model.state_dict()
    state_dict.update(ema_shadow)
    model.load_state_dict(state_dict)