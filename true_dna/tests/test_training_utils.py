from itertools import pairwise

import pytest
import torch
from dna_model.utils import EMA, get_cosine_schedule_with_warmup_and_cooldown


def test_cosine_cooldown_is_continuous_and_monotonic():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = get_cosine_schedule_with_warmup_and_cooldown(
        optimizer,
        num_warmup_steps=10,
        num_training_steps=100,
        num_cooldown_steps=10,
        min_lr_ratio=0.1,
    )

    multiplier = scheduler.lr_lambdas[0]
    decay_values = [multiplier(step) for step in range(10, 101)]

    assert all(left >= right for left, right in pairwise(decay_values))
    assert multiplier(90) <= multiplier(89)
    assert multiplier(100) == pytest.approx(0.1)


def test_ema_load_accepts_distributed_wrapper_prefix():
    model = torch.nn.Linear(2, 1)
    ema = EMA(model)
    saved = {
        "module.weight": torch.full_like(model.weight, 2.0),
        "module.bias": torch.full_like(model.bias, 3.0),
    }

    ema.load_state_dict(saved)

    assert torch.equal(ema.shadow["weight"], saved["module.weight"])
    assert torch.equal(ema.shadow["bias"], saved["module.bias"])
