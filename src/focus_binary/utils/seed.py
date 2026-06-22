from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


def set_global_seed(seed: int, deterministic: Optional[bool] = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if tf is not None:
        tf.random.set_seed(seed)
        if deterministic:
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
