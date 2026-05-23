"""Deterministic seeding utilities.

Reproducibility in this codebase is layered: the experiment driver chooses a
seed for the *debate dynamics* (decoding noise, topology shuffling) and an
*independent* seed for *agent relabelings*. Both are advanced through this
module so that the two axes can be analysed separately.
"""

from __future__ import annotations

import os
import random
from typing import Optional


def seeded_rng(seed: int) -> random.Random:
    """Return a :class:`random.Random` instance bound to ``seed``.

    The function is a one-liner but exists so that the rest of the codebase
    only imports from ``utils`` rather than from ``random`` directly.
    That makes it easy to swap in a different RNG (e.g. ``numpy.random``) in
    a future version without touching every call site.
    """
    return random.Random(seed)


def set_global_seeds(seed: int, *, torch_seed: Optional[int] = None) -> None:
    """Best-effort attempt to seed every RNG that might be in scope.

    We seed ``random`` and ``PYTHONHASHSEED`` unconditionally, and seed
    ``numpy``/``torch`` only if those packages are importable (the open-source
    extras may or may not be installed).

    This should be called once at process start. Per-example reproducibility
    is enforced inside the graph by :func:`seeded_rng`, not by this helper.
    """
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    try:
        import numpy as np  # noqa: WPS433 (intentional optional import)

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch  # noqa: WPS433

        torch.manual_seed(torch_seed if torch_seed is not None else seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed if torch_seed is not None else seed)
    except Exception:
        pass
