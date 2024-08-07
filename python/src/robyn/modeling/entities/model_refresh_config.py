from dataclasses import dataclass
from typing import Literal

#TODO review refresh_mode and if we need this config at all.
@dataclass(frozen=True)
class ModelRefreshConfig:
    refresh_steps: int = 4
    refresh_mode: Literal['manual', 'auto'] = 'manual'
    refresh_iters: int = 1000
    refresh_trials: int = 3