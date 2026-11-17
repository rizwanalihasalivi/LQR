"""Same eleven observations; avoid float32 cancellation before conditioning."""
from pathlib import Path
import sys
import numpy as np

LEGACY = Path(__file__).resolve().parents[2] / 'lans-tracking-study/code'
sys.path.insert(1, str(LEGACY))
from features import NetworkEnv, whiten
from core import XS


class PrecisionEnv(NetworkEnv):
    def obs(self):
        # Algebraically identical to ImprovedEnv.obs before rounding. The four
        # previews contain reference information, with present error and heading
        # removed. Compute this difference before the single float32 conversion.
        h = .01 * np.arange(1, 5)
        preview = (self.ref[0, self.k] + self.ref[1, self.k] * h
                   - self.ref[0, self.k + np.arange(1, 5)]) / (10 * h * h)
        raw = np.r_[self.x / XS, self.vx / 25, preview,
                    self.last_delay / 200,
                    (np.mean(self.history) if self.history else 0.) / 200]
        return whiten(raw)
