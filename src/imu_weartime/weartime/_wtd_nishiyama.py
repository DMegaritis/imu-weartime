from typing_extensions import Self
from typing import Any, Literal

from typing_extensions import Unpack
import pandas as pd
import numpy as np
from imu_weartime.weartime.base_weartime_detector import BaseWeartimeDetector
from imu_weartime.weartime.utils.ActivityCounts import ActivityCounts
from imu_weartime.weartime.utils.weartime_calc import (
    generate_weartime_list_from_minutes,
)
from imu_weartime.weartime.utils.weartime_calc import per_minute_counts
from mobgap.consts import GRAV_MS2


class WtdNishiyama(BaseWeartimeDetector):
    """
    Wear-time detection based on the activity-count–derived probability method
    described by Nishiyama et al. [1]

    Algorithm overview:
    1. Compute activity counts from the vector magnitude of triaxial acceleration.
    2. Aggregate activity counts to 1-minute resolution (sum per minute).
    3. Convert minute-level activity counts into a binary activity indicator
       (active = non-zero counts).
    4. For each minute:
       a) Compute upstream wear probability as the proportion of active minutes
          in the preceding `period_minutes` window.
       b) Compute downstream wear probability as the proportion of active minutes
          in the following `period_minutes` window.
    5. Classify a minute as non-wear if BOTH upstream and downstream wear
       probabilities are below the threshold `wear_prob`.
    6. Apply a continuity rule: any run of `cont_zeros` or more consecutive
       zero-activity minutes is classified as non-wear, overriding the
       probability-based decision.
    7. Convert minute-level wear/non-wear decisions back to sample-level
       intervals and compute total wear time.

    Notes:
    - Boundary windows (start/end of recording) use shorter windows implicitly.
    - Wear probability is defined as the fraction of non-zero activity minutes
      within a window. The value was not provided in the original publication and was subsequently estimated through fine-tuning on an independent sample.
    - The continuity rule acts as a hard override to suppress spurious wear
      detections during sustained inactivity.
    - All minutes, including the last partial minute, are evaluated; none are removed
    - The original algorithm uses proprieratary activity counts or METs, here we use a reverse engineered model for actigraph activity counts and then summed to the minute level.

    [1] Nishiyama N, Konda S, Ogasawara I, Nakata K (2024) A proof of concept for wear/non-wear classification using
     accelerometer data in daily activity recording: Synthetic algorithm leveraging probability and continuity of zero counts.
     PLoS ONE 19(10): e0309917. https://doi.org/10.1371/journal.pone.0309917
    """

    def __init__(
        self,
        *,
        period_minutes: int = 60,
        wear_prob: float = 0.5,
        cont_zeros: int = 10,
        zero_thresh: float = 1e-6,
        position: Literal["wrist", "lowback"] = "lowback",
    ) -> None:
        self.period_minutes = period_minutes
        self.cont_zeros = cont_zeros
        self.wear_prob = wear_prob
        self.zero_thresh = zero_thresh
        self.position = position

    def detect(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float = 100,
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.data_length = len(data)

        # Calculating norm
        cols = ["acc_is", "acc_ml", "acc_pa"]
        data_acc = self.data[cols]
        # Signal vector magnitude
        acc = np.linalg.norm(data_acc, axis=1)

        # Convert to g-units
        acc = acc / GRAV_MS2

        # Compute activity counts per second
        activity_counts = (
            ActivityCounts()
            .calculate(data=acc.copy(), sampling_rate=self.sampling_rate_hz)
            .activity_counts_
        )

        # Convert to per-minute counts
        activity_counts_pm = per_minute_counts(activity_counts)
        n_minutes = len(activity_counts_pm)

        # Identifying non zero periods
        is_active = (activity_counts_pm > self.zero_thresh).astype(int)

        wear_prob_up = np.zeros(n_minutes)
        wear_prob_down = np.zeros(n_minutes)

        for t in range(n_minutes):
            # Upstream window
            up_start = max(0, t - self.period_minutes)
            up_end = t
            if up_end > up_start:
                wear_prob_up[t] = is_active[up_start:up_end].mean()

            # Downstream window
            down_start = t + 1
            down_end = min(n_minutes, down_start + self.period_minutes)
            if down_end > down_start:
                wear_prob_down[t] = is_active[down_start:down_end].mean()

        # Average upstream and downstream probabilities
        wear_prob_avg = (wear_prob_up + wear_prob_down) / 2

        weartime_flags = np.ones(n_minutes, dtype=int)

        for t in range(n_minutes):
            if wear_prob_avg[t] < self.wear_prob:
                weartime_flags[t] = 0  # non-wear

        # Applying 10 min consecutive 0s check
        zero_runs = is_active == 0  # zero-count runs
        run_length = 0

        for t in range(n_minutes):
            if zero_runs[t]:
                run_length += 1
            else:
                # short non-wear → restore to wear
                if 0 < run_length <= self.cont_zeros:
                    weartime_flags[t - run_length : t] = 1
                run_length = 0

        # handle run at the end
        if 0 < run_length < self.cont_zeros:
            weartime_flags[-run_length:] = 1

        # Output
        self.weartime_list_ = generate_weartime_list_from_minutes(
            weartime_flags, sampling_rate=int(sampling_rate_hz)
        )
        # Clip end to actual data length
        self.weartime_list_["end"] = self.weartime_list_["end"].clip(
            upper=self.data_length
        )
        self.total_weartime_samples_ = (
            self.weartime_list_["end"] - self.weartime_list_["start"]
        ).sum()
        self.total_weartime_minutes_ = self.total_weartime_samples_ / (
            60 * self.sampling_rate_hz
        )
        self.total_weartime_hours_ = self.total_weartime_samples_ / (
            3600 * self.sampling_rate_hz
        )
        return self
