from typing_extensions import Self
from typing import Any, Unpack, Literal
import pandas as pd
import numpy as np
from imu_weartime.weartime.base_weartime_detector import BaseWeartimeDetector
from imu_weartime.weartime.utils.ActivityCounts import ActivityCounts
from imu_weartime.weartime.utils.weartime_calc import generate_weartime_list_from_minutes
from imu_weartime.weartime.utils.weartime_calc import per_minute_counts
from mobgap.consts import GRAV_MS2


class WtdHecht(BaseWeartimeDetector):
    """
     Hecht Wear Time Detector [1].

    This detector implements the minute-level wear time detection algorithm
    described in Hecht et al. The method uses vector magnitude units (VMU)
    aggregated per minute and applies a three-condition logic:

    1. Current minute VMU·min⁻¹ > threshold
    2. At least two of the following 20 minutes VMU·min⁻¹ > threshold
    3. At least two of the preceding 20 minutes VMU·min⁻¹ > threshold

    A minute is considered "worn" if **at least two of the three conditions are met**.

    Attributes
    ----------
    low_activity_thresh : float
        Threshold for low activity in VMU·min⁻¹ to distinguish wear vs non-wear.
    weartime_list_ : pd.DataFrame
        List of start and end indices of continuous wear periods (in samples).
    total_weartime_samples_ : int
        Total number of samples classified as wear time.
    total_weartime_minutes_ : float
        Total wear time in minutes.
    total_weartime_hours_ : float
        Total wear time in hours.

    Notes:
    - In this versions we use the activity counts of the acceleration norm which is then summed into minute bins. This is different from the original algorithm which uses proprietary software.
    - For deriving the low activity threshold, we followed a similar approach as the original method
     who empirically characterised the noise floor using stationary reference recordings processed through the full count-generation pipeline,
     we likewise derive a non-wear threshold by analysing confirmed immobile periods specific to our own hardware–software implementation.
     We observed that the activity counts algorithm returns 0 during periods of complete device stability; to account for minor signal
     fluctuations and avoid underestimating low activity, we set the threshold slightly above 0 (1e-6).
     - All data are included; any leftover seconds at the end of the data (last partial minute) are counted as a
    final partial minute are assessed with the same rules as fll minutes

    [1] Hecht A, Ma S, Porszasz J, Casaburi R; COPD Clinical Research Network. Methodology for using long-term
    accelerometry monitoring to describe daily activity patterns in COPD. COPD. 2009 Apr;6(2):121-9.
    doi: 10.1080/15412550902755044. PMID: 19378225; PMCID: PMC2862250.
    """

    def __init__(
        self,
        *,
        low_activity_thresh: float = 1e-6,
        position: Literal['wrist', 'lowback'] = 'lowback',
    ) -> None:
        self.low_activity_thresh = low_activity_thresh
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

        # Step detection based on device version
        cols = ['acc_is', 'acc_ml', 'acc_pa']
        data_acc = self.data[cols]
        # Signal vector magnitude
        acc = np.linalg.norm(data_acc, axis=1)

        # Convert to g-units
        acc = acc / GRAV_MS2

        # Compute activity counts per second
        activity_counts = ActivityCounts().calculate(data=acc.copy(),
                                                     sampling_rate=self.sampling_rate_hz).activity_counts_

        # Convert to per-minute counts
        activity_counts_pm = per_minute_counts(activity_counts)

        wear_mask = np.zeros_like(activity_counts_pm, dtype=bool)

        for i in range(len(activity_counts_pm)):
            vm = activity_counts_pm[i]

            # Check conditions
            cond1 = vm > self.low_activity_thresh

            prev_window = activity_counts_pm[max(0, i - 20):i]
            next_window = activity_counts_pm[i + 1:min(len(activity_counts_pm), i + 21)]

            cond2 = np.sum(next_window > self.low_activity_thresh) >= 2
            cond3 = np.sum(prev_window > self.low_activity_thresh) >= 2

            # At least two conditions true → worn
            conditions = [cond1, cond2, cond3]
            wear_mask[i] = np.sum(conditions) >= 2

        wear_mask = wear_mask.astype(int)

        # Outputs
        self.weartime_list_ = generate_weartime_list_from_minutes(
            wear_mask, sampling_rate=int(sampling_rate_hz)
        )
        # Clip end to actual data length
        self.weartime_list_["end"] = self.weartime_list_["end"].clip(upper=self.data_length)
        self.total_weartime_samples_ = (self.weartime_list_['end'] - self.weartime_list_['start']).sum()
        self.total_weartime_minutes_ = self.total_weartime_samples_ / (60 * sampling_rate_hz)
        self.total_weartime_hours_ = self.total_weartime_samples_ / (3600 * sampling_rate_hz)

        return self
