from typing_extensions import Self
from typing import Any, Literal

from typing_extensions import Unpack
import pandas as pd
from imu_weartime.weartime.base_weartime_detector import BaseWeartimeDetector
from imu_weartime.weartime.utils.weartime_calc import (
    generate_weartime_list_from_seconds,
)
from mobgap.data_transform import (
    Resample,
    chain_transformers,
)


class WtdDuncan(BaseWeartimeDetector):
    """
    Wear-time detection based on single-sensor temperature data according to the method by Duncan et al. (2018) [1]
    using a dynamic threshold and hysteresis to prevent rapid toggling due to small temperature fluctuations.

    The algorithm follows these steps:
    1. Downsample temperature data to 1 Hz.
    2. Split temperature into values above and below 18°C to compute group means.
    3. Define the dynamic threshold as the midpoint between the two means.
       - If no data exists in one of the groups, the median of the temperature series is used.
    4. Classify each second as candidate wear (1) or non-wear (0) based on the threshold.
    5. Apply hysteresis sequentially:
       - If previous state is wear, switch to non-wear only if temperature < threshold - 2°C.
       - If previous state is non-wear, switch to wear only if temperature > threshold + 2°C.
    6. Generate wear-time bouts in sample indices at the original sampling rate.

    Attributes
    ----------
    _target_sampling_rate_hz : int
        Target sampling rate (Hz) for downsampling temperature data (default 1 Hz).
    weartime_list_ : pd.DataFrame
        Wear-time bouts with 'start' and 'end' sample indices after detection.
    total_weartime_samples_ : int
        Total number of samples classified as wear.
    total_weartime_minutes_ : float
        Total wear time in minutes.
    total_weartime_hours_ : float
        Total wear time in hours.

    Notes:
    - All seconds are evaluated; no truncation occurs

    [1] Duncan S, Stewart T, Mackay L, Neville J, Narayanan A, Walker C, Berry S, Morton S. Wear-Time Compliance
    with a Dual-Accelerometer System for Capturing 24-h Behavioural Profiles in Children and Adults.
    Int J Environ Res Public Health. 2018 Jun 21;15(7):1296. doi: 10.3390/ijerph15071296. PMID: 29933548; PMCID: PMC6069278.
    """

    def __init__(
        self,
        *,
        _target_sampling_rate_hz: int = 1,
        position: Literal["wrist", "lowback"] = "lowback",
    ) -> None:
        self._target_sampling_rate_hz = _target_sampling_rate_hz
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

        # Temperature downsizing
        temp = data["temperature"].to_numpy()
        temp_ds = chain_transformers(
            temp,
            [("resample", Resample(self._target_sampling_rate_hz))],
            sampling_rate_hz=self.sampling_rate_hz,
        )

        # Convert to pandas Series for convenience
        temp_ds = pd.Series(temp_ds).reset_index(drop=True)

        # Split into two groups
        below_18 = temp_ds[temp_ds < 18]
        above_18 = temp_ds[temp_ds >= 18]

        # Calculate means
        mean_below = below_18.mean()
        mean_above = above_18.mean()

        # Threshold = midpoint
        threshold = (mean_below + mean_above) / 2

        if pd.isna(threshold):
            threshold = temp_ds.median()

        # Initial candidate: 1 = wear, 0 = non-wear
        candidate = (temp_ds >= threshold).astype(int).to_numpy()

        # Apply hysteresis sequentially
        wear_hyst = candidate.copy()
        for i in range(1, len(wear_hyst)):
            if wear_hyst[i - 1] == 1:  # previous state is wear
                if temp_ds[i] < threshold - 2:
                    wear_hyst[i] = 0
                else:
                    wear_hyst[i] = 1
            else:  # previous state is non-wear
                if temp_ds[i] > threshold + 2:
                    wear_hyst[i] = 1
                else:
                    wear_hyst[i] = 0

        # Generate wear periods (weartime list) using helper
        self.weartime_list_ = generate_weartime_list_from_seconds(
            wear_hyst, sampling_rate=int(self.sampling_rate_hz)
        )
        # Clip end to actual data length
        self.weartime_list_["end"] = self.weartime_list_["end"].clip(
            upper=len(self.data)
        )
        # Summary stats
        self.total_weartime_samples_ = (
            self.weartime_list_["end"] - self.weartime_list_["start"]
        ).sum()
        self.total_weartime_minutes_ = self.total_weartime_samples_ / 60
        self.total_weartime_hours_ = self.total_weartime_samples_ / 3600

        return self
