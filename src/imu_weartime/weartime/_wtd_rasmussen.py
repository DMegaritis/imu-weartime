from typing_extensions import Self
from typing import Any, Unpack, Literal
import pandas as pd
import numpy as np
from imu_weartime.weartime.base_weartime_detector import BaseWeartimeDetector
from imu_weartime.weartime.utils.weartime_calc import generate_weartime_list_from_minutes
from mobgap.consts import GRAV_MS2


class WtdRasmussen(BaseWeartimeDetector):
    """
   Wear-time detector implementing a simplified Rasmussen [1] algorithm.

    Detection is based on two complementary signals: accelerometry and temperature.

    Non-wear rules implemented:
    1. Quiet periods longer than 120 minutes are always non-wear.
    2. Quiet periods of 45–120 minutes are non-wear if the average temperature
       during that period is below a specified threshold (default 26°C).

    Accelerometry details:
    - Vector magnitude of tri-axial acceleration (axes: IS, ML, PA) is computed.
    - Values are converted to g-units.
    - A per-sample threshold (default 20 mg) is applied: all samples within a minute
      must be below this threshold for the minute to be considered 'quiet'.
    - Full-minute aggregation ensures consistency with other wear-time algorithms.

    Temperature details:
    - Average temperature is calculated per minute over the same period.
    - Threshold is currently fixed (default 26°C) based on literature and device calibration.
    - Mid-length quiet periods (45–120 min) are flagged as non-wear if mean temperature
      is below the threshold.

    Notes:
    - Full-minute aggregation is used to ensure consistency with other algorithms.
    - For this validation project, we removed the non-wear rule that classified quiet periods of 10–45 minutes with low temperature as non-wear,
    as it also required the quiet period to end during waking hours (06:00–22:00) which is not consistent with how we assess wear time.
    Validation will be done only during the Mob-D waking hours.
    - The temperature threshold is currently fixed for all participants.
      In practice, this is not optimal due to differences in device properties, participant behaviour,
      and ambient conditions [see December report of SUSTAIN project]. We use a temp fro mthe literture (26C) as we have shown that algorithms with fixed threshold are not optimal.
    - Threshold fine-tuning: The low-activity threshold was empirically tuned
      using verified quiet periods where the device is resting stationary. The 75th percentile
      of the vector magnitude over these stationary recordings (without filtering out gravity)
      is used to account for sensor noise and the constant gravity component. This ensures
      that stationary periods are correctly identified as non-wear.
    - Max 59 seconds (last partial minute) removed due to per-minute aggregation

    [1] Rasmussen, M.G.B., Pedersen, J., Olesen, L.G. et al. Short-term efficacy of reducing screen media use on physical activity,
    sleep, and physiological stress in families with children aged 4–14: study protocol for the SCREENS randomized controlled trial.
    BMC Public Health 20, 380 (2020). https://doi.org/10.1186/s12889-020-8458-6
    """

    def __init__(
        self,
        *,
        low_activity_thresh_mg: float = 107,
        temp_thresh: float = 26.0,
        position: Literal['wrist', 'lowback'] = 'lowback'
    ) -> None:
        self.low_activity_thresh_mg = low_activity_thresh_mg
        self.low_activity_thresh = low_activity_thresh_mg / 1000
        self.temp_thresh = temp_thresh
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

        # Binary array on the per sample basis False=acc>20mg and True=acc<20mg
        quiet_mask = acc < self.low_activity_thresh

        # Turning to minute mask
        samples_per_min = int(60 * self.sampling_rate_hz)

        # Truncate to full minutes
        n_full_minutes = len(quiet_mask) // samples_per_min
        quiet_mask_trunc = quiet_mask[:n_full_minutes * samples_per_min]

        # Reshape: (n_minutes, samples_per_min)
        quiet_mask_min = quiet_mask_trunc.reshape(n_full_minutes, samples_per_min)

        # Per-minute quietness: 1 if ALL samples are < 20 mg
        quiet_min = np.all(quiet_mask_min, axis=1).astype(int)

        quiet = quiet_min == 1

        # Label consecutive runs
        groups = (quiet != np.roll(quiet, 1)).cumsum()

        # Compute run lengths
        df = pd.DataFrame({"quiet": quiet, "group": groups})
        quiet_lengths = df.groupby("group")["quiet"].transform("size")

        # Rule #1: quiet periods longer than 120 min are non-wear
        nonwear_long = quiet & (quiet_lengths >= 120)

        # Rule #2: quiet periods 45–120 min, non-wear if avg temperature < threshold
        temp = data["temperature"].to_numpy()
        temp_trunc = temp[:n_full_minutes * samples_per_min]

        # Aggregate temperature per minute first
        temp_min = temp_trunc[:n_full_minutes * samples_per_min].reshape(n_full_minutes, samples_per_min).mean(axis=1)

        nonwear_mid = np.zeros_like(quiet, dtype=bool)

        for grp, subdf in df.groupby("group"):
            if subdf["quiet"].all():
                length = len(subdf)
                if 45 <= length < 120:
                    avg_temp = temp_min[subdf.index].mean()
                    if avg_temp < self.temp_thresh:
                        nonwear_mid[subdf.index] = True

        # Combine long and mid non-wear periods
        nonwear_min = nonwear_long | nonwear_mid
        wear_min = (~nonwear_min).astype(int)

        # Generate wear time list scaled to sample indices
        self.weartime_list_ = generate_weartime_list_from_minutes(
            wear_min.to_numpy(), sampling_rate=int(sampling_rate_hz)
        )
        # Clip end to actual data length
        self.weartime_list_["end"] = self.weartime_list_["end"].clip(upper=self.data_length)
        self.total_weartime_samples_ = (self.weartime_list_['end'] - self.weartime_list_['start']).sum()
        self.total_weartime_minutes_ = self.total_weartime_samples_ / (60 * sampling_rate_hz)
        self.total_weartime_hours_ = self.total_weartime_samples_ / (3600 * sampling_rate_hz)

        return self
