from typing_extensions import Self
from typing import Any, Unpack, Literal
import pandas as pd
from imu_weartime.weartime.base_weartime_detector import BaseWeartimeDetector
from imu_weartime.weartime.utils.weartime_calc import (
    generate_weartime_list_from_minutes,
)
from multigait.ICD.ICD4 import ZijlstraIC


class WtdKing(BaseWeartimeDetector):
    """
    Wear-time detection using the algorithm from King et al. (2011) [1].

    This class implements a minute-level non-wear detection algorithm for
    wrist- or lowback-worn devices based on step detection:

    Algorithm overview:
    1. Detect initial contacts (steps) using device-specific ICD algorithm:
       - 'lowback' → IcdIonescu
       - 'wrist'   → IcdShin
    2. Convert detected samples to seconds and aggregate into minute bins.
    3. Count steps per minute and fill missing minutes with zeros.
    4. Identify inactive minutes (zero steps) and group consecutive inactive periods.
    5. Apply a non-wear threshold (`period_minutes`) on group length:
       - Any consecutive zero-step period longer than the threshold is flagged as non-wear.
       - Output is a binary per-minute array: 1 = wear, 0 = non-wear.
    6. Convert per-minute wear flags to sample-level intervals using
       `generate_weartime_list_from_minutes`.
    7. Compute summary statistics: total wear time in samples, minutes, and hours.

    Note:
    - For this implementation, **all quiet periods are treated as non-wear**.
      The original ICD model considers quiet periods during non-waking hours
      as sleep and does not flag them as non-wear. This simplification assumes
      that only waking hours are provided as input for testing and using this model.
    -We used the ZilstraIC algorithm for step detection, which is a well-established method for detecting initial contacts (steps) from accelerometer data. The choice of algorithm is based on the absense of ICs in quiet periods, whereas mobgap algorithms would detecs ICs most likely due to filtereing and noise.
    - All minutes containing ICs, including the final partial minute with ICs, are evaluated; minutes after the last IC are not assessed

    [1] King WC, Li J, Leishear K, Mitchell JE, Belle SH. Determining activity monitor wear time: an influential decision rule.
    J Phys Act Health. 2011 May;8(4):566-80. doi: 10.1123/jpah.8.4.566. PMID: 21597130; PMCID: PMC3711095.
    """

    def __init__(
        self,
        *,
        position: Literal["wrist", "lowback"] = "lowback",
        period_minutes: Literal[60, 90, 120, 150] = 120,
    ) -> None:
        self.period_minutes = period_minutes
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
        if self.position == "lowback":
            ics = (
                ZijlstraIC(version="original_lowback")
                .detect(self.data, sampling_rate_hz=self.sampling_rate_hz)
                .ic_list_
            )
        elif self.position == "wrist":
            ics = (
                ZijlstraIC(version="wrist")
                .detect(self.data, sampling_rate_hz=self.sampling_rate_hz)
                .ic_list_
            )

        # Convert sample indices to seconds and then to minute bins
        ics["time_sec"] = ics["ic"] / self.sampling_rate_hz
        ics["minute"] = (ics["time_sec"] // 60).astype(int)

        # Count steps per minute and fill missing minutes with zeros
        steps_per_minute = (
            ics.groupby("minute")
            .size()
            .rename("steps")
            .reindex(
                pd.RangeIndex(ics["minute"].min(), ics["minute"].max() + 1),
                fill_value=0,
            )
            .rename_axis("minute")
            .reset_index()
        )

        # Non-wear detection
        # Binary inactive column
        steps_per_minute["inactive"] = steps_per_minute["steps"] == 0

        # Label consecutive groups
        steps_per_minute["group"] = (
            steps_per_minute["inactive"] != steps_per_minute["inactive"].shift()
        ).cumsum()

        # Group sizes
        steps_per_minute["group_size"] = steps_per_minute.groupby("group")[
            "inactive"
        ].transform("size")

        # Vectorized non-wear flag per minute: 1 = wear, 0 = non-wear
        weartime_flags = (
            ~(
                steps_per_minute["inactive"]
                & (steps_per_minute["group_size"] >= self.period_minutes)
            )
        ).astype(int)

        # Generate wear time list scaled to sample indices
        self.weartime_list_ = generate_weartime_list_from_minutes(
            weartime_flags.to_numpy(), sampling_rate=int(sampling_rate_hz)
        )
        # Clip end to actual data length
        self.weartime_list_["end"] = self.weartime_list_["end"].clip(
            upper=self.data_length
        )
        self.total_weartime_samples_ = (
            self.weartime_list_["end"] - self.weartime_list_["start"]
        ).sum()
        self.total_weartime_minutes_ = self.total_weartime_samples_ / (
            60 * sampling_rate_hz
        )
        self.total_weartime_hours_ = self.total_weartime_samples_ / (
            3600 * sampling_rate_hz
        )

        return self
