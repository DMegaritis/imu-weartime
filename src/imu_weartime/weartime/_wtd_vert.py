from typing_extensions import Self
from typing import Any, Literal

from typing_extensions import Unpack
import pandas as pd
import numpy as np
from imu_weartime.weartime.base_weartime_detector import BaseWeartimeDetector
from imu_weartime.weartime.utils.weartime_calc import (
    generate_weartime_list_from_samples,
)
from mobgap.consts import GRAV_MS2
from mobgap.data_transform import (
    ButterworthFilter,
    Resample,
    chain_transformers,
)


def _fwd_std(x: np.ndarray, window: int) -> pd.Series:
    """Forward-looking rolling standard deviation."""
    return pd.Series(x)[::-1].rolling(window).std()[::-1]


class WtdVert(BaseWeartimeDetector):
    """
    Wear-time detection for wrist accelerometer using tri-axial acceleration
    and temperature based on the Vert et al. (2022) algorithm [1].

    Features
    --------
    - Tri-axial acceleration: rolling 1-min SD (forward & backward)
    - Temperature: low-pass filtered, resampled to 0.25 Hz
    - 5-minute aggregation for quiet axes & smoothed temperature

    State machine
    -------------
    - START non-wear: either temp ΔT <= temp_dec_roc AND quiet axes ≥ 2 OR temp < low_temperature_cutoff
    - END non-wear: ΔT >= temp_inc_roc OR temp > high_temperature_cutoff, AND quiet axes <= 0.5
    - ≥5-min bout enforcement ensures short fluctuations are ignored

    Outputs
    -------
    - `weartime_list_`: DataFrame of wear-time bouts in sample indices
    - `total_weartime_samples_`, `_seconds_`, `_minutes_`, `_hours_`: cumulative wear time

    Notes
    -------
    - Max 4 sec removed due to per-minute aggregation
    - Test the single version.

    [1] Vert A, Weber KS, Thai V, Turner E, Beyer KB, Cornish BF, Godkin FE, Wong C, McIlroy WE, Van Ooteghem K.
    Detecting accelerometer non-wear periods using change in acceleration combined with rate-of-change in temperature.
    BMC Med Res Methodol. 2022 May 20;22(1):147. doi: 10.1186/s12874-022-01633-6. PMID: 35596151; PMCID: PMC9123693.
    """

    def __init__(
        self,
        *,
        window_min: int = 1,
        low_temperature_cutoff: float = 26.0,
        high_temperature_cutoff: float = 30.0,
        temp_dec_roc: float = -0.2,
        temp_inc_roc: float = 0.1,
        acc_sd_thresh_mg: float = 8.0,
        _target_sampling_rate_hz: float = 0.25,
        position: Literal["wrist", "lowback"] = "lowback",
    ) -> None:
        self.window_min = window_min
        self.low_temperature_cutoff = low_temperature_cutoff
        self.high_temperature_cutoff = high_temperature_cutoff
        self.temp_dec_roc = temp_dec_roc
        self.temp_inc_roc = temp_inc_roc
        self.acc_sd_thresh_mg = acc_sd_thresh_mg
        self.acc_sd_thresh_g = self.acc_sd_thresh_mg / 1000.0
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

        # Acceleration (g)
        acc = data[["acc_is", "acc_ml", "acc_pa"]].to_numpy() / GRAV_MS2

        # Temperature preprocessing: resampling to the original algorithm sample rate of the method + low-pass filtering
        temp = data["temperature"].to_numpy()
        temp_ds = chain_transformers(
            temp,
            [("resample", Resample(self._target_sampling_rate_hz))],
            sampling_rate_hz=self.sampling_rate_hz,
        )

        temp_filt = chain_transformers(
            temp_ds,
            [
                (
                    "butter",
                    ButterworthFilter(
                        order=2, cutoff_freq_hz=0.005, filter_type="lowpass"
                    ),
                )
            ],
            sampling_rate_hz=self._target_sampling_rate_hz,
        )

        temp = pd.Series(temp_filt).reset_index(drop=True)

        # Acceleration rolling SD (1-min, forward & backward)
        win_acc = int(60 * sampling_rate_hz)

        std_df = pd.DataFrame(
            {
                "is_fwd": _fwd_std(acc[:, 0], win_acc),
                "ml_fwd": _fwd_std(acc[:, 1], win_acc),
                "pa_fwd": _fwd_std(acc[:, 2], win_acc),
                "is_back": pd.Series(acc[:, 0]).rolling(win_acc).std(),
                "ml_back": pd.Series(acc[:, 1]).rolling(win_acc).std(),
                "pa_back": pd.Series(acc[:, 2]).rolling(win_acc).std(),
            }
        )

        # Quiet axes count
        std_df["num_axes_fwd"] = (
            std_df[["is_fwd", "ml_fwd", "pa_fwd"]] < self.acc_sd_thresh_g
        ).sum(axis=1)

        std_df["num_axes_back"] = (
            std_df[["is_back", "ml_back", "pa_back"]] < self.acc_sd_thresh_g
        ).sum(axis=1)

        # Downsample acc features to temp timebase
        ds = int(self.sampling_rate_hz / self._target_sampling_rate_hz)
        std_ds = std_df.iloc[::ds].reset_index(drop=True)
        std_ds = std_ds.iloc[: len(temp)]

        # 5-minute aggregations
        win_5 = int(5 * 60 * self._target_sampling_rate_hz)

        std_ds["quiet_fwd_5m"] = (
            (std_ds["num_axes_fwd"] >= 2)[::-1].rolling(win_5).mean()[::-1]
        )
        std_ds["quiet_back_5m"] = (
            (std_ds["num_axes_back"] >= 2)[::-1].rolling(win_5).mean()[::-1]
        )

        # 5-minute smoothed temperature → then compute ΔT (°C/min)
        win_5 = int(5 * self._target_sampling_rate_hz)
        temp_smooth_5m = temp[::-1].rolling(win_5, min_periods=1).mean()[::-1]
        # temp rate-of-change over the 5-min window
        temp_roc_5m = (temp_smooth_5m - temp_smooth_5m.shift(win_5)) / (5.0)  # °C/min
        temp_roc_5m = temp_roc_5m.fillna(0.0)

        # State machine with ≥5-min bout enforcement
        state = 0
        nw_state = np.zeros(len(temp), dtype=int)
        last_start = -np.inf

        for i in range(len(temp)):
            # START non-wear
            if state == 0:
                start_roc_path = (
                    temp_roc_5m.iloc[i] <= self.temp_dec_roc
                    and temp.iloc[i] < self.high_temperature_cutoff
                    and std_ds.loc[i, "quiet_fwd_5m"] >= 0.9
                )

                start_abs_path = (
                    temp.iloc[i] < self.low_temperature_cutoff
                    and std_ds.loc[i, "quiet_fwd_5m"] >= 0.9
                )

                if start_roc_path or start_abs_path:
                    state = 1
                    last_start = i

            # END non-wear
            else:
                if i - last_start >= win_5:
                    end_roc_path = temp_roc_5m.iloc[i] >= self.temp_inc_roc
                    end_abs_path = temp.iloc[i] > self.high_temperature_cutoff

                    if (end_roc_path or end_abs_path) and std_ds.loc[
                        i, "quiet_back_5m"
                    ] <= 0.5:
                        state = 0

            nw_state[i] = state

        # Upsample to accelerometer resolution
        nw_state_acc = np.repeat(nw_state, ds)[: self.data_length]
        self.weartime_list_ = generate_weartime_list_from_samples(nw_state_acc)

        # Binary wear flag: 1 = wear, 0 = non-wear
        wear_samples = np.sum(nw_state_acc == 1)
        self.total_weartime_samples_ = wear_samples
        self.total_weartime_seconds_ = wear_samples / self.sampling_rate_hz
        self.total_weartime_minutes_ = self.total_weartime_seconds_ / 60.0
        self.total_weartime_hours_ = self.total_weartime_seconds_ / 3600.0

        return self
