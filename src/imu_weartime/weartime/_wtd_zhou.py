from typing_extensions import Self
from typing import Any, Unpack, Literal
import pandas as pd
import numpy as np
from scipy.ndimage import uniform_filter1d
from imu_weartime.weartime.base_weartime_detector import BaseWeartimeDetector
from imu_weartime.weartime.utils.weartime_calc import generate_weartime_list_from_seconds
from mobgap.consts import GRAV_MS2

class WtdZhou(BaseWeartimeDetector):
    """
    Wear / non-wear time detection based on the temperature-only and
    combined temperature–acceleration (CTA) algorithms proposed by
    Zhou et al. (2015) [1].

    The detector operates at a per-second temporal resolution using
    sliding windows with a fixed duration and a 1-second update step.

    Algorithm variants
    ------------------
    Two algorithmic variants are supported:

    1. Temperature-only model ("temp")
       The model relies exclusively on temperature dynamics.

       Steps:
       - Input: continuous temperature time series.
       - Temperature is smoothed using a moving average with a window
         length of ``window_min`` minutes and a step size of 1 second.
       - At each second ``t``, two window-level statistics are computed:
            ``Tt``: mean temperature in the current window.
            ``Tt_ws``: mean temperature in the immediately preceding window
             of equal duration.
       - Wear / non-wear classification is performed according to the
         following rules:
           1. If ``Tt > T0`` → classify as wear.
           2. If ``Tt ≤ T0`` and ``Tt > Tt_ws`` → classify as wear
              (increasing temperature trend).
           3. If ``Tt ≤ T0`` and ``Tt < Tt_ws`` → classify as non-wear
              (decreasing temperature trend).
           4. If ``Tt == Tt_ws`` → retain the previous classification.

       The temperature threshold ``T0`` is dataset-dependent and should
       be tuned empirically.

    2. Combined temperature + acceleration model ("cta")
       This model extends the temperature-only approach by incorporating
       acceleration variability.

       Steps:
       - Inputs: temperature time series and tri-axial acceleration.
       - Temperature preprocessing and windowing are identical to the
         temperature-only model.
       - At each second ``t``, compute:
           * ``Tt``: mean smoothed temperature in the current window.
           * ``SD_acc``: standard deviation of acceleration within the
             same window, computed separately for each axis.
       - Classification rules are applied in the following order:
           1. If ``Tt < T0`` and ``SD_acc < 13 mg`` on all axes
              → classify as non-wear.
           2. Else if ``Tt ≥ T0`` → classify as wear.
           3. Else (``Tt < T0`` and acceleration variability is high)
              → fall back to the temperature trend rules described in the
              temperature-only model.

    Outputs
    -------
    - Per-second binary wear/non-wear decisions.
    - Wear time bouts expressed in sample indices.
    - Total wear time reported in seconds, minutes, hours, and samples.

    Notes
    -----
    - Some data removal due to rounding happens at the second level because we keep the last integer only (minor impact).
    - All decisions are made at 1-second resolution; minute- and
      hour-level summaries are derived post hoc.
    - The acceleration standard deviation threshold of 13 mg follows
      the original paper.
    - This implementation adheres to the algorithmic description in
      Zhou et al. (2015); minor differences may exist relative to other
      software packages due to interpretation or implementation details.
    - Max 1 min removed due to per-minute aggregation
    - Version to assess: "temp" and "cta"

    [1] Zhou S-M, Hill RA, Morgan K, et al. Classification of accelerometer wear and non-wear events in seconds for
    monitoring freeliving physical activity. BMJ Open 2015;5:e007447. doi:10.1136/bmjopen-2014- 007447
    """

    def __init__(
        self,
        *,
        version: Literal["temp", "cta"] = "cta",
        window_min: int = 1,
        t0: float = 26.0,
        acc_sd_thresh_mg: float = 13.0,
        position: Literal['wrist', 'lowback'] = 'lowback'
    ) -> None:
        self.version = version
        self.window_min = window_min
        self.t0 = t0
        self.acc_sd_thresh_mg = acc_sd_thresh_mg
        self.acc_sd_thresh_g = self.acc_sd_thresh_mg / 1000.0
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

        # Temperature preprocessing
        if "temperature" not in data.columns:
            raise ValueError("Temperature column ('temperature') is required for WtdZhou.")

        temp = data["temperature"].to_numpy()

        window_samples = int(self.window_min * 60 * self.sampling_rate_hz)

        # Moving average smoothing (1-min window, 1-s step)
        temp_smooth = uniform_filter1d(
            temp.astype(float),
            size=window_samples,
            mode="nearest"
        )

        # Acceleration preprocessing (CTA only)
        if self.version == "cta":
            acc = data[["acc_is", "acc_ml", "acc_pa"]].to_numpy() / GRAV_MS2

        # Per-second classification
        n_samples = len(data)
        n_seconds = n_samples // self.sampling_rate_hz

        wear_sec = np.ones(n_seconds, dtype=int)

        # handling the first window for both versions (temp: temperature-only comparison with t0, cta: sd of acc + temp comparison with t0)
        partial_window_sec = min(window_samples // self.sampling_rate_hz, n_seconds)
        for i in range(partial_window_sec):
            end_sample = (i + 1) * self.sampling_rate_hz

            # Partial mean temperature
            Tt_partial = temp_smooth[:end_sample].mean()

            if self.version == "temp":
                # Temperature-only classification
                wear_sec[i] = 1 if Tt_partial > self.t0 else 0

            elif self.version == "cta":
                # Partial acceleration SD
                acc_win = acc[:end_sample]
                sd_acc = np.std(acc_win, axis=0)

                # CTA classification for partial window
                if Tt_partial >= self.t0:
                    wear_sec[i] = 1
                elif Tt_partial < self.t0 and np.all(sd_acc < self.acc_sd_thresh_g):
                    wear_sec[i] = 0
                else:
                    # fallback to temperature-only (trend) rules not possible for first partial window
                    # keep wear_sec[i] as initially assigned by temperature
                    wear_sec[i] = 1 if Tt_partial > self.t0 else 0

        # Main loop for all remaining windows
        for i in range(partial_window_sec, n_seconds):
            start_sample = i * self.sampling_rate_hz - window_samples
            end_sample = (i + 1) * self.sampling_rate_hz

            # Temperature window
            Tt = temp_smooth[start_sample:end_sample].mean()

            # Previous window (for trend)
            prev_end = end_sample - window_samples
            prev_start = prev_end - window_samples
            if prev_start < 0:
                continue
            Tt_ws = temp_smooth[prev_start:prev_end].mean()

            if self.version == "temp":
                # Temperature-only rules
                if Tt > self.t0:
                    wear_sec[i] = 1
                elif Tt <= self.t0 and Tt > Tt_ws:
                    wear_sec[i] = 1
                elif Tt <= self.t0 and Tt < Tt_ws:
                    wear_sec[i] = 0
                else:  # Tt == Tt_ws
                    wear_sec[i] = wear_sec[i - 1]

            elif self.version == "cta":
                # Full-window acceleration SD
                start_acc = i * self.sampling_rate_hz - window_samples
                end_acc = (i + 1) * self.sampling_rate_hz
                acc_win = acc[start_acc:end_acc]
                sd_acc = np.std(acc_win, axis=0)

                if Tt < self.t0 and np.all(sd_acc < self.acc_sd_thresh_g):
                    wear_sec[i] = 0
                elif Tt >= self.t0:
                    wear_sec[i] = 1
                else:  # Tt < T0 and SD_acc >= threshold
                    # Fall back to temperature trend rules
                    if Tt > Tt_ws:
                        wear_sec[i] = 1
                    elif Tt < Tt_ws:
                        wear_sec[i] = 0
                    else:  # Tt == Tt_ws
                        wear_sec[i] = wear_sec[i - 1]

        # Outputs
        self.weartime_list_ = generate_weartime_list_from_seconds(
            wear_sec, sampling_rate=int(self.sampling_rate_hz)
        )
        self.weartime_list_["end"] = self.weartime_list_["end"].clip(
            upper=self.data_length
        )

        self.total_weartime_samples_ = (self.weartime_list_['end'] - self.weartime_list_['start']).sum()
        self.total_weartime_minutes_ = self.total_weartime_samples_ / (60 * sampling_rate_hz)
        self.total_weartime_hours_ = self.total_weartime_samples_ / (3600 * sampling_rate_hz)
        return self
