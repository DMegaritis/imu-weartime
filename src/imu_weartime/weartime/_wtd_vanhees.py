import pandas as pd
import numpy as np
from typing import Any, Unpack, Literal
from typing_extensions import Self
from imu_weartime.weartime.base_weartime_detector import BaseWeartimeDetector
from imu_weartime.weartime.utils.weartime_calc import generate_weartime_list_from_minutes
from mobgap.consts import GRAV_MS2


class WtdVanHees(BaseWeartimeDetector):
    """
    Implementation of the Van Hees non-wear detection algorithm [1].

    Non-wear is detected using raw acceleration variability:
    - 60-min overlapping windows (15-min step)
    - Non-wear if >=2 axes have low SD or low range
    - Iterative reclassification of short wear periods between non-wear

    Notes:
    -Validation with 30 and 60 minute windows (two publications) [1, 2]
    - Max 59 seconds removed due to per-minute aggregation from the classification logic. The final partial minute takes
    the value of the last full minute classification.

    [1] van Hees VT, Gorzelniak L, Dean León EC, Eder M, Pias M, et al. (2013) Separating Movement and Gravity Components in an Acceleration
    Signal and Implications for the Assessment of Human Daily Physical Activity. PLOS ONE 8(4): e61691. https://doi.org/10.1371/journal.pone.0061691

    [2] van Hees VT, Renström F, Wright A, Gradmark A, Catt M, Chen KY, Löf M, Bluck L, Pomeroy J, Wareham NJ, Ekelund U, Brage S, Franks PW.
    Estimation of daily energy expenditure in pregnant and non-pregnant women using a wrist-worn tri-axial accelerometer. PLoS One. 2011;6(7):e22922.
    doi: 10.1371/journal.pone.0022922. Epub 2011 Jul 29. PMID: 21829556; PMCID: PMC3146494.

    """

    def __init__(
        self,
        *,
        window_min: int = 60,
        step_min: int = 15,
        std_thresh_mg: float = 13.0,
        range_thresh_mg: float = 50.0,
        num_axes_required: int = 2,
        position: Literal['wrist', 'lowback'] = 'lowback'
    ) -> None:
        self.window_min = window_min
        self.step_min = step_min
        self.range_thresh_mg = range_thresh_mg
        self.std_thresh_mg = std_thresh_mg
        self.std_thresh_g = self.std_thresh_mg / 1000
        self.range_thresh_g = self.range_thresh_mg / 1000
        self.num_axes_required = num_axes_required
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

        # Load raw acceleration in g-units
        acc = data[["acc_is", "acc_ml", "acc_pa"]].to_numpy() / GRAV_MS2
        acc = acc.T  # shape: (3, n_samples)

        n_samples = acc.shape[1]

        window_samples = int(self.window_min * 60 * sampling_rate_hz)
        step_samples = int(self.step_min * 60 * sampling_rate_hz)

        non_wear_samples = np.zeros(n_samples, dtype=bool)

        # Sliding window detection
        for start in range(0, n_samples - window_samples + 1, step_samples):
            end = start + window_samples
            window = acc[:, start:end]

            stds = window.std(axis=1)
            ranges = np.ptp(window, axis=1)

            std_axes = (stds < self.std_thresh_g).sum()
            range_axes = (ranges < self.range_thresh_g).sum()

            if (std_axes >= self.num_axes_required) or (
                range_axes >= self.num_axes_required
            ):
                non_wear_samples[start:end] = True

        # Convert to per-minute flags
        samples_per_min = int(60 * self.sampling_rate_hz)
        n_minutes = n_samples // samples_per_min

        non_wear_minutes = np.array(
            [
                non_wear_samples[i * samples_per_min : (i + 1) * samples_per_min].all()
                for i in range(n_minutes)
            ],
            dtype=int,
        )

        weartime_flags = 1 - non_wear_minutes  # 1 = wear, 0 = non-wear

        # Border / plausibility rules (single-pass)
        # Here we use a snapshot of the weartime_flags to avoid cascading changes (which might be more robust but is not in the original algorithm)

        df = pd.DataFrame({
            "wear": weartime_flags,
            "block": (pd.Series(weartime_flags).diff() != 0).cumsum()
        })

        blocks = (
            df.groupby("block")
            .agg(
                wear=("wear", "first"),
                duration=("wear", "size"),
                period_start=("wear", lambda x: x.index[0]),
                period_end=("wear", lambda x: x.index[-1] + 1),
            )
            .reset_index(drop=True)
        )

        # Apply rules ONCE, using the snapshot
        for i in range(1, len(blocks) - 1):
            if blocks.loc[i, "wear"] == 1:  # wear block only
                dur = blocks.loc[i, "duration"]
                adj = blocks.loc[i - 1, "duration"] + blocks.loc[i + 1, "duration"]
                ratio = dur / adj if adj > 0 else 1.0

                if (dur <= 180 and ratio < 0.8) or (dur <= 360 and ratio < 0.3):
                    s = blocks.loc[i, "period_start"]
                    e = blocks.loc[i, "period_end"]
                    weartime_flags[s:e] = 0

        # Handle trailing partial minute BEFORE generating weartime list
        remaining_samples = self.data_length % samples_per_min
        if remaining_samples > 0:
            # Extend the last full minute classification to cover remaining samples
            last_flag = weartime_flags[-1]
            weartime_flags = np.append(weartime_flags, last_flag)

        # Output
        self.weartime_list_ = generate_weartime_list_from_minutes(
            weartime_flags, sampling_rate=int(sampling_rate_hz)
        )
        # Clip end to actual data length
        self.weartime_list_["end"] = self.weartime_list_["end"].clip(upper=self.data_length)
        self.total_weartime_samples_ = (self.weartime_list_['end'] - self.weartime_list_['start']).sum()
        self.total_weartime_minutes_ = self.total_weartime_samples_ / (60 * self.sampling_rate_hz)
        self.total_weartime_hours_ = self.total_weartime_samples_ / (3600 * self.sampling_rate_hz)
        return self
