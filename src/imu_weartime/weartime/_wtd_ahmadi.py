from typing_extensions import Self
from typing import Any, Unpack, Literal
import pandas as pd
import numpy as np
from imu_weartime.weartime.base_weartime_detector import BaseWeartimeDetector
from imu_weartime.weartime.utils.weartime_calc import generate_weartime_list_from_minutes
from mobgap.consts import GRAV_MS2
from mobgap.data_transform import chain_transformers, ButterworthFilter

class WtdAhmadi(BaseWeartimeDetector):
    """
    Implementation of the four Ahmadi non-wear detection algorithms.

    This class implements four non-wear detection variants originally proposed
    by Ahmadi et al., using raw tri-axial accelerometer data. The detector
    operates at a per-second resolution internally, aggregates results into
    fixed-length windows (default: 30 minutes), applies plausibility rules,
    and outputs wear-time intervals at minute-level resolution.

    Implemented algorithmic variants:
    1. "sd_xyz":
        Non-wear is detected when the standard deviation of each acceleration
        axis (IS, ML, PA) within a 1-second window is below a fixed threshold.

    2. "sd_vm":
        Non-wear is detected when the standard deviation of the vector magnitude
        of acceleration within a 1-second window is below a fixed threshold.

    3. "sum_hpf":
        Non-wear is detected when the summed absolute values of high-pass
        filtered acceleration signals across axes are approximately zero,
        indicating device immobility.

    4. "tilt":
        Non-wear is detected when changes in tilt angles (derived from
        tri-axial acceleration) remain below a fixed threshold, indicating
        static device orientation.

    Processing pipeline overview:
    1. Convert raw acceleration to g-units.
    2. Compute per-second non-wear flags according to the selected variant.
    3. Aggregate per-second flags into fixed-length windows (default: 30 min).
    4. Apply single-pass plausibility rules to remove implausible short wear
       segments surrounded by non-wear.
    5. Expand window-level decisions to per-minute wear-time flags.
    6. Generate wear-time intervals and summary statistics.

    Parameters
    ----------
    window_min : int
        Window size in minutes (default: 30)
    std_thresh_mg : float
        Standard deviation threshold in milligrams (default: 13.0)
    tilt : float
        Tilt angle threshold in degrees (default: 1.0)
    sum_hpf_thresh : float
        Sum of high-pass filtered signal threshold (default: 0.009)
    version : Literal["sd_xyz", "sd_vm", "sum_hpf", "tilt"]
        Algorithm variant (default: "sd_xyz")
    position : Literal['wrist', 'lowback']
        Sensor position (default: 'lowback')

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(weartime_list_)s
    %(total_weartime_samples_)s
    %(total_weartime_minutes_)s
    %(total_weartime_hours_)s
    %(perf_)s

    Notes:
    - A window is classified as non-wear only if all seconds within that
      window satisfy the non-wear condition.
    - This implementation prioritizes methodological consistency across
      algorithm variants to facilitate comparative validation analyses.
    -Final partial seconds are evaluated; none are removed
    -For the sum_hpf variant, the non-wear threshold was empirically fine-tuned using verified quiet periods where the
    device was resting on a table. The threshold was derived from the 99th percentile of the summed absolute high-pass–filtered
    acceleration across the three axes (x, y, z) observed during these stationary recordings,
    in order to account for sensor noise and numerical residuals.
    - Versions to test: "sd_xyz", "sd_vm", "sum_hpf", "tilt".
    """

    def __init__(
        self,
        *,
        window_min: int = 30,
        std_thresh_mg: float = 13.0,
        tilt: float = 1.0,
        sum_hpf_thresh: float = 0.009,
        version: Literal["sd_xyz", "sd_vm", "sum_hpf", "tilt"] = "sd_xyz",
        position: Literal['wrist', 'lowback'] = 'lowback'
    ) -> None:
        self.window_min = window_min
        self.std_thresh_mg = std_thresh_mg
        self.tilt = tilt
        self.sum_hpf_thresh = sum_hpf_thresh
        self.version = version
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
        self.std_thresh_g = self.std_thresh_mg / 1000

        # Select version and compute acc variable accordingly
        if self.version == "sd_xyz":
            # SD of each axis
            acc = data[["acc_is", "acc_ml", "acc_pa"]].to_numpy() / GRAV_MS2

        elif self.version == "sd_vm":
            # SD of vector magnitude
            acc_arr = data[["acc_is", "acc_ml", "acc_pa"]].to_numpy() / GRAV_MS2
            acc = np.linalg.norm(acc_arr, axis=1, keepdims=True)

        elif self.version == "sum_hpf":
            # High-pass filter each axis
            acc_filt = []
            cutoff = 0.25
            filter_chain = [("butter", ButterworthFilter(order=4, cutoff_freq_hz=cutoff, filter_type='highpass'))]
            for col in ["acc_is", "acc_ml", "acc_pa"]:
                acc_col = data[col].to_numpy() / GRAV_MS2
                acc_filt.append(chain_transformers(acc_col, filter_chain, sampling_rate_hz=self.sampling_rate_hz))
            acc = np.stack(acc_filt, axis=1)

        elif self.version == "tilt":
            # Tilt: compute angle relative to horizontal for each axis
            acc_arr = data[["acc_is", "acc_ml", "acc_pa"]].to_numpy() / GRAV_MS2
            # arctan(axis / sqrt(sum of squares of other two axes))
            tilt_x = np.arctan2(acc_arr[:, 0], np.sqrt(acc_arr[:, 1] ** 2 + acc_arr[:, 2] ** 2)) * 180 / np.pi
            tilt_y = np.arctan2(acc_arr[:, 1], np.sqrt(acc_arr[:, 0] ** 2 + acc_arr[:, 2] ** 2)) * 180 / np.pi
            tilt_z = np.arctan2(acc_arr[:, 2], np.sqrt(acc_arr[:, 0] ** 2 + acc_arr[:, 1] ** 2)) * 180 / np.pi
            acc = np.stack([tilt_x, tilt_y, tilt_z], axis=1)

        # For each version we compute the 1-second flags
        samples_per_sec = int(self.sampling_rate_hz)
        n_secs = len(acc) // samples_per_sec  # full seconds only
        acc_trim = acc[:n_secs * samples_per_sec]  # trim excess samples at end

        # reshape to (n_secs, samples_per_sec, n_axes)
        acc_sec = acc_trim.reshape(n_secs, samples_per_sec, acc.shape[1])

        if self.version == "sd_xyz":
            # SD per axis over each 1-second window
            std_sec = np.std(acc_sec, axis=1)  # shape: (n_secs, n_axes)
            sec_flags = (std_sec < self.std_thresh_g).all(axis=1)  # True = non-wear

        elif self.version == "sd_vm":
            # Compute vector magnitude per sample, then SD per 1-second window
            vm = np.linalg.norm(acc_sec, axis=2)  # shape: (n_secs, samples_per_sec)
            std_vm = np.std(vm, axis=1)  # shape: (n_secs,)
            sec_flags = (std_vm < self.std_thresh_g)  # True = non-wear

        elif self.version == "sum_hpf":
            # Sum absolute filtered axes per 1-second window
            sum_abs = np.sum(np.abs(acc_sec), axis=2)  # shape: (n_secs, samples_per_sec)
            sec_flags = (sum_abs.max(axis=1) < self.sum_hpf_thresh)  # True = non-wear

        elif self.version == "tilt":
            # Change in tilt per 1-second window
            delta_tilt = np.diff(acc_sec, axis=1, prepend=acc_sec[:, 0:1, :])
            sec_flags = (np.abs(delta_tilt).max(axis=1) < self.tilt).all(axis=1)  # True = non-wear

        # 30-min windows
        window_secs = self.window_min * 60  # seconds per 30-min block
        n_seconds = len(sec_flags)

        n_blocks_full = n_seconds // window_secs
        remainder_secs = n_seconds % window_secs  # leftover seconds at the end
        weartime_blocks = np.ones(n_blocks_full + (1 if remainder_secs > 0 else 0), dtype=int)

        for i in range(n_blocks_full):
            start = i * window_secs
            end = start + window_secs
            if sec_flags[start:end].all():
                weartime_blocks[i] = 0

        # Final partial block handling here
        if remainder_secs > 0:
            start = n_blocks_full * window_secs
            end = start + remainder_secs
            if sec_flags[start:end].all():  # same strict per-second rule
                weartime_blocks[-1] = 0  # mark last block as non-wear

        # Plausibility rules on 30-min blocks (single-pass)
        df = pd.DataFrame({
            "wear": weartime_blocks,
            "block": (pd.Series(weartime_blocks).diff() != 0).cumsum()
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
        for i in range(1, len(blocks) - 1):
            if blocks.loc[i, "wear"] == 1:  # wear block only
                dur = blocks.loc[i, "duration"]
                adj = blocks.loc[i - 1, "duration"] + blocks.loc[i + 1, "duration"]
                ratio = dur / adj if adj > 0 else 1.0
                if (dur <= 30 and ratio < 0.3):
                    s = blocks.loc[i, "period_start"]
                    e = blocks.loc[i, "period_end"]
                    weartime_blocks[s:e] = 0

        # Expand 30-min blocks to per-minute flags
        samples_per_min = int(self.sampling_rate_hz * 60)
        minutes_total = len(data) // samples_per_min
        weartime_flags = np.repeat(weartime_blocks, self.window_min)[:minutes_total]

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
