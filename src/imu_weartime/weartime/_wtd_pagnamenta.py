from typing_extensions import Self
from typing import Any, Unpack, Literal
import pandas as pd
import numpy as np
from imu_weartime.weartime.base_weartime_detector import BaseWeartimeDetector
from imu_weartime.weartime.utils.weartime_calc import generate_weartime_list_from_minutes
from mobgap.consts import GRAV_MS2
from scipy.signal import savgol_filter
from mobgap.data_transform import (
    Resample,
    chain_transformers,
)

class WtdPagnamenta(BaseWeartimeDetector):
    """
    Wear-time detection algorithm based on temperature changes, and a
    combined temperature–acceleration approach by Pagnamenta et al. (2022) [1].

    This implementation follows the best performing version of the temperature-based and combined models
    for identifying wear and non-wear periods in IMU data.

    Two operating modes are supported via `version`:
    - ``"temp"``: temperature-only detection
    - ``"temp_acc"``: combined temperature and acceleration detection

    --------------------
    Temperature-based algorithm
    --------------------
    1. Temperature signal preprocessing:
       - Resample the raw temperature signal to 1/6 of the original sampling rate
         (default: from 100 Hz to ~16.7 Hz) using interpolation.
       - Apply a Savitzky–Golay filter.

    2. Feature extraction:
       - Compute the first derivative of temperature (°C/s).
       - Identify candidate transition points where the absolute derivative
         exceeds ``temp_thresh_derivative``.

    3. Peak consolidation:
       - Consecutive derivative samples with the same sign are grouped.
       - Within each group, only the sample with the maximum absolute derivative
         is retained.

    4. Window-based validation:
       - For each candidate peak, compute the mean temperature in the
         preceding and following 5-minute windows.
       - A peak is retained if the absolute temperature difference between
         the two windows exceeds ``temp_thresh_windows``.

    5. State inference:
       - The sign of the temperature change determines the transition type:
         * Positive sign → end of non-wear
         * Negative sign → start of non-wear
       - Continuous wear / non-wear periods are reconstructed accordingly.

    --------------------
    Combined temperature–acceleration algorithm
    --------------------
    In ``"temp_acc"`` mode, temperature-based non-wear periods are further
    constrained using acceleration data:

    1. Acceleration processing:
       - Compute the vector magnitude (VM) of the tri-axial acceleration signal.
       - Convert VM to units of g.
       - Aggregate VM into non-overlapping 1-minute windows.
       - Compute the standard deviation per minute.

    2. Acceleration rule:
       - A minute is flagged as a non-wear candidate if VM SD is below
         ``low_activity_thresh_mg`` (default: 13 mg).

    3. Combination rule:
       - Temperature-based non-wear is reduced to minutes using a majority rule.
       - A minute is classified as non-wear only if *both*:
         * temperature-based non-wear is present, and
         * acceleration-based low activity is present.

    Notes:
    - This implementation uses only the novel algorithms proposed in the referenced paper.
    - Among the temperature-based models introduced, we selected the one demonstrating the best windowing performance according to the paper’s reported results.
    - The savgol filtering characteristics are not specified in the paper, they are chosen based on pilot test with AX6 temperature data.
    - Target sampling rate is 1/6 of 100Hz as the paper selects only every 6th value. Due to noise in the temperature data instead of selecting every 6 values we are resampling to that frequency.
    - Partial minutes at the end are truncated and not evaluated

    [1] Pagnamenta, S.; Grønvik, K.B.; Aminian, K.; Vereijken, B.; Paraschiv-Ionescu, A. Putting
    Temperature into the Equation: Development and Validation of Algorithms to Distinguish Non-Wearing from Inactivity and
    Sleep in Wearable Sensors. Sensors 2022, 22, 1117. https://doi.org/10.3390/s22031117
    """

    def __init__(
        self,
        *,
        version: Literal["temp", "temp_acc"] = "temp",
        temp_thresh_derivative: float = 0.02,
        temp_thresh_windows: float = 3.0,
        low_activity_thresh_mg: float = 13,
        _target_sampling_rate_hz: float = 100/6,
        position: Literal['wrist', 'lowback'] = 'lowback'
    ) -> None:
        self.version = version
        self.low_activity_thresh_mg = low_activity_thresh_mg
        self.low_activity_thresh = low_activity_thresh_mg / 1000
        self.temp_thresh_derivative = temp_thresh_derivative
        self.temp_thresh_windows = temp_thresh_windows
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

        # Temperature preprocessing: resampling to the 1/6th of the original freq (100Hz) + Savitzky–Golay filtering
        temp = data["temperature"].to_numpy()
        temp_ds = chain_transformers(
            temp,
            [("resample", Resample(self._target_sampling_rate_hz))],
            sampling_rate_hz=self.sampling_rate_hz,
        )

        # Using scipy directly for now
        window_length = int(15 * self._target_sampling_rate_hz) // 2 * 2 + 1  # ensure odd
        polyorder = 2
        temp_filt = savgol_filter(temp_ds, window_length=window_length, polyorder=polyorder, mode="mirror")

        temp = pd.Series(temp_filt).reset_index(drop=True)

        # Temperature derivative (°C per second)
        temp_derivative = np.gradient(temp.to_numpy(), 1 / self._target_sampling_rate_hz)

        # Calculate peaks of derivative as abs > 0.02
        peak_idx = np.where(np.abs(temp_derivative) > self.temp_thresh_derivative)[0]

        # We group consecutive peaks with same sign and keep only the max (or min) of each group (pruning)
        # This assesses the peaks which are consecutive in time (index)
        if len(peak_idx) > 0:
            # Get the signs of the derivative at candidate points
            signs = np.sign(temp_derivative[peak_idx])

            # Initialise list for filtered peaks
            filtered_peaks = []

            # Start first group
            group_start = 0
            for i in range(1, len(peak_idx)):
                # If sign changes or gap > 1 → new group
                if signs[i] != signs[i - 1] or peak_idx[i] != peak_idx[i - 1] + 1:
                    group = peak_idx[group_start:i]
                    # pick peak with max absolute derivative in the group
                    max_idx = group[np.argmax(np.abs(temp_derivative[group]))]
                    filtered_peaks.append(max_idx)
                    group_start = i

            # handle last group
            group = peak_idx[group_start:]
            max_idx = group[np.argmax(np.abs(temp_derivative[group]))]
            filtered_peaks.append(max_idx)

            # Convert to NumPy array
            peak_idx = np.array(filtered_peaks)

        # 5 min windows
        window_5min = int(5 * 60 * self._target_sampling_rate_hz)

        valid_peaks = []

        for i in peak_idx:
            prev_start = i - window_5min
            next_end = i + window_5min

            # Skip peaks too close to boundaries
            if prev_start < 0 or next_end >= len(temp):
                continue

            temp_prev = temp.iloc[prev_start:i].mean()
            temp_next = temp.iloc[i:next_end].mean()

            temp_diff = temp_next - temp_prev

            if np.abs(temp_diff) >= self.temp_thresh_windows:
                valid_peaks.append(
                    {
                        "index": i,
                        "temp_diff": temp_diff,
                        "sign": np.sign(temp_diff),
                    }
                )

        # Initialize periods
        periods = []

        if not valid_peaks:
            # No peaks, assume all wear
            periods.append({"start": 0, "end": len(temp), "state": "wear"})
        else:
            # Handle the first period
            first_peak = valid_peaks[0]
            if first_peak["sign"] > 0:  # end of non-wear
                periods.append({"start": 0, "end": first_peak["index"], "state": "non-wear"})
            elif first_peak["sign"] < 0:  # start of non-wear
                periods.append({"start": 0, "end": first_peak["index"], "state": "wear"})

            # Handle intermediate periods
            for i in range(len(valid_peaks) - 1):
                current_peak = valid_peaks[i]
                next_peak = valid_peaks[i + 1]

                # Current state
                state = "non-wear" if current_peak["sign"] < 0 else "wear"
                periods.append({"start": current_peak["index"], "end": next_peak["index"], "state": state})

            # Handle the last period
            last_peak = valid_peaks[-1]
            state = "non-wear" if last_peak["sign"] < 0 else "wear"
            periods.append({"start": last_peak["index"], "end": len(temp), "state": state})

        # Convert to series for easier mapping
        wear_series = pd.Series("wear", index=range(len(temp)))  # default wear
        for p in periods:
            wear_series[p["start"]:p["end"]] = p["state"]

        # Back to original sampling rate
        factor = int(round(self.sampling_rate_hz / self._target_sampling_rate_hz))
        wear_series_orig = wear_series.repeat(factor).iloc[: self.data_length].reset_index(drop=True)

        s = wear_series_orig

        # Boolean mask for wear
        is_wear = s.eq("wear")

        # Identify transitions (new wear segment starts when wear == True
        # and previous value was False)
        segment_id = (is_wear & ~is_wear.shift(fill_value=False)).cumsum()

        # Keep only wear rows and group by segment
        wear_segments = (
            s[is_wear]
            .groupby(segment_id[is_wear])
            .apply(lambda x: pd.Series({
                "start": x.index[0],
                "end": x.index[-1] + 1
            }))
        )

        # Add wt_id index
        wear_segments.index.name = "wt_id"

        wear_segments = wear_segments.unstack()
        wear_segments.index.name = "wt_id"

        if self.version == "temp":
            self.weartime_list_ = wear_segments
            # Clip end to actual data length
            self.weartime_list_["end"] = self.weartime_list_["end"].clip(upper=self.data_length)
            self.total_weartime_samples_ = (self.weartime_list_['end'] - self.weartime_list_['start']).sum()
            self.total_weartime_minutes_ = self.total_weartime_samples_ / (60 * sampling_rate_hz)
            self.total_weartime_hours_ = self.total_weartime_samples_ / (3600 * sampling_rate_hz)
            return self

        if self.version == "temp_acc":
            # Calculating norm
            cols = ['acc_is', 'acc_ml', 'acc_pa']
            data_acc = self.data[cols]
            # Signal vector magnitude
            acc = np.linalg.norm(data_acc, axis=1)

            # Convert to g-units
            acc = acc / GRAV_MS2

            samples_per_min = int(60 * self.sampling_rate_hz)
            vm_minute_sd = pd.Series(acc).groupby(np.arange(len(acc)) // samples_per_min).std()

            # Comparing with thresh
            acc_non_wear_candidate = vm_minute_sd < self.low_activity_thresh # 1 when quiet (indicating non wear)

            # Temperature rules to minutes for comparison
            samples_per_min = int(60 * self.sampling_rate_hz)

            temp_non_wear_min = (
                    s.eq("non-wear")
                    .groupby(np.arange(len(s)) // samples_per_min)
                    .mean() > 0.5
            ) # 1 when non-wear

            combined_non_wear_min = acc_non_wear_candidate & temp_non_wear_min
            combined_wear_min = (~combined_non_wear_min).astype(int).to_numpy()

            # Output
            self.weartime_list_ = generate_weartime_list_from_minutes(
                combined_wear_min, sampling_rate=int(self.sampling_rate_hz)
            )
            # Clip end to actual data length
            self.weartime_list_["end"] = self.weartime_list_["end"].clip(upper=self.data_length)
            self.total_weartime_samples_ = (self.weartime_list_['end'] - self.weartime_list_['start']).sum()
            self.total_weartime_minutes_ = self.total_weartime_samples_ / (60 * self.sampling_rate_hz)
            self.total_weartime_hours_ = self.total_weartime_samples_ / (3600 * self.sampling_rate_hz)
            return self
