import pandas as pd
import numpy as np
from typing import Any, Literal
from typing_extensions import Self, Unpack
from imu_weartime.weartime.base_weartime_detector import BaseWeartimeDetector
from imu_weartime.weartime.utils.ActivityCounts import ActivityCounts
from imu_weartime.weartime.utils.weartime_calc import per_minute_counts, generate_weartime_list_from_minutes
from mobgap.consts import GRAV_MS2


class WtdTroiano(BaseWeartimeDetector):
    """
    Implementation of a Wear Time Detection (WTD) algorithm by Troiano et al. (2008) [1].

    The algorithm detects wear and non-wear periods in accelerometer data based on
    per-minute activity counts.

    - With `nci=False`, it applies a sliding 60-minute window (step = 1 min) and classifies any
      window as non-wear if the total number of non-zero minutes within that window
      does not exceed the tolerance (`tol`). The non-zero minutes can occur anywhere
      in the window; their positions do not matter.

    - With `nci=True`, it implements the original NCI/Troiano logic. A non-wear
      period begins at a zero-count minute and continues until consecutive non-zero
      minutes exceed the tolerance (`tol`) or a minute exceeds `tol_upper`. Only
      periods with a total duration (excluding tolerated consecutive non-zero minutes)
      equal to or greater than the minimum window length are marked as non-wear.

    Parameters
    ----------

    Notes:
    -Deviations: we use a reversed engineered version of actigraphy counts algorithm
    -In the possibility were 2 minutes have interuption we do not take into account any threshold as in Troiano since we do not use the same counts.
    -Since the activity counts algorithm is a reverse engineered version of the actigraphy counts algorithm, we do not use the same thresholds.
     The maximum tolerated activity (tol_upper) count during brief non-wear interruptions was empirically derived from confirmed non-wear data as the
     25th percentile of non-zero activity counts, representing typical incidental device movement rather than true physical activity.

     [1] Troiano RP, Berrigan D, Dodd KW, Mâsse LC, Tilert T, McDowell M. Physical activity in the United States measured by accelerometer.
     Med Sci Sports Exerc. 2008 Jan;40(1):181-8. doi: 10.1249/mss.0b013e31815a51b3. PMID: 18091006.

     Notes:
     -Versions to test: 1) nci true, 2) nci false
     - All data are included; any leftover seconds at the end of the data (last partial minute) are counted as a
    final partial minute are assessed with the same rules as full minutes
    """

    def __init__(self, *,
                 window: int = 60,
                 tol: int = 2,
                 tol_upper: int = 8,
                 nci: bool = True,
                 zero_thresh:float = 1e-6,
                 position: Literal['lowback'] = 'lowback'):
        self.data = None
        self.window = window
        self.tol =tol
        self.tol_upper = tol_upper
        self.nci = nci
        self.zero_thresh = zero_thresh
        self.position = position

    def detect(
            self,
            data: pd.DataFrame,
            *,
            sampling_rate_hz: float = 100,
            **_: Unpack[dict[str, Any]]
    ) -> Self:
        """
        Detect wear time periods in accelerometer data using Troiano et al.'s method
        with overlapping 60-min windows.

        Parameters
        ----------
        data : pd.DataFrame
            Accelerometer data with at least 'acc_is' column.
        sampling_rate_hz : float
            Sampling rate in Hz.
        nci : bool
            Whether to use the NCI/Troiano consecutive-non-zero logic.
        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.data_length = len(self.data)

        # Require at least 60 minutes of data
        required_samples = 60 * 60 * sampling_rate_hz
        if len(self.data) < required_samples:
            raise ValueError(
                f"Input data must have at least 60 minutes of samples "
                f"({required_samples} samples), but got {len(self.data)}"
            )

        # Use vertical axis
        acc = self.data['acc_is'].to_numpy()
        acc = acc / GRAV_MS2  # convert to g-units

        # Compute activity counts per second
        activity_counts = ActivityCounts().calculate(
            data=acc.copy(), sampling_rate=self.sampling_rate_hz
        ).activity_counts_

        # Convert to per-minute counts
        activity_counts_pm = per_minute_counts(activity_counts)
        n_minutes = len(activity_counts_pm)

        # Initialize wear-time flags: 1 = wear, 0 = non-wear
        weartime_flags = np.ones(n_minutes, dtype=int)

        if not self.nci:
            # Sliding window sum logic
            status = np.zeros(n_minutes, dtype=int)
            status[(activity_counts_pm > self.zero_thresh) & (activity_counts_pm <= self.tol_upper)] = 1 # the minute is non-zero but under the threshold
            status[activity_counts_pm > self.tol_upper] = self.tol + 1 # the minute is over the threshold

            for start in range(n_minutes - self.window + 1):
                end = start + self.window
                if np.sum(status[start:end]) <= self.tol:
                    weartime_flags[start:end] = 0  # mark as non-wear

        else:
            # NCI/Troiano consecutive-non-zero logic
            zeros = 0
            tolcount = 0
            flag = False

            for i in range(n_minutes):
                c = activity_counts_pm[i]

                if zeros == 0 and c != 0:
                    continue  # waiting for non-wear start

                if c == 0:
                    zeros += 1
                    tolcount = 0
                elif 0 < c <= self.tol_upper:
                    zeros += 1
                    tolcount += 1
                else:  # c > tol_upper
                    zeros += 1
                    tolcount += 1
                    flag = True

                # End of non-wear period
                if tolcount > self.tol or flag or i == n_minutes - 1:
                    if zeros - tolcount >= self.window:
                        start_idx = i - zeros + 1
                        end_idx = i - tolcount + 1
                        weartime_flags[start_idx:end_idx] = 0  # mark non-wear
                    zeros = 0
                    tolcount = 0
                    flag = False

        # Generate list of wear/non-wear intervals
        self.weartime_list_ = generate_weartime_list_from_minutes(weartime_flags)
        # Clip end to actual data length
        self.weartime_list_["end"] = self.weartime_list_["end"].clip(upper=self.data_length)
        self.total_weartime_samples_ = (self.weartime_list_['end'] - self.weartime_list_['start']).sum()
        self.total_weartime_minutes_ = self.total_weartime_samples_ / (60 * self.sampling_rate_hz)
        self.total_weartime_hours_ = self.total_weartime_samples_ / (3600 * self.sampling_rate_hz)
        return self
