import pandas as pd
import numpy as np
from typing import Any, Unpack, Literal
from numba import njit
from typing_extensions import Self
from imu_weartime.weartime.base_weartime_detector import BaseWeartimeDetector
from imu_weartime.weartime.utils.ActivityCounts import ActivityCounts
from imu_weartime.weartime.utils.weartime_calc import (
    per_minute_counts,
    generate_weartime_list_from_minutes,
)
from mobgap.consts import GRAV_MS2


class WtdChoi(BaseWeartimeDetector):
    """
    Implementation of Choi et al. (2011) Wear Time Detection (WTD) algorithm for the lowback [1] and wrist worn [2] devices.

    Detects wear and non-wear periods in accelerometer data based on
    per-minute activity counts using a 90-min window (Window 1), with
    allowance for short interruptions of up to 2 non-zero minutes if
    surrounded by 30 consecutive zero minutes upstream and downstream
    (Window 2). The difference between the lowback and wrist versions is the axis used for computing the activity counts.

    [1] Choi L, Liu Z, Matthews CE, Buchowski MS. Validation of accelerometer wear and nonwear time classification algorithm.
    Med Sci Sports Exerc. 2011 Feb;43(2):357-64. doi: 10.1249/MSS.0b013e3181ed61a3. PMID: 20581716; PMCID: PMC3184184.

    [2] Choi L, Ward SC, Schnelle JF, Buchowski MS. Assessment of wear/nonwear time classification algorithms for triaxial accelerometer.
    Med Sci Sports Exerc. 2012 Oct;44(10):2009-16. doi: 10.1249/MSS.0b013e318258cb36. PMID: 22525772; PMCID: PMC3443532.

    Notes:
    -Version to test: 1) vertical (lowback), 2) norm (wrist)
    - Max 59 seconds removed due to per-minute aggregation from the classification logic. The final partial minute takes
    the value of the last full minute classification. This follows the original Choi et al. algorithm, where only
    full 90-min windows can be classified as non-wear; any incomplete window at the end defaults to wear by
    definition, consistent with the original methodology.
    - Fine tune with longer periods of 60, 90, 120, 180
        We fine tune different periods to accommodate another publication with similar methodology
        (Aadland, E., et al., A comparison of 10 accelerometer non-wear time criteria and logbooks in children. BMC Public Health, 2018. 18(1): p. 323.;
        Hutto B, Howard VJ, Blair SN, Colabianchi N, Vena JE, Rhodes D, Hooker SP. Identifying accelerometer nonwear and wear time in older adults. Int J Behav Nutr Phys Act. 2013 Oct 25;10:120. doi: 10.1186/1479-5868-10-120. PMID: 24156309; PMCID: PMC4015851.)
    """

    def __init__(
        self,
        *,
        window: int = 90,
        tol: int = 2,
        window2: int = 30,
        zero_thresh: float = 1e-6,
        position: Literal["wrist", "lowback"] = "lowback",
        version: Literal["norm", "vertical"] = "vertical",
    ) -> None:
        self.data = None
        self.window = window
        self.tol = tol
        self.window2 = window2
        self.zero_thresh = zero_thresh
        self.position = position
        self.version = version

    def detect(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float = 100,
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        """
        Detect wear time periods using Choi et al.'s method.
        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.data_length = len(data)

        # Require at least 90 minutes of data
        required_samples = 90 * 60 * sampling_rate_hz
        if len(data) < required_samples:
            raise ValueError(
                f"Input data must have at least 60 minutes of samples "
                f"({required_samples} samples), but got {len(data)}"
            )

        if self.version == "vertical" and self.position == "lowback":
            # Use vertical axis
            acc = self.data["acc_is"].to_numpy()
        elif self.version == "norm" and self.position in ["lowback", "wrist"]:
            cols = ["acc_is", "acc_ml", "acc_pa"]
            data_acc = self.data[cols]
            # Signal vector magnitude
            acc = np.linalg.norm(data_acc, axis=1)
        else:
            raise ValueError(
                "Unsupported configuration: "
                "use version='norm' with position in {'wrist', 'lowback'}, "
                "or version='vertical' with position='lowback' only."
            )

        # Convert to g-units
        acc = acc / GRAV_MS2

        # Compute activity counts per second
        activity_counts = (
            ActivityCounts()
            .calculate(data=acc.copy(), sampling_rate=self.sampling_rate_hz)
            .activity_counts_
        )

        # Convert to per-minute counts
        activity_counts_pm = per_minute_counts(activity_counts)
        n_minutes = len(activity_counts_pm)

        # Initialize wear-time flags: 1 = wear, 0 = non-wear
        weartime_flags = np.ones(n_minutes, dtype=int)

        # Create binary array
        zero_flags = (activity_counts_pm <= self.zero_thresh).astype(int)

        # Precomputing cumulative sum once
        cumsum = np.cumsum(zero_flags)

        @njit()
        def mark_weartime_numba(zero_flags, window=90, tol=2, window2=30):
            """
            Choi et al. logic wear-time detection (Numba-optimized)

            Parameters
            ----------
            zero_flags : np.ndarray
                1 if per-minute activity count <= threshold, 0 otherwise
            window : int
                Length of primary non-wear window (default 90 min)
            tol : int
                Allowance of non-zero minutes within the window (default 2)
            window2 : int
                Upstream/downstream zero minutes for allowance check (default 30)

            Returns
            -------
            weartime_flags : np.ndarray
                1 = wear, 0 = non-wear
            """
            n_minutes = len(zero_flags)
            weartime_flags = np.ones(n_minutes, dtype=np.int32)

            # Precompute cumulative sum for sliding-window zeros
            cumsum = np.zeros(n_minutes, dtype=np.int32)
            cumsum[0] = zero_flags[0]
            for i in range(1, n_minutes):
                cumsum[i] = cumsum[i - 1] + zero_flags[i]

            # Helper: count zeros in [start, end) interval
            def count_zeros(start, end):
                if start >= end:
                    return 0
                return cumsum[end - 1] - (cumsum[start - 1] if start > 0 else 0)

            i = 0
            while i <= n_minutes - window:
                # Total zeros in current window
                zeros_in_window = count_zeros(i, i + window)

                if zeros_in_window >= window - tol:
                    # Candidate non-wear period
                    candidate_start = i
                    candidate_end = i + window

                    # Extend candidate for allowance
                    j = candidate_start
                    while j < candidate_end:
                        if zero_flags[j] == 0:
                            allowance_start = j
                            allowance_end = min(j + tol, candidate_end)

                            # Check upstream zeros
                            upstream_start = max(
                                candidate_start, allowance_start - window2
                            )
                            upstream_zeros = count_zeros(
                                upstream_start, allowance_start
                            )
                            upstream_ok = upstream_zeros == (
                                allowance_start - upstream_start
                            )

                            # Check downstream zeros
                            downstream_end = min(candidate_end, allowance_end + window2)
                            downstream_zeros = count_zeros(
                                allowance_end, downstream_end
                            )
                            downstream_ok = downstream_zeros == (
                                downstream_end - allowance_end
                            )

                            if upstream_ok and downstream_ok:
                                # Skip allowance
                                j = allowance_end
                            else:
                                # Cannot allow this non-zero; shrink candidate
                                candidate_end = j
                                break
                        else:
                            j += 1

                    # Mark candidate window as non-wear
                    for k in range(candidate_start, candidate_end):
                        weartime_flags[k] = 0

                    # Ensure monotonic progress
                    i = max(candidate_end, i + 1)
                else:
                    i += 1

            return weartime_flags

        # Call the njit function
        weartime_flags = mark_weartime_numba(
            zero_flags, window=self.window, tol=self.tol, window2=self.window2
        )

        # Generate list of wear/non-wear intervals
        self.weartime_list_ = generate_weartime_list_from_minutes(weartime_flags)
        # Clip end to actual data length
        self.weartime_list_["end"] = self.weartime_list_["end"].clip(
            upper=self.data_length
        )
        self.total_weartime_samples_ = (
            self.weartime_list_["end"] - self.weartime_list_["start"]
        ).sum()
        self.total_weartime_minutes_ = self.total_weartime_samples_ / (
            60 * self.sampling_rate_hz
        )
        self.total_weartime_hours_ = self.total_weartime_samples_ / (
            3600 * self.sampling_rate_hz
        )
        return self
