# Copyright 2026 Dr Dimitrios Megaritis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
from importlib.resources import files
import pickle
from typing import Any, Unpack, Literal
from typing_extensions import Self
from imu_weartime.weartime.utils.ml_feature_extraction import rolling_window_indices
from imu_weartime.weartime.base_weartime_detector import (
    BaseWeartimeDetector,
    base_weartime_docfiller,
    _unify_weartime_df,
)
from mobgap._utils_internal.misc import timed_action_method
from imu_weartime.weartime.utils.feature_extraction import (
    extract_full_features,
    extract_features_95pct,
)
from imu_weartime.weartime.utils.windows_to_weartime import (
    overlapping_windows_to_sample_labels,
)


@base_weartime_docfiller
class WtdMegaritisLogReg(BaseWeartimeDetector):
    """
    Logistic Regression-based wear-time detection for lower-back worn IMU sensors.

    Uses pre-trained Logistic Regression models with time-domain and frequency-domain
    features extracted from overlapping 5-second windows. Includes biomechanically-informed
    post-processing to filter short bouts and apply confidence thresholds.

    Two model variants are available:
    - Full: 230 features, highest accuracy
    - Lightweight: 99 features (95%% SHAP importance), faster inference

    Post-processing steps:
    1. Majority voting across overlapping windows to obtain sample-level predictions
    2. Removal of wear bouts shorter than 15 seconds (biomechanically implausible)
    3. Confidence filtering for wear bouts under 20 minutes (requires >90%% vote agreement)
    4. Merging of short non-wear gaps (<15s) between wear periods

    Parameters
    ----------
    window_sec : float
        Window size in seconds for feature extraction (default: 5.0)
    overlap : float
        Window overlap fraction, 0.0 to 1.0 (default: 0.75)
    version : Literal["full", "lightweight"]
        Model variant: "full" (230 features) or "lightweight" (99 features, default)
    position : Literal['lowback']
        Sensor position (default: 'lowback', only supported position)

    Other Parameters
    ----------------
    %(other_parameters)s
    model : sklearn.linear_model.LogisticRegression
        Pre-trained Logistic Regression model loaded during initialization
    scaler : sklearn.preprocessing.StandardScaler
        Pre-trained feature scaler for normalization
    feature_names : list[str]
        Ordered list of feature names matching model training order

    Attributes
    ----------
    %(weartime_list_)s
    %(total_weartime_samples_)s
    %(total_weartime_minutes_)s
    %(total_weartime_hours_)s
    %(perf_)s

    Notes
    -----
    Pre-trained models and scalers are loaded from the package's production_models folder.
    Logistic Regression requires StandardScaler for feature normalization.

    Feature extraction dominates computation time. The lightweight version
    reduces feature count from 230 to 99, providing ~20-30%% faster inference
    for large datasets.
    """

    # Type hints
    data_length: int
    feature_names: list[str]
    model: Any
    scaler: Any

    def __init__(
        self,
        *,
        window_sec: float = 5.0,
        overlap: float = 0.75,
        version: Literal["full", "lightweight"] = "lightweight",
        position: Literal["lowback"] = "lowback",
    ) -> None:
        self.window_sec = window_sec
        self.overlap = overlap
        self.version = version
        self.position = position

        # Load pre-trained Logistic Regression model and scaler
        if self.version == "full":
            model_file = files("imu_weartime.weartime.production_models").joinpath(
                "logreg_fullfeatures_lowback_model.pkl"
            )
            scaler_file = files("imu_weartime.weartime.production_models").joinpath(
                "logreg_fullfeatures_lowback_scaler.pkl"
            )
            feature_order_file = files(
                "imu_weartime.weartime.production_models"
            ).joinpath("logreg_fullfeatures_lowback_feature_order.pkl")
        else:  # lightweight (95% features)
            model_file = files("imu_weartime.weartime.production_models").joinpath(
                "logreg_95pct_lowback_model.pkl"
            )
            scaler_file = files("imu_weartime.weartime.production_models").joinpath(
                "logreg_95pct_lowback_scaler.pkl"
            )
            feature_order_file = files(
                "imu_weartime.weartime.production_models"
            ).joinpath("logreg_95pct_lowback_feature_order.pkl")

        with model_file.open("rb") as f:
            self.model = pickle.load(f)

        with scaler_file.open("rb") as f:
            self.scaler = pickle.load(f)

        with feature_order_file.open("rb") as f:
            self.feature_names = pickle.load(f)

    @timed_action_method
    @base_weartime_docfiller
    def detect(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float = 100,
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        """
        %(detect_short)s using Logistic Regression classifier with overlapping windows.

        Processes IMU data in overlapping windows, extracts features, applies feature
        scaling and the pre-trained Logistic Regression model, then converts window-level
        predictions to sample-level wear-time segments using majority voting and
        biomechanical post-processing rules.

        Parameters
        ----------
        %(detect_para)s

        %(detect_return)s

        Notes
        -----
        Features are standardized using the scaler fitted during training before
        prediction.

        Post-processing pipeline:

        1. Majority voting: Each sample receives votes from overlapping windows
        2. Short bout removal: Wear bouts <15s are removed (too short to don/doff)
        3. Confidence filter: Wear bouts <20min require ≥90%% vote agreement
           (boundary bouts at start/end of data are exempt)
        4. Gap merging: Non-wear gaps <15s between wear periods are merged

        Notes:
            Features are standardized using the scaler fitted during training before
            prediction. Feature extraction dominates computation time - consider using
            version="lightweight" for ~20-30%% faster inference on large datasets.
        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.data_length = len(data)

        win_samples = int(self.window_sec * self.sampling_rate_hz)
        step = int(win_samples * (1 - self.overlap))

        all_predictions = []

        # Extract features and predict for each window
        for start, end in rolling_window_indices(self.data_length, win_samples, step):
            win = self.data.iloc[start:end]

            # Extract features based on model version
            if self.version == "full":
                features_dict = extract_full_features(win)
            else:  # lightweight (95% features)
                features_dict = extract_features_95pct(win)

            # Convert to DataFrame and scale
            features_df = pd.DataFrame([features_dict])

            # Handle NaN values (occur with zero/flat signal - kurtosis, skewness undefined)
            features_df = features_df.fillna(0)

            # Reorder columns to match training order
            features_df = features_df[self.feature_names]

            X_scaled = self.scaler.transform(features_df)

            # Predict weartime for this window
            y_pred = self.model.predict(X_scaled)[0]

            all_predictions.append(y_pred)

        # Convert window predictions to sample-level weartime with post-processing
        (
            self.weartime_list_,
            self.total_weartime_samples_,
            total_weartime_seconds,
            self.total_weartime_minutes_,
            self.total_weartime_hours_,
            coverage,
        ) = overlapping_windows_to_sample_labels(
            predictions=all_predictions,
            data_len=self.data_length,
            window_size=win_samples,
            stride=step,
            sampling_rate_hz=int(sampling_rate_hz),
            min_confidence_short_bouts=0.90,
            short_bout_threshold_minutes=20,
            min_bout_duration_seconds=15,
        )

        # Ensure end indices don't exceed data length
        self.weartime_list_["end"] = self.weartime_list_["end"].clip(
            upper=self.data_length
        )

        # Unify format (adds wt_id index, ensures correct dtypes)
        self.weartime_list_ = _unify_weartime_df(self.weartime_list_)

        return self
