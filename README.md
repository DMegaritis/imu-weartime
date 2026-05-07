# IMU Wear-Time Detection

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://img.shields.io/pypi/v/imu-weartime.svg)](https://pypi.org/project/imu-weartime/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20066962.svg)](https://doi.org/10.5281/zenodo.20066962)

A Python library providing validated wear-time detection algorithms for lower-back worn IMU sensors. Includes both established methods from the literature and novel signal processing, machine learning, and deep learning approaches.

This library was developed as part of the SUSTAIN Mobilise-D project and implements algorithms designed to work within the [MobGap](https://github.com/mobilise-d/mobgap) ecosystem for mobility analysis.

**Novel algorithms outperform all literature methods by 8- to 131-fold in wear-time quantification accuracy.**

Software developed by Dr Dimitrios Megaritis. Scientific authorship is listed in `CITATION.cff`.

---
## Installation

### From PyPI (recommended)

```bash
pip install imu-weartime
```

### From source

```bash
git clone https://github.com/[username]/imu-weartime.git
cd imu-weartime
pip install .
```

---

## Quick Start

```python
from mobgap.data import LabExampleDataset
from mobgap.utils.conversions import to_body_frame
from imu_weartime.weartime import WtdMegaritis_CNN

# Load example data
data = LabExampleDataset().get_subset(
    cohort="HA", participant_id="001", test="Test11", trial="Trial1"
)

# Apply wear-time detection
wtd = WtdMegaritis_CNN(version="cnn_lstm").detect(
    to_body_frame(data.data_ss), 
    sampling_rate_hz=data.sampling_rate_hz
)

print(f"Total wear-time: {wtd.total_weartime_hours_:.2f} hours")
print(wtd.weartime_list_)
```

---

## Available Algorithms

### Novel Algorithms (Megaritis et al.)

Developed and validated specifically for this library (ranked by performance):

1. **`WtdMegaritis_CNN`** - Deep learning with 1D CNN and CNN-LSTM variants (**best-performing**)
   - Versions: `"cnn"` (baseline) or `"cnn_lstm"` (with LSTM, default)
   - Operates on raw windowed IMU data (no feature engineering)

2. **`WtdMegaritis_XGBoost`** - Gradient boosting classifier
   - Versions: `"full"` (230 features) or `"lightweight"` (79 features, default)
   - SHAP-based feature selection for lightweight variant

3. **`WtdMegaritis_LogReg`** - Logistic regression with feature scaling
   - Versions: `"full"` (230 features) or `"lightweight"` (99 features, default)
   - StandardScaler normalization included

4. **`Wtd_Megaritis_signal`** - Signal processing using gyroscope spectral patterns
   - Gyroscope spectral centroids (ML and IS axes) with accelerometer variability (PA axis)
   - Multi-level voting with biomechanical post-processing
   - No machine learning required

### Literature Algorithms

Implementations of published methods (adapted for compatibility):

- **`WtdTroiano`**, **`WtdChoi`**, **`WtdVanHees`**, **`WtdAhmadi`**, **`WtdKing`**, **`WtdDuncan`**, **`WtdRasmussen`**, **`WtdZhou`**, **`WtdVert`**, **`WtdPagnamenta`**, **`WtdNishiyama`**, **`WtdHecht`**

See individual algorithm documentation for references and parameter details.

---

## Usage Examples

### Deep Learning (Recommended)
```python
from imu_weartime.weartime import WtdMegaritis_CNN

# CNN-LSTM (best performing)
wtd = WtdMegaritis_CNN(version="cnn_lstm").detect(
    imu_data, sampling_rate_hz=100.0
)

# CNN baseline
wtd = WtdMegaritis_CNN(version="cnn").detect(
    imu_data, sampling_rate_hz=100.0
)
```

### Machine Learning
```python
from imu_weartime.weartime import WtdMegaritis_XGBoost, WtdMegaritis_LogReg

# XGBoost (lightweight by default)
wtd_xgb = WtdMegaritis_XGBoost(version="lightweight").detect(
    imu_data, sampling_rate_hz=100.0
)

# Logistic Regression
wtd_lr = WtdMegaritis_LogReg(version="lightweight").detect(
    imu_data, sampling_rate_hz=100.0
)
```

### Signal Processing
```python
from imu_weartime.weartime import Wtd_Megaritis_signal

wtd = Wtd_Megaritis_signal().detect(imu_data, sampling_rate_hz=100.0)
```

### Literature Methods
```python
from imu_weartime.weartime import WtdTroiano, WtdVanHees, WtdAhmadi

wtd_troiano = WtdTroiano(nci=False).detect(imu_data, sampling_rate_hz=100.0)
wtd_vanhees = WtdVanHees().detect(imu_data, sampling_rate_hz=100.0)
wtd_ahmadi = WtdAhmadi(version="sd_xyz").detect(imu_data, sampling_rate_hz=100.0)
```

---

## Algorithm Performance

Novel algorithms ranked by validation performance:

1. **CNN-LSTM** (best overall)
2. **CNN**
3. **XGBoost** (lightweight)
4. **Signal Processing** = **Logistic Regression**

**Novel methods outperform all literature algorithms by 8- to 131-fold in wear-time quantification accuracy.** 

Detailed performance metrics and validation results will be published in the accompanying manuscript.

---

## Citation

If you use this library in your research, please cite:

```bibtex
@software{megaritis2026imu_weartime,
  author    = {Megaritis, Dimitrios},
  title     = {IMU Wear-Time Detection: Validated Algorithms for Wearable Sensors},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/[username]/imu-weartime},
  license   = {Apache-2.0}
}
```

**Validation reference** (manuscript in preparation):

> Megaritis, D., Del Din, S., Tasca, P., Cereatti, A., & Vogiatzis, I. (2026). 
> Novel wear-time detection algorithms for lower-back worn IMU sensors. *Manuscript in preparation.*

For algorithms from the literature, please cite the original papers listed in the respective algorithm documentation.

---

## Compatibility

- **Python**: 3.9 to 3.13
- **Dependencies**: `mobgap==1.1.0`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `tensorflow`
- **Sensor**: Lower-back worn IMU (100 Hz sampling recommended)
- **Data format**: MobGap body-frame coordinates

---

## Funding and Support

This work was supported by the SUSTAIN Mobilise-D Consortium. Content in this publication reflects the authors' view and neither EFPIA, or any Associated Partners are responsible for any use that may be made of the information contained herein.

---

## License

Licensed under the Apache License 2.0. Free to use for any purpose, including commercial use. See [LICENSE](LICENSE) for full terms.

**Disclaimer**: This software is provided "as is" without warranty. It is not a medical product nor licensed for medical use.