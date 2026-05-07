from mobgap.data import LabExampleDataset
from mobgap.utils.conversions import to_body_frame
import pandas as pd
import numpy as np
from weartime import WtdPagnamenta


example_data = LabExampleDataset(
    reference_system="INDIP", reference_para_level="wb"
)

single_test = example_data.get_subset(
    cohort="HA", participant_id="001", test="Test11", trial="Trial1"
)

imu_data = to_body_frame(single_test.data_ss)

# adding temp data
np.random.seed(42)
# Base random temperature between 18-28°C
imu_data['temperature'] = np.random.uniform(25, 28, size=len(imu_data))

# Calling algorithm
WTD = WtdPagnamenta().detect(imu_data)

print(WTD.total_weartime_hours_)
print(WTD.total_weartime_minutes_)
print(WTD.total_weartime_samples_)
print(WTD.weartime_list_)