from mobgap.data import LabExampleDataset
from mobgap.utils.conversions import to_body_frame
import pandas as pd
from weartime import WtdHecht


example_data = LabExampleDataset(
    reference_system="INDIP", reference_para_level="wb"
)

single_test = example_data.get_subset(
    cohort="HA", participant_id="001", test="Test11", trial="Trial1"
)

imu_data = to_body_frame(single_test.data_ss)

# Calling algorithm
WTD = WtdHecht().detect(imu_data)

print(WTD.total_weartime_hours_)
print(WTD.total_weartime_minutes_)
print(WTD.total_weartime_samples_)
print(WTD.weartime_list_)