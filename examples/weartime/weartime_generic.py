from mobgap.data import LabExampleDataset
from mobgap.utils.conversions import to_body_frame

from weartime.class_structure import ExampleWTD


example_data = LabExampleDataset(
    reference_system="INDIP", reference_para_level="wb"
)

single_test = example_data.get_subset(
    cohort="HA", participant_id="001", test="Test11", trial="Trial1"
)

imu_data = to_body_frame(single_test.data_ss)


# Calling algorithm
WTD = ExampleWTD().detect(imu_data)
