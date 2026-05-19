"""Algorithms to detect wear time from raw IMU data."""

from imu_weartime.weartime._wtd_troiano import WtdTroiano
from imu_weartime.weartime._wtd_choi import WtdChoi
from imu_weartime.weartime._wtd_vanhees import WtdVanHees
from imu_weartime.weartime._wtd_ahmadi import WtdAhmadi
from imu_weartime.weartime._wtd_zhou import WtdZhou
from imu_weartime.weartime._wtd_vert import WtdVert
from imu_weartime.weartime._wtd_king import WtdKing
from imu_weartime.weartime._wtd_nishiyama import WtdNishiyama
from imu_weartime.weartime._wtd_hecht import WtdHecht
from imu_weartime.weartime._wtd_rasmussen import WtdRasmussen
from imu_weartime.weartime._wtd_pagnamenta import WtdPagnamenta
from imu_weartime.weartime._wtd_duncan import WtdDuncan
from imu_weartime.weartime._wtd_megaritis_signal import WtdMegaritisSignal
from imu_weartime.weartime._wtd_megaritis_xgboost import WtdMegaritisXGBoost
from imu_weartime.weartime._wtd_megaritis_logreg import WtdMegaritisLogReg
from imu_weartime.weartime._wtd_megaritis_cnn import WtdMegaritisCNN

__all__ = [
    "WtdTroiano",
    "WtdChoi",
    "WtdVanHees",
    "WtdAhmadi",
    "WtdZhou",
    "WtdVert",
    "WtdKing",
    "WtdHecht",
    "WtdRasmussen",
    "WtdPagnamenta",
    "WtdDuncan",
    "WtdNishiyama",
    "WtdMegaritisSignal",
    "WtdMegaritisCNN",
    "WtdMegaritisXGBoost",
    "WtdMegaritisLogReg",
    "WtdMegaritisCNN",
]
