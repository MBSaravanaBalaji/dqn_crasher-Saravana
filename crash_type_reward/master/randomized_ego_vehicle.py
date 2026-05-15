"""
Randomized ego vehicle for domain-randomized master NPC training.

Each episode, IDM/MOBIL params are sampled uniformly from realistic ranges,
forcing the NPC to learn manipulation strategies that generalize across
cautious-to-aggressive ego driving styles.
"""

import numpy as np
from highway_env.vehicle.behavior import IDMVehicle


class RandomizedEgoVehicle(IDMVehicle):
    TIME_WANTED_RANGE              = (0.8,  1.5)   # s
    COMFORT_ACC_MIN_RANGE          = (-5.0, -2.5)  # m/s²
    LANE_CHANGE_MIN_ACC_GAIN_RANGE = (0.05, 0.15)  # m/s²
    POLITENESS_RANGE               = (0.0,  0.2)

    @classmethod
    def create_random(cls, road, speed=None, lane_from=None, lane_to=None,
                      lane_id=None, spacing=1):
        vehicle = super().create_random(
            road, speed=speed, lane_from=lane_from,
            lane_to=lane_to, lane_id=lane_id, spacing=spacing,
        )
        rng = road.np_random
        vehicle.TIME_WANTED              = float(rng.uniform(*cls.TIME_WANTED_RANGE))
        vehicle.COMFORT_ACC_MIN          = float(rng.uniform(*cls.COMFORT_ACC_MIN_RANGE))
        vehicle.LANE_CHANGE_MIN_ACC_GAIN = float(rng.uniform(*cls.LANE_CHANGE_MIN_ACC_GAIN_RANGE))
        vehicle.POLITENESS               = float(rng.uniform(*cls.POLITENESS_RANGE))
        return vehicle


class TransferRandomizedEgoVehicle(IDMVehicle):
    """MOBIL-enabled randomized ego for stage-2 transfer to normal behavior.

    Broader than constrained stage-1, but avoids extremely evasive lane-change
    settings that can erase rear-end learning signal.
    """

    TIME_WANTED_RANGE              = (0.9,  1.4)   # s
    COMFORT_ACC_MIN_RANGE          = (-4.0, -2.6)  # m/s²
    LANE_CHANGE_MIN_ACC_GAIN_RANGE = (0.12, 0.28)  # m/s²
    POLITENESS_RANGE               = (0.1,  0.3)

    @classmethod
    def create_random(cls, road, speed=None, lane_from=None, lane_to=None,
                      lane_id=None, spacing=1):
        vehicle = super().create_random(
            road, speed=speed, lane_from=lane_from,
            lane_to=lane_to, lane_id=lane_id, spacing=spacing,
        )
        rng = road.np_random
        vehicle.TIME_WANTED              = float(rng.uniform(*cls.TIME_WANTED_RANGE))
        vehicle.COMFORT_ACC_MIN          = float(rng.uniform(*cls.COMFORT_ACC_MIN_RANGE))
        vehicle.LANE_CHANGE_MIN_ACC_GAIN = float(rng.uniform(*cls.LANE_CHANGE_MIN_ACC_GAIN_RANGE))
        vehicle.POLITENESS               = float(rng.uniform(*cls.POLITENESS_RANGE))
        return vehicle


class CautiousEgoVehicle(IDMVehicle):
    """Fixed cautious ego for generalization eval — large gap, hard braking, reluctant lane changes."""
    TIME_WANTED              = 2.0
    COMFORT_ACC_MIN          = -5.0
    LANE_CHANGE_MIN_ACC_GAIN = 0.3
    POLITENESS               = 0.2


class AggressiveEgoVehicle(IDMVehicle):
    """Fixed aggressive ego for generalization eval — tight gap, soft braking, frequent lane changes."""
    TIME_WANTED              = 0.8
    COMFORT_ACC_MIN          = -2.5
    LANE_CHANGE_MIN_ACC_GAIN = 0.05
    POLITENESS               = 0.0
