"""
Tuned IDM/MOBIL ego vehicle for master NPC training.

IDM/MOBIL parameters are class-level constants — subclassing is the only way
to tune them per-vehicle without affecting other IDM vehicles on the road.

Tuning rationale (vs highway_env defaults):
  TIME_WANTED:                  1.0  (default 1.5s)  — follows closer, less reaction time
  COMFORT_ACC_MIN:             -3.0  (default -5.0)  — softer braking, harder to avoid rear-end trap
  LANE_CHANGE_MIN_ACC_GAIN:     0.1  (default 0.2)   — changes lanes more readily
  LANE_CHANGE_MAX_BRAKING_IMPOSED: 3.0 (default 2.0) — less conservative safety check for lane changes

These make the ego more susceptible to NPC manipulation while remaining
plausible as a real (slightly aggressive) driver.
"""

from highway_env.vehicle.behavior import IDMVehicle


class MasterEgoVehicle(IDMVehicle):
    TIME_WANTED                     = 1.0
    COMFORT_ACC_MIN                 = -3.0
    LANE_CHANGE_MIN_ACC_GAIN        = 0.1
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 3.0
    POLITENESS                      = 0.1


class ConstrainedMobilEgoVehicle(IDMVehicle):
    """MOBIL-enabled ego tuned for lower evasiveness and better transfer learning.

    Goal: keep lane-change capability, but require clearer utility/safety margin
    before changing lanes, so rear-end induction remains learnable.
    """

    TIME_WANTED                     = 1.1
    COMFORT_ACC_MIN                 = -3.2
    LANE_CHANGE_MIN_ACC_GAIN        = 0.22
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.2
    POLITENESS                      = 0.2


class NoMobilEgoVehicle(MasterEgoVehicle):
    """MasterEgoVehicle with MOBIL lane changes disabled.

    Ego stays in its lane and can only respond longitudinally (IDM braking).
    Use this for rear-end training so ego cannot escape by lane-changing
    when the NPC decelerates in front of it.
    """

    def change_lane_policy(self) -> None:
        pass
