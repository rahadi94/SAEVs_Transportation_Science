from dataclasses import dataclass


@dataclass(frozen=True)
class RLSettings:
    action_base: int = 4  # K
    vehicles_per_decision: int = 10


rl_settings = RLSettings()


