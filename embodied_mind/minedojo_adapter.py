"""
MineDojo adapter.

The agent works with readable dict actions ({"forward": 1, "camera": [0, 15], ...})
and a flat observation dict (rgb as HxWx3, inventory as {name: count}).
MineDojo uses an 8-dim MultiDiscrete action and channel-first frames, so this
wrapper translates in both directions.

MineDojo action layout (MultiDiscrete([3, 3, 4, 25, 25, 8, 244, 36])):
    0 forward/back        0 noop, 1 forward, 2 back
    1 left/right          0 noop, 1 left, 2 right
    2 jump/sneak/sprint   0 noop, 1 jump, 2 sneak, 3 sprint
    3 camera pitch        0..24, 12 = no change, 15 degrees per bin
    4 camera yaw          0..24, 12 = no change, 15 degrees per bin
    5 functional          0 noop, 1 use, 2 drop, 3 attack, 4 craft, 5 equip, 6 place, 7 destroy
    6 craft argument      index into the craft/smelt item list
    7 inventory slot      argument for equip/place/destroy
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

CAMERA_BIN_DEG = 15
CAMERA_CENTER = 12


def _craft_items() -> list:
    """Craftable item names in MineDojo's craft-argument order, if MineDojo is installed."""
    try:
        from minedojo.sim.mc_meta import mc as MC  # type: ignore
        return list(MC.ALL_CRAFT_SMELT_ITEMS)
    except Exception:  # MineDojo not installed or layout changed
        return []


def deg_to_bin(deg: float) -> int:
    return int(np.clip(CAMERA_CENTER + round(deg / CAMERA_BIN_DEG), 0, 24))


def dict_to_multidiscrete(action: dict, craft_items: Optional[list] = None) -> np.ndarray:
    """Convert a readable action dict to MineDojo's 8-dim action array."""
    a = np.array([0, 0, 0, CAMERA_CENTER, CAMERA_CENTER, 0, 0, 0], dtype=np.int64)

    if action.get("forward"):
        a[0] = 1
    elif action.get("backward"):
        a[0] = 2

    if action.get("left"):
        a[1] = 1
    elif action.get("right"):
        a[1] = 2

    if action.get("jump"):
        a[2] = 1
    elif action.get("sneak"):
        a[2] = 2
    elif action.get("sprint"):
        a[2] = 3

    pitch, yaw = action.get("camera", [0, 0])
    a[3] = deg_to_bin(pitch)
    a[4] = deg_to_bin(yaw)

    craft = action.get("craft")
    if craft:
        items = craft_items if craft_items is not None else []
        if craft in items:
            a[5] = 4
            a[6] = items.index(craft)
        else:
            logger.debug(f"Unknown craft item '{craft}', sending no functional action")
    elif action.get("attack"):
        a[5] = 3
    elif action.get("use"):
        a[5] = 1

    return a


def normalize_obs(raw: dict) -> dict:
    """Flatten a MineDojo observation into the format the agent expects."""
    obs = {}

    rgb = np.asarray(raw.get("rgb"))
    if rgb.ndim == 3 and rgb.shape[0] == 3:  # channel-first -> HWC
        rgb = np.transpose(rgb, (1, 2, 0))
    obs["rgb"] = np.ascontiguousarray(rgb.astype(np.uint8))

    inv = {}
    raw_inv = raw.get("inventory")
    if isinstance(raw_inv, dict) and "name" in raw_inv:
        for name, qty in zip(raw_inv["name"], raw_inv["quantity"]):
            name = str(name).replace(" ", "_")
            if name in ("air", "none"):
                continue
            inv[name] = inv.get(name, 0) + int(qty)
    obs["inventory"] = inv

    loc = raw.get("location_stats", {}) or {}
    pos = np.asarray(loc.get("pos", [0, 0, 0])).reshape(-1)
    obs["x"], obs["y"], obs["z"] = (float(v) for v in (list(pos) + [0, 0, 0])[:3])
    obs["yaw"] = float(np.asarray(loc.get("yaw", 0)).reshape(-1)[0])
    obs["pitch"] = float(np.asarray(loc.get("pitch", 0)).reshape(-1)[0])
    return obs


class _ActionSpace:
    """Minimal action-space shim so the agent can ask for a no-op."""

    @staticmethod
    def no_op() -> dict:
        return {"forward": 0, "backward": 0, "left": 0, "right": 0, "jump": 0,
                "sprint": 0, "sneak": 0, "attack": 0, "use": 0, "camera": [0, 0]}


class MineDojoWrapper:
    """Wraps a MineDojo env so it accepts dict actions and returns flat observations."""

    def __init__(self, env):
        self.env = env
        self.action_space = _ActionSpace()
        self.craft_items = _craft_items()

    def reset(self) -> dict:
        return normalize_obs(self.env.reset())

    def step(self, action: dict):
        md_action = dict_to_multidiscrete(action, self.craft_items)
        raw, reward, done, info = self.env.step(md_action)
        return normalize_obs(raw), float(reward), bool(done), info

    def close(self):
        self.env.close()
