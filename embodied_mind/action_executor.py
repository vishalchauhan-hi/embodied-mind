"""
Action Executor — Translates high-level subgoals into MineDojo low-level actions.
Handles navigation, interaction, combat, and crafting action sequences.
"""

import logging
from typing import Optional
from enum import Enum

import numpy as np

from .planner import find_in_perception

logger = logging.getLogger(__name__)

# Planner subgoal suffix -> Minecraft item id, where they differ
CRAFT_NAMES = {"sticks": "stick", "plank": "planks"}


class ActionType(Enum):
    """MineDojo action categories."""
    MOVE_FORWARD = "forward"
    MOVE_BACKWARD = "backward"
    STRAFE_LEFT = "left"
    STRAFE_RIGHT = "right"
    JUMP = "jump"
    ATTACK = "attack"
    USE = "use"
    CRAFT = "craft"
    CAMERA_UP = "camera_up"
    CAMERA_DOWN = "camera_down"
    CAMERA_LEFT = "camera_left"
    CAMERA_RIGHT = "camera_right"
    NO_OP = "no_op"


# Mapping from subgoal patterns to action sequences
SUBGOAL_ACTION_MAP = {
    "find_": "explore",        # Wander and look around
    "approach_": "navigate",   # Move toward target
    "mine_": "mine",           # Look down + attack
    "attack_": "combat",       # Face target + attack
    "craft_": "craft",         # Open inventory + craft
    "use_": "interact",        # Face target + use
    "eat_": "interact",
    "explore_": "explore",
    "build_": "place",
    "place_": "place",
    "smelt_": "interact",
}


class ActionExecutor:
    """
    Translates subgoals into sequences of MineDojo-compatible actions.
    Uses a combination of heuristic behaviors and observation-based navigation.
    """

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self._explore_step = 0  # Counter for exploration pattern
        self.last_was_steer = False  # True when the last action turned toward a perceived target

    def subgoal_to_action(self, subgoal: str, obs: dict, perception: Optional[dict] = None) -> dict:
        """
        Convert a subgoal string to an action dict (MineDojoWrapper turns it
        into MineDojo's MultiDiscrete format).

        Args:
            subgoal: High-level subgoal string (e.g., "find_cow", "mine_log")
            obs: Current observation
            perception: Latest VLM scene description, used to steer toward targets

        Returns:
            action dict
        """
        behavior = self._classify_subgoal(subgoal)
        target = subgoal.split("_", 1)[1] if "_" in subgoal else subgoal
        self.last_was_steer = False

        if behavior == "explore":
            seen = find_in_perception(perception, target)
            return self._steer_action(seen) if seen else self._explore_action(obs)
        elif behavior == "navigate":
            return self._navigate_action(subgoal, obs, perception)
        elif behavior in ("mine", "combat"):
            seen = find_in_perception(perception, target)
            if seen is None:
                return self._explore_action(obs)
            if seen.get("distance") != "near" or seen.get("direction") != "center":
                return self._steer_action(seen)
            return self._mine_action(obs) if behavior == "mine" else self._combat_action(obs)
        elif behavior == "craft":
            return self._craft_action(subgoal, obs)
        elif behavior == "interact":
            return self._interact_action(obs)
        elif behavior == "place":
            return self._place_action(obs)
        else:
            return self._no_op()

    def _classify_subgoal(self, subgoal: str) -> str:
        """Determine which behavior to execute for a given subgoal."""
        for prefix, behavior in SUBGOAL_ACTION_MAP.items():
            if subgoal.startswith(prefix):
                return behavior
        return "explore"

    def _explore_action(self, obs: dict) -> dict:
        """
        Systematic exploration: walk forward, sweep the camera, turn to a new
        heading now and then, and jump to clear small obstacles.
        Camera moves are multiples of 15 degrees, MineDojo's camera bin size.
        """
        self._explore_step += 1
        action = self._no_op()
        phase = self._explore_step % 40

        if phase < 20:
            action["forward"] = 1
            action["sprint"] = 1
        elif phase < 26:
            action["camera"] = [0, 30]          # look around
        elif phase < 30:
            action["forward"] = 1
        elif phase < 35:
            action["camera"] = [0, int(np.random.choice([-45, 45]))]  # new heading
        else:
            action["forward"] = 1
            action["jump"] = 1
        return action

    def unstick_action(self, steps_left: int) -> dict:
        """Recovery after a stuck subgoal: turn 90 degrees, then walk and jump forward."""
        self.last_was_steer = False
        action = self._no_op()
        if steps_left >= 6:
            action["camera"] = [0, 45]
        else:
            action["forward"] = 1
            action["jump"] = 1
        return action

    def _steer_action(self, seen: dict) -> dict:
        """Turn toward a perceived target and walk to it."""
        action = self._no_op()
        self.last_was_steer = True
        direction = seen.get("direction", "center")
        if direction == "left":
            action["camera"] = [0, -15]
        elif direction == "right":
            action["camera"] = [0, 15]
        if seen.get("distance") != "near":
            action["forward"] = 1
            if seen.get("distance") == "far":
                action["sprint"] = 1
        return action

    def _navigate_action(self, subgoal: str, obs: dict, perception: Optional[dict] = None) -> dict:
        """Move toward the subgoal's target using the latest perception; explore if not seen."""
        target = subgoal.split("_", 1)[1] if "_" in subgoal else subgoal
        seen = find_in_perception(perception, target)
        if seen is None:
            return self._explore_action(obs)
        return self._steer_action(seen)

    def _mine_action(self, obs: dict) -> dict:
        """Mine/break a block: look slightly down and attack."""
        action = self._no_op()
        action["attack"] = 1
        return action

    def _combat_action(self, obs: dict) -> dict:
        """Engage in combat: face target and attack."""
        action = self._no_op()
        action["attack"] = 1
        return action

    def _craft_action(self, subgoal: str, obs: dict) -> dict:
        """
        Craft an item. MineDojo handles crafting through specific action indices.
        This is simplified — real implementation maps to MineDojo craft commands.
        """
        action = self._no_op()

        item = subgoal.replace("craft_", "")
        action["craft"] = CRAFT_NAMES.get(item, item)  # MineDojoWrapper maps the name to its craft index
        return action

    def _interact_action(self, obs: dict) -> dict:
        """Use/interact with the block or entity in front of the agent."""
        action = self._no_op()
        action["use"] = 1
        return action

    def _place_action(self, obs: dict) -> dict:
        """Place a block."""
        action = self._no_op()
        action["use"] = 1
        # Look slightly down for block placement
        action["camera"] = [15, 0]
        return action

    def _no_op(self) -> dict:
        """Return a no-operation action dict."""
        return {
            "forward": 0,
            "backward": 0,
            "left": 0,
            "right": 0,
            "jump": 0,
            "sprint": 0,
            "sneak": 0,
            "attack": 0,
            "use": 0,
            "camera": [0, 0],
        }
