"""
MockMinecraftEnv: a small offline stand-in for MineDojo.

It exists so the full agent loop (perceive, recall, plan, act, reflect) can be run
and tested on any machine, with no Minecraft, Java or API keys. It is a test
harness, not a benchmark. Numbers from this environment say nothing about
performance in MineDojo and must not be reported as MineDojo results.

World model: a few objects (trees, stone, sheep) placed around the agent at a
distance and bearing. Turning the camera rotates bearings, moving forward closes
the distance to objects in front, attacking a near object in front collects its
item, and crafting follows a small recipe table.
"""

import random
from dataclasses import dataclass

import numpy as np

from .minedojo_adapter import _ActionSpace

SOURCES = {  # object kind -> (item dropped, hits needed, tool required)
    "tree": ("log", 3, None),
    "stone": ("cobblestone", 4, "wooden_pickaxe"),
    "sheep": ("wool", 2, None),
}

RECIPES = {  # item -> (ingredients, amount made, needs crafting table)
    "planks": ({"log": 1}, 4, False),
    "stick": ({"planks": 2}, 4, False),
    "crafting_table": ({"planks": 4}, 1, False),
    "wooden_pickaxe": ({"planks": 3, "stick": 2}, 1, True),
    "stone_pickaxe": ({"cobblestone": 3, "stick": 2}, 1, True),
}

TASK_GOALS = {  # task -> (goal item, amount)
    "harvest_wood": ("log", 1),
    "harvest_wool": ("wool", 1),
    "harvest_cobblestone": ("cobblestone", 1),
    "craft_wooden_pickaxe": ("wooden_pickaxe", 1),
    "craft_stone_pickaxe": ("stone_pickaxe", 1),
}


@dataclass
class _Obj:
    kind: str
    dist: float
    bearing: float  # degrees, 0 = straight ahead, positive = to the right
    hits: int = 0


class MockMinecraftEnv:
    """Offline environment with a gym-style reset/step API and dict actions."""

    def __init__(self, task: str = "harvest_wood", seed: int = 0, max_steps: int = 3000):
        if task not in TASK_GOALS:
            raise ValueError(f"Unknown mock task '{task}'. Options: {sorted(TASK_GOALS)}")
        self.task = task
        self.goal_item, self.goal_amount = TASK_GOALS[task]
        self.seed = seed
        self.max_steps = max_steps
        self.action_space = _ActionSpace()
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------ API
    def reset(self) -> dict:
        self.rng = random.Random(self.seed)
        self.t = 0
        self.inventory: dict = {}
        self.yaw = 0.0
        self.objects = [self._spawn(k) for k in ["tree", "tree", "tree", "stone", "stone", "sheep"]]
        return self._obs()

    def step(self, action: dict):
        self.t += 1
        yaw_turn = float(action.get("camera", [0, 0])[1])
        if yaw_turn:
            self.yaw = (self.yaw + yaw_turn) % 360
            for o in self.objects:
                o.bearing = self._wrap(o.bearing - yaw_turn)

        if action.get("forward"):
            speed = 1.5 if action.get("sprint") else 1.0
            for o in self.objects:
                if abs(o.bearing) < 30:
                    o.dist = max(1.0, o.dist - speed)
                else:
                    o.dist += 0.1 * speed

        info = {}
        if action.get("craft"):
            info["craft_ok"] = self._craft(action["craft"])
        elif action.get("attack") or action.get("use"):
            self._hit()

        done_goal = self.inventory.get(self.goal_item, 0) >= self.goal_amount
        reward = 1.0 if done_goal else 0.0
        done = done_goal or self.t >= self.max_steps
        info["success"] = done_goal
        return self._obs(), reward, done, info

    def close(self):
        pass

    # ------------------------------------------------------------ internals
    def _spawn(self, kind: str) -> _Obj:
        return _Obj(kind, dist=self.rng.uniform(8, 22), bearing=self.rng.uniform(-180, 180))

    @staticmethod
    def _wrap(deg: float) -> float:
        return (deg + 180) % 360 - 180

    def _front_near(self):
        cands = [o for o in self.objects if o.dist <= 2.0 and abs(o.bearing) < 20]
        return min(cands, key=lambda o: o.dist) if cands else None

    def _hit(self):
        o = self._front_near()
        if o is None:
            return
        item, hits_needed, tool = SOURCES[o.kind]
        if tool and self.inventory.get(tool, 0) == 0:
            return
        o.hits += 1
        if o.hits >= hits_needed:
            self.inventory[item] = self.inventory.get(item, 0) + 1
            self.objects.remove(o)
            self.objects.append(self._spawn(o.kind))

    def _craft(self, item: str) -> bool:
        if item not in RECIPES:
            return False
        needs, amount, table = RECIPES[item]
        if table and self.inventory.get("crafting_table", 0) == 0:
            return False
        if any(self.inventory.get(k, 0) < v for k, v in needs.items()):
            return False
        for k, v in needs.items():
            self.inventory[k] -= v
        self.inventory[item] = self.inventory.get(item, 0) + amount
        return True

    def _visible(self) -> list:
        out = []
        for o in self.objects:
            if abs(o.bearing) > 60 or o.dist > 25:
                continue
            dist = "near" if o.dist <= 2.0 else ("medium" if o.dist <= 10 else "far")
            direction = "center" if abs(o.bearing) < 15 else ("left" if o.bearing < 0 else "right")
            out.append({"type": o.kind, "distance": dist, "direction": direction})
        return out

    def _obs(self) -> dict:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:32] = (120, 170, 230)   # sky
        frame[32:] = (90, 150, 60)     # grass
        for v in self._visible():
            colour = {"tree": (60, 100, 40), "stone": (128, 128, 128), "sheep": (235, 235, 235)}[v["type"]]
            x0 = {"left": 8, "center": 28, "right": 48}[v["direction"]]
            size = {"near": 14, "medium": 8, "far": 4}[v["distance"]]
            frame[32 - size:32 + size // 2, x0:x0 + size] = colour
        return {
            "rgb": frame,
            "inventory": {k: v for k, v in self.inventory.items() if v > 0},
            "x": 0.0, "y": 64.0, "z": 0.0,
            "yaw": self.yaw, "pitch": 0.0,
            "mock_visible": self._visible(),
        }
