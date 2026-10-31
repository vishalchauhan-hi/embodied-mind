"""
Skill Library — Stores and retrieves reusable, composable action sequences
discovered through agent experience. Inspired by Voyager's skill library.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Recorded low-level actions only transfer between episodes when they do not
# depend on where things are in the world. Navigation and mining depend on the
# scene, so they are left to the planner and executor.
REPLAYABLE_PREFIXES = ("craft_", "smelt_", "eat_", "equip_")


@dataclass
class Skill:
    """A reusable action sequence for a specific subgoal."""
    name: str
    description: str
    actions: list[dict]
    success_count: int = 0
    fail_count: int = 0
    avg_steps: float = 0.0
    _step_idx: int = field(default=0, repr=False)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.fail_count
        return self.success_count / total if total > 0 else 0.0

    def next_action(self, obs: dict) -> dict:
        """Return the next action in the sequence."""
        if self._step_idx < len(self.actions):
            action = self.actions[self._step_idx]
            self._step_idx += 1
            return action
        # Repeat last action if sequence is exhausted
        return self.actions[-1] if self.actions else {}

    def reset(self):
        self._step_idx = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "actions": self.actions,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "avg_steps": self.avg_steps,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Skill":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SkillLibrary:
    """
    Manages a library of reusable skills discovered through agent experience.
    Skills are keyed by subgoal name and can be retrieved for future episodes.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self.skills: dict[str, Skill] = {}
        self.persist_path = Path(persist_path) if persist_path else None
        self._load_from_disk()
        logger.info(f"SkillLibrary initialized | skills={len(self.skills)}")

    def get(self, subgoal: str) -> Optional[Skill]:
        """Retrieve a skill for the given subgoal, if available and reliable."""
        if not subgoal.startswith(REPLAYABLE_PREFIXES):
            return None
        skill = self.skills.get(subgoal)
        if skill and skill.success_rate >= 0.5:
            skill.reset()
            logger.debug(f"Skill found: {subgoal} (success_rate={skill.success_rate:.0%})")
            return skill
        return None

    def register(self, name: str, description: str, actions: list[dict], success: bool):
        """Register or update a skill in the library."""
        if name in self.skills:
            skill = self.skills[name]
            if success:
                skill.success_count += 1
                # Update action sequence with the successful one
                skill.actions = actions
            else:
                skill.fail_count += 1
        else:
            self.skills[name] = Skill(
                name=name,
                description=description,
                actions=actions,
                success_count=1 if success else 0,
                fail_count=0 if success else 1,
            )

        self._save_to_disk()
        logger.debug(f"Skill registered: {name} (success={success})")

    def discover_from_episode(self, action_history: list, subgoals: list):
        """
        Extract successful action subsequences from an episode and
        register them as skills.
        """
        if not action_history or not subgoals:
            return

        # Group actions by subgoal
        subgoal_actions: dict[str, list] = {}
        for entry in action_history:
            sg = entry.get("subgoal", "unknown")
            if sg not in subgoal_actions:
                subgoal_actions[sg] = []
            subgoal_actions[sg].append(entry["action"])

        # Register successful subgoal action sequences
        for sg, actions in subgoal_actions.items():
            if not actions or not sg.startswith(REPLAYABLE_PREFIXES):
                continue
            # Check if this subgoal had positive reward
            rewards = [e.get("reward", 0) for e in action_history if e.get("subgoal") == sg]
            success = any(r > 0 for r in rewards)

            if success:
                self.register(
                    name=sg,
                    description=f"Learned sequence for '{sg}' ({len(actions)} steps)",
                    actions=actions,
                    success=True,
                )

        logger.info(f"Skill discovery complete | library size={len(self.skills)}")

    def discover_from_completed(self, completed: list):
        """
        Register skills from subgoals that the agent finished in a successful
        episode. Only scene-independent subgoals (see REPLAYABLE_PREFIXES) are kept,
        using the last few actions that completed them.
        """
        for sg, actions in completed:
            if actions and sg.startswith(REPLAYABLE_PREFIXES):
                tail = actions[-3:]
                self.register(sg, f"Learned sequence for '{sg}' ({len(tail)} steps)", tail, success=True)
        logger.info(f"Skill discovery complete | library size={len(self.skills)}")

    def list_skills(self) -> list[dict]:
        """Return a summary of all skills in the library."""
        return [
            {
                "name": s.name,
                "success_rate": f"{s.success_rate:.0%}",
                "uses": s.success_count + s.fail_count,
                "action_length": len(s.actions),
            }
            for s in sorted(self.skills.values(), key=lambda x: x.success_rate, reverse=True)
        ]

    def _save_to_disk(self):
        if not self.persist_path:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v.to_dict() for k, v in self.skills.items()}
        self.persist_path.write_text(json.dumps(data, indent=2))

    def _load_from_disk(self):
        if not self.persist_path or not self.persist_path.exists():
            return
        try:
            data = json.loads(self.persist_path.read_text())
            for name, skill_data in data.items():
                self.skills[name] = Skill.from_dict(skill_data)
            logger.info(f"Loaded {len(self.skills)} skills from {self.persist_path}")
        except Exception as e:
            logger.warning(f"Failed to load skills: {e}")
