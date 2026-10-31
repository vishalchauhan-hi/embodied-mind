"""
EmbodiedMind Agent — Main agent loop integrating perception, planning, memory, and action.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .perceiver import VLMPerceiver
from .planner import HierarchicalPlanner
from .memory import EpisodicMemory
from .skills import SkillLibrary
from .action_executor import ActionExecutor

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for EmbodiedMind agent."""
    model: str = "gemini-2.5-flash"
    max_steps: int = 3000
    memory_enabled: bool = True
    memory_top_k: int = 3
    replan_interval: int = 10          # Re-plan every N steps
    replan_on_failure: bool = True     # Re-plan immediately on action failure
    perception_interval: int = 5       # Run VLM perception every N steps
    skill_discovery: bool = True
    verbose: bool = False
    embed_model: str = "text-embedding-3-small"   # falls back to offline hashing without OPENAI_API_KEY
    memory_path: Optional[str] = None             # JSON file; set it to keep memory across episodes
    skills_path: Optional[str] = None


@dataclass
class AgentState:
    """Tracks the agent's internal state across the episode."""
    step: int = 0
    current_goal: str = ""
    current_subgoals: list = field(default_factory=list)
    active_subgoal_idx: int = 0
    last_perception: Optional[dict] = None
    last_plan: Optional[str] = None
    total_reward: float = 0.0
    action_history: list = field(default_factory=list)
    failures: list = field(default_factory=list)
    subgoal_start_inventory: dict = field(default_factory=dict)
    active_skill: Optional[object] = None
    success: bool = False
    subgoal_actions: list = field(default_factory=list)       # actions since current subgoal began
    completed_subgoals: list = field(default_factory=list)    # (subgoal, actions) that succeeded
    recover_steps: int = 0  # steps left in a "back off and turn" manoeuvre after getting stuck


class EmbodiedMindAgent:
    """
    Multimodal embodied agent that perceives, reasons, plans, and acts
    in MineDojo environments using VLM perception, LLM planning,
    and episodic memory for in-context learning.
    """

    def __init__(self, env, config: Optional[AgentConfig] = None):
        self.env = env
        self.cfg = config or AgentConfig()

        # Core modules
        self.perceiver = VLMPerceiver(model=self.cfg.model)
        self.planner = HierarchicalPlanner(model=self.cfg.model)
        self.memory = (EpisodicMemory(embed_model=self.cfg.embed_model, persist_path=self.cfg.memory_path)
                       if self.cfg.memory_enabled else None)
        self.skills = SkillLibrary(persist_path=self.cfg.skills_path)
        self.executor = ActionExecutor(env)

        # State
        self.state = AgentState()

        logger.info(
            f"EmbodiedMind initialized | model={self.cfg.model} "
            f"memory={self.cfg.memory_enabled}"
        )

    def run_episode(self, task: str, seed: int = 0) -> dict:
        """Run a single episode for the given task."""
        logger.info(f"Starting episode | task='{task}' seed={seed}")

        np.random.seed(seed)  # makes the executor's exploration turns repeatable
        obs = self.env.reset()
        self.state = AgentState(current_goal=task)
        episode_log = []
        done = False

        while not done and self.state.step < self.cfg.max_steps:
            result = self._step(obs)
            obs, reward, done, info = result["env_step"]

            self.state.total_reward += reward
            self.state.step += 1
            if (isinstance(info, dict) and info.get("success")) or reward > 0:
                self.state.success = True
            episode_log.append(result["log"])

            if self.cfg.verbose and self.state.step % 50 == 0:
                logger.info(
                    f"  step={self.state.step} reward={self.state.total_reward:.2f} "
                    f"subgoal={self._active_subgoal()}"
                )

        # Post-episode reflection and memory storage
        summary = self._reflect_on_episode(episode_log)

        return {
            "task": task,
            "steps": self.state.step,
            "total_reward": self.state.total_reward,
            "success": self.state.success,
            "final_inventory": dict(obs.get("inventory", {})),
            "summary": summary,
            "log": episode_log,
        }

    def _step(self, obs) -> dict:
        """Single agent step: perceive → recall → plan → act."""
        log = {"step": self.state.step}

        # 1. Perceive (at configured interval or first step)
        if self._should_perceive():
            perception = self.perceiver.perceive(obs["rgb"], obs)
            self.state.last_perception = perception
            log["perception"] = perception

        # 2. Recall relevant memories
        memory_context = ""
        if self.memory and self._should_replan():
            summary = (self.state.last_perception or {}).get("summary", "")
            query = f"{self.state.current_goal} | {summary}"
            memories = self.memory.recall(query, top_k=self.cfg.memory_top_k)
            if memories:
                memory_context = self._format_memories(memories)
                log["memories_recalled"] = len(memories)

        # 3. Plan (at configured interval, on failure, or first step)
        if self._should_replan():
            plan = self.planner.plan(
                goal=self.state.current_goal,
                perception=self.state.last_perception,
                memory_context=memory_context,
                inventory=obs.get("inventory", {}),
                failures=self.state.failures[-5:],  # Last 5 failures
            )
            self.state.current_subgoals = plan["subgoals"]
            self._set_subgoal(0, obs)
            self.state.last_plan = plan["reasoning"]
            log["plan"] = plan

        # 4. Act — execute the current subgoal
        subgoal = self._active_subgoal()
        if subgoal:
            if self.state.recover_steps > 0:
                # Recently stuck: turn away and walk a few steps before trying again
                self.state.recover_steps -= 1
                action = self.executor.unstick_action(self.state.recover_steps)
                log["action_source"] = "recovery"
            else:
                # Check the skill library once per subgoal, then replay the skill
                if self.state.active_skill is None:
                    self.state.active_skill = self.skills.get(subgoal) or False
                if self.state.active_skill:
                    action = self.state.active_skill.next_action(obs)
                    log["action_source"] = "skill_library"
                else:
                    action = self.executor.subgoal_to_action(subgoal, obs, self.state.last_perception)
                    log["action_source"] = "planner"

            env_step = self.env.step(action)
            obs_next, reward, done, info = env_step
            log["action"] = action
            log["reward"] = reward

            # Track action result
            self.state.action_history.append({
                "subgoal": subgoal,
                "action": action,
                "reward": reward,
                "step": self.state.step,
            })

            # Check subgoal completion or failure
            log["subgoal"] = subgoal
            if log["action_source"] != "recovery":
                self.state.subgoal_actions.append(action)
            if reward > 0 or self._subgoal_completed(subgoal, obs_next):
                self.state.completed_subgoals.append((subgoal, list(self.state.subgoal_actions)))
                self._advance_subgoal(obs_next)
            elif self._subgoal_stuck():
                self.state.failures.append({
                    "subgoal": subgoal,
                    "step": self.state.step,
                    "context": self.state.last_perception,
                })
                self.state.recover_steps = 8
                if self.state.active_skill:  # replayed skill did not work here; lower its success rate
                    self.skills.register(subgoal, self.state.active_skill.description,
                                         self.state.active_skill.actions, success=False)
                self._advance_subgoal(obs_next)
        else:
            # No subgoal — take noop
            env_step = self.env.step(self.env.action_space.no_op())
            log["action"] = "no_op"

        log["total_reward"] = self.state.total_reward
        return {"env_step": env_step, "log": log}

    # --- Perception scheduling ---
    def _should_perceive(self) -> bool:
        # Look again right after steering toward a target, so the next turn
        # uses a fresh view instead of repeating a stale direction.
        return (
            self.executor.last_was_steer
            or self.state.step == 0
            or self.state.step % self.cfg.perception_interval == 0
            or self.state.last_perception is None
        )

    # --- Planning scheduling ---
    def _should_replan(self) -> bool:
        if self.state.step == 0 or self._active_subgoal() is None:
            return True
        # Periodic and failure-triggered replanning only help a planner that reads the
        # scene; the rule planner would restart its fixed list, so it replans only
        # when the list runs out.
        if self.planner.api == "rule":
            return False
        if self.state.step % self.cfg.replan_interval == 0:
            return True
        if self.cfg.replan_on_failure and self.state.failures:
            last_fail = self.state.failures[-1]
            if last_fail["step"] == self.state.step - 1:
                return True
        return False

    # --- Subgoal management ---
    def _active_subgoal(self) -> Optional[str]:
        if self.state.active_subgoal_idx < len(self.state.current_subgoals):
            return self.state.current_subgoals[self.state.active_subgoal_idx]
        return None

    def _set_subgoal(self, idx: int, obs: dict):
        self.state.active_subgoal_idx = idx
        self.state.subgoal_start_inventory = dict(obs.get("inventory", {}))
        self.state.active_skill = None
        self.state.subgoal_actions = []

    def _advance_subgoal(self, obs: dict):
        self._set_subgoal(self.state.active_subgoal_idx + 1, obs)
        if self.state.active_subgoal_idx >= len(self.state.current_subgoals):
            logger.debug("All subgoals exhausted, will re-plan next step")

    def _subgoal_completed(self, subgoal: str, obs: dict) -> bool:
        """Inventory-based check for collect/craft, perception-based for find/approach."""
        return self.planner.check_subgoal_completion(
            subgoal, obs, self.state.last_perception, self.state.subgoal_start_inventory)

    def _subgoal_stuck(self, patience: int = 20) -> bool:
        """Check if agent has been on the same subgoal too long with no reward."""
        recent = self.state.action_history[-patience:]
        if len(recent) < patience:
            return False
        subgoal = self._active_subgoal()
        if subgoal.startswith(("find_", "explore_")):
            patience = patience * 10  # searching legitimately takes longer
            recent = self.state.action_history[-patience:]
            if len(recent) < patience:
                return False
        same_subgoal = all(a["subgoal"] == subgoal for a in recent)
        no_reward = all(a["reward"] <= 0 for a in recent)
        return same_subgoal and no_reward

    # --- Memory helpers ---
    def _format_memories(self, memories: list) -> str:
        parts = []
        for i, m in enumerate(memories):
            outcome = "SUCCESS" if m.get("reward", 0) > 0 else "FAILED"
            parts.append(
                f"[Memory {i+1}] Task: {m['task']} | Outcome: {outcome}\n"
                f"  Plan: {m['plan']}\n"
                f"  Lesson: {m.get('lesson', 'N/A')}"
            )
        return "\n".join(parts)

    # --- Post-episode reflection ---
    def _reflect_on_episode(self, episode_log: list) -> str:
        """Generate a reflection and store experience in memory."""
        summary = self.planner.reflect(
            goal=self.state.current_goal,
            total_reward=self.state.total_reward,
            steps=self.state.step,
            failures=self.state.failures,
            action_count=len(self.state.action_history),
        )

        if self.memory:
            self.memory.store({
                "task": self.state.current_goal,
                "observation": str(self.state.last_perception),
                "plan": self.state.last_plan or "",
                "actions": [a["action"] for a in self.state.action_history[-20:]],
                "outcome": summary,
                "reward": self.state.total_reward,
                "lesson": summary,
            })
            logger.info(f"Stored episode in memory | total memories={self.memory.size}")

        # Discover skills from successful action sequences
        if self.cfg.skill_discovery and self.state.success:
            self.skills.discover_from_completed(self.state.completed_subgoals)

        return summary
