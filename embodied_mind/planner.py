"""
Hierarchical LLM Planner — Decomposes high-level goals into ordered subgoals
and translates them into executable action sequences.
"""

import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

OFFLINE_MODELS = {"rule", "mock", "offline"}

PLANNING_SYSTEM_PROMPT = """You are the planning module of an embodied AI agent in Minecraft (MineDojo).
Your job is to decompose high-level task goals into an ordered list of subgoals that
the agent can execute step by step.

RULES:
1. Each subgoal must be specific and achievable (e.g., "find_cow", "approach_cow", "use_bucket_on_cow")
2. Consider the agent's current perception, inventory, and past failures
3. If memory of past attempts is provided, learn from successes and avoid repeated failures
4. Subgoals should follow Minecraft game logic (e.g., need crafting table before crafting tools)
5. Always include navigation subgoals when targets are far away

Respond ONLY with a JSON object (no markdown):
{
  "reasoning": "Brief explanation of your plan strategy",
  "subgoals": ["subgoal_1", "subgoal_2", "subgoal_3", ...],
  "estimated_difficulty": "easy/medium/hard",
  "key_risks": ["risk_1", "risk_2"]
}
"""

REFLECTION_SYSTEM_PROMPT = """You are reviewing an embodied agent's performance on a Minecraft task.
Analyze the results and generate a brief lesson learned.
Respond with a single paragraph (2-3 sentences) summarizing what worked, what failed, and
what the agent should do differently next time."""

# Minecraft knowledge for subgoal→action mapping
MINECRAFT_RECIPES = {
    "craft_planks": {"requires": ["log"], "gives": "planks", "tool": None},
    "craft_stick": {"requires": ["planks"], "gives": "stick", "tool": None},
    "craft_crafting_table": {"requires": ["planks"], "gives": "crafting_table", "tool": None},
    "craft_wooden_pickaxe": {"requires": ["planks", "stick"], "gives": "wooden_pickaxe", "tool": "crafting_table"},
    "craft_stone_pickaxe": {"requires": ["cobblestone", "stick"], "gives": "stone_pickaxe", "tool": "crafting_table"},
    "craft_furnace": {"requires": ["cobblestone"], "gives": "furnace", "tool": "crafting_table"},
    "craft_bucket": {"requires": ["iron_ingot"], "gives": "bucket", "tool": "crafting_table"},
}


class HierarchicalPlanner:
    """
    Two-level planner:
    - Strategist (LLM): Decomposes task goal → ordered subgoals
    - Tactician (rule-based + LLM): Maps subgoal → MineDojo action sequence
    """

    def __init__(self, model: str = "gemini-3.8-flash"):
        self.model = model
        self._init_client()
        logger.info(f"HierarchicalPlanner initialized | model={model}")

    def _init_client(self):
        if self.model in OFFLINE_MODELS:
            self.client, self.api = None, "rule"
        elif "gemini" in self.model:
            from .perceiver import _gemini_client
            self.client, self.api = _gemini_client(), "gemini"
        elif "gpt" in self.model:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.api = "openai"
        else:
            raise ValueError(f"Unsupported model for planner: {self.model}")

    def plan(
        self,
        goal: str,
        perception: Optional[dict] = None,
        memory_context: str = "",
        inventory: Optional[dict] = None,
        failures: Optional[list] = None,
    ) -> dict:
        """
        Generate a hierarchical plan for the given goal.
        
        Returns:
            dict with keys: reasoning, subgoals, estimated_difficulty, key_risks
        """
        if self.api == "rule":
            return self._fallback_plan(goal, inventory or {})

        prompt = self._build_planning_prompt(goal, perception, memory_context, inventory, failures)

        try:
            response = self._call_llm(prompt, system=PLANNING_SYSTEM_PROMPT)
            plan = self._parse_plan(response)
            logger.info(f"Plan generated: {len(plan['subgoals'])} subgoals | {plan['reasoning'][:80]}")
            return plan
        except Exception as e:
            logger.warning(f"Planning failed: {e}. Using fallback plan.")
            return self._fallback_plan(goal, inventory or {})

    def _build_planning_prompt(self, goal, perception, memory_context, inventory, failures):
        parts = [f"GOAL: {goal}"]

        if perception:
            parts.append(f"\nCURRENT PERCEPTION:\n{json.dumps(perception, indent=2)}")
        if inventory:
            parts.append(f"\nINVENTORY: {json.dumps(inventory)}")
        if failures:
            fail_strs = [f"- Failed '{f['subgoal']}' at step {f['step']}" for f in failures]
            parts.append(f"\nPAST FAILURES (avoid repeating):\n" + "\n".join(fail_strs))
        if memory_context:
            parts.append(f"\nRELEVANT PAST EXPERIENCES:\n{memory_context}")

        parts.append(f"\nMINECRAFT RECIPES AVAILABLE:\n{json.dumps(MINECRAFT_RECIPES, indent=2)}")
        parts.append("\nGenerate an optimal plan.")

        return "\n".join(parts)

    def check_subgoal_completion(self, subgoal: str, obs: dict, perception: Optional[dict] = None,
                                 baseline_inventory: Optional[dict] = None) -> bool:
        """
        Heuristic check if a subgoal has been completed. Collect/craft subgoals are
        done when the item count rose above its count when the subgoal started;
        find/approach subgoals use the latest perception.
        """
        inv = obs.get("inventory", {})
        base = baseline_inventory or {}

        for prefix in ["harvest_", "obtain_", "mine_", "collect_", "craft_", "attack_"]:
            if subgoal.startswith(prefix):
                item = ITEM_FROM_SUBGOAL.get(subgoal, subgoal[len(prefix):])
                return inv.get(item, 0) > base.get(item, 0)

        if subgoal.startswith("find_") or subgoal.startswith("approach_"):
            target = subgoal.split("_", 1)[1]
            seen = find_in_perception(perception, target)
            if seen is None:
                return False
            return True if subgoal.startswith("find_") else seen.get("distance") == "near"

        return False

    def reflect(
        self,
        goal: str,
        total_reward: float,
        steps: int,
        failures: list,
        action_count: int,
    ) -> str:
        """Generate a post-episode reflection for memory storage."""
        outcome = "SUCCESS" if total_reward > 0 else "FAILED"
        prompt = (
            f"Task: {goal}\n"
            f"Outcome: {outcome} (reward={total_reward:.2f})\n"
            f"Steps taken: {steps}, Actions: {action_count}\n"
            f"Failures encountered: {len(failures)}\n"
        )
        if failures:
            prompt += "Failure details:\n"
            for f in failures[:5]:
                prompt += f"  - '{f['subgoal']}' at step {f['step']}\n"

        prompt += "\nGenerate a brief lesson learned (2-3 sentences)."

        if self.api == "rule":
            if failures:
                stuck = ", ".join(sorted({f["subgoal"] for f in failures})[:3])
                return f"Episode {outcome} in {steps} steps. Got stuck on: {stuck}."
            return f"Episode {outcome} in {steps} steps with no stuck subgoals."

        try:
            return self._call_llm(prompt, system=REFLECTION_SYSTEM_PROMPT)
        except Exception as e:
            logger.warning(f"Reflection failed: {e}")
            return f"Episode {outcome}. Reward: {total_reward:.2f}. Steps: {steps}."

    def _call_llm(self, prompt: str, system: str = "") -> str:
        if self.api == "gemini":
            kind, client = self.client
            if kind == "genai":
                from google.genai import types
                response = client.models.generate_content(
                    model=self.model, contents=prompt,
                    config=types.GenerateContentConfig(system_instruction=system or None),
                )
            else:
                response = client.GenerativeModel(self.model, system_instruction=system or None).generate_content(prompt)
            return response.text
        elif self.api == "openai":
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=800,
            )
            return response.choices[0].message.content
        raise RuntimeError(f"No LLM backend for api={self.api}")

    def _parse_plan(self, text: str) -> dict:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
        
        result = json.loads(cleaned)
        assert "subgoals" in result and len(result["subgoals"]) > 0
        return result

    def _fallback_plan(self, goal: str, inventory: Optional[dict] = None) -> dict:
        """Rule-based plan, used offline or when LLM planning fails. Skips steps already done."""
        g = goal.lower()
        inv = inventory or {}
        get_planks = ["find_tree", "approach_tree", "mine_log", "craft_planks"]
        wood_tools = get_planks * 3 + ["craft_stick", "craft_crafting_table", "craft_wooden_pickaxe"]

        if "milk" in g:
            subgoals = ["find_iron_ore", "mine_iron_ore", "craft_furnace",
                        "smelt_iron_ingot", "craft_bucket", "find_cow",
                        "approach_cow", "use_bucket_on_cow"]
        elif "wool" in g:
            subgoals = ["find_sheep", "approach_sheep", "attack_sheep"]
        elif "stone_pickaxe" in g:
            subgoals = wood_tools + ["find_stone", "approach_stone", "mine_cobblestone",
                                     "mine_cobblestone", "mine_cobblestone", "craft_stone_pickaxe"]
        elif "wooden_pickaxe" in g:
            subgoals = list(wood_tools)
        elif "cobblestone" in g or "stone" in g:
            subgoals = wood_tools + ["find_stone", "approach_stone", "mine_cobblestone"]
        elif "wood" in g or "log" in g:
            subgoals = ["find_tree", "approach_tree", "mine_log"]
        elif "survive" in g:
            subgoals = ["find_shelter_materials", "build_shelter", "find_food", "eat_food"]
        else:
            subgoals = ["explore_environment", "identify_target", "approach_target",
                        "interact_with_target"]

        # Drop craft steps whose product is already held (keeps replans from repeating work)
        held = {k for k, v in inv.items() if v > 0}
        subgoals = [s for s in subgoals
                    if not (s.startswith("craft_") and s[len("craft_"):] in held and s != "craft_planks")]

        return {
            "reasoning": f"Rule-based plan for: {goal}",
            "subgoals": subgoals,
            "estimated_difficulty": "medium",
            "key_risks": ["rule-based plan may not fit the scene"],
        }


# Item a subgoal should produce, when it differs from the subgoal suffix
ITEM_FROM_SUBGOAL = {
    "attack_sheep": "wool",
    "mine_tree": "log",
    "mine_stone": "cobblestone",
    "craft_sticks": "stick",
}

# Which visible thing to look for when a subgoal names an item
TARGET_ALIASES = {
    "log": "tree", "wood": "tree", "tree": "tree",
    "cobblestone": "stone", "stone": "stone",
    "wool": "sheep", "sheep": "sheep",
    "milk": "cow", "cow": "cow",
}


def find_in_perception(perception: Optional[dict], target: str) -> Optional[dict]:
    """Return the closest perceived entity/resource matching target, or None."""
    if not perception:
        return None
    want = TARGET_ALIASES.get(target, target)
    order = {"near": 0, "medium": 1, "far": 2}
    hits = [x for x in perception.get("entities", []) + perception.get("resources", [])
            if isinstance(x, dict) and want in str(x.get("type", "")).lower()]
    return min(hits, key=lambda x: order.get(x.get("distance"), 3)) if hits else None
