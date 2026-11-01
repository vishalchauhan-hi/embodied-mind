"""Offline tests: run with `pytest -q`. No Minecraft, Java or API keys needed."""

import numpy as np
import pytest

from embodied_mind.agent import AgentConfig, EmbodiedMindAgent
from embodied_mind.memory import EpisodicMemory
from embodied_mind.minedojo_adapter import dict_to_multidiscrete, normalize_obs
from embodied_mind.mock_env import MockMinecraftEnv
from embodied_mind.planner import HierarchicalPlanner, find_in_perception


def make_agent(task, tmp_path, memory=True, max_steps=2000, seed=0):
    env = MockMinecraftEnv(task=task, seed=seed, max_steps=max_steps)
    cfg = AgentConfig(model="rule", max_steps=max_steps, memory_enabled=memory,
                      memory_path=str(tmp_path / "mem.json"), skills_path=str(tmp_path / "skills.json"))
    return EmbodiedMindAgent(env, cfg)


def test_harvest_wood_succeeds_offline(tmp_path):
    result = make_agent("harvest_wood", tmp_path).run_episode("harvest_wood")
    assert result["success"]
    assert result["final_inventory"].get("log", 0) >= 1


def test_wooden_pickaxe_chain_offline(tmp_path):
    wins = sum(make_agent("craft_wooden_pickaxe", tmp_path, seed=s).run_episode("craft_wooden_pickaxe")["success"]
               for s in range(5))
    assert wins >= 3


def test_memory_persists_between_episodes(tmp_path):
    make_agent("harvest_wood", tmp_path).run_episode("harvest_wood")
    reloaded = EpisodicMemory(embed_model="hash", persist_path=str(tmp_path / "mem.json"))
    assert reloaded.size == 1
    hits = reloaded.recall("harvest_wood tree log", top_k=1)
    assert hits and hits[0]["task"] == "harvest_wood"


def test_hash_embedding_is_semantic(tmp_path):
    m = EpisodicMemory(embed_model="hash")
    a, b, c = (np.array(m._embed(t)) for t in ["mine log from tree", "tree log mining", "sheep wool shears"])
    assert a @ b > a @ c


def test_action_conversion_matches_minedojo_layout():
    a = dict_to_multidiscrete({"forward": 1, "sprint": 1, "camera": [0, 15], "attack": 1})
    assert a.tolist() == [1, 0, 3, 12, 13, 3, 0, 0]
    noop = dict_to_multidiscrete({})
    assert noop.tolist() == [0, 0, 0, 12, 12, 0, 0, 0]
    craft = dict_to_multidiscrete({"craft": "stick"}, craft_items=["planks", "stick"])
    assert craft[5] == 4 and craft[6] == 1


def test_observation_normalisation():
    raw = {
        "rgb": np.zeros((3, 8, 16), dtype=np.uint8),
        "inventory": {"name": np.array(["log", "air", "log"]), "quantity": np.array([2, 0, 1])},
        "location_stats": {"pos": np.array([1.0, 2.0, 3.0]), "yaw": np.array([90.0]), "pitch": np.array([0.0])},
    }
    obs = normalize_obs(raw)
    assert obs["rgb"].shape == (8, 16, 3)
    assert obs["inventory"] == {"log": 3}
    assert (obs["x"], obs["z"], obs["yaw"]) == (1.0, 3.0, 90.0)


def test_subgoal_completion_uses_baseline():
    p = HierarchicalPlanner(model="rule")
    obs = {"inventory": {"planks": 4}}
    assert not p.check_subgoal_completion("craft_planks", obs, baseline_inventory={"planks": 4})
    assert p.check_subgoal_completion("craft_planks", {"inventory": {"planks": 8}}, baseline_inventory={"planks": 4})


def test_find_in_perception_aliases():
    perc = {"entities": [], "resources": [{"type": "tree", "distance": "far"}, {"type": "tree", "distance": "near"}]}
    assert find_in_perception(perc, "log")["distance"] == "near"
    assert find_in_perception(perc, "sheep") is None


def test_unknown_mock_task():
    with pytest.raises(ValueError):
        MockMinecraftEnv(task="fly_to_moon")


def test_ollama_planner_parses_local_model_reply(monkeypatch):
    import json
    import requests

    sent = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            plan = {"reasoning": "trees are visible", "subgoals": ["find_tree", "mine_log"],
                    "estimated_difficulty": "easy", "key_risks": []}
            return {"message": {"content": json.dumps(plan)}}

    def fake_post(url, json=None, timeout=None):
        sent["url"], sent["body"] = url, json
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    planner = HierarchicalPlanner(model="ollama/qwen2.5vl:7b")
    plan = planner.plan("harvest_wood", inventory={})
    assert plan["subgoals"] == ["find_tree", "mine_log"]
    assert sent["url"].endswith("/api/chat")
    assert sent["body"]["model"] == "qwen2.5vl:7b"


def test_resume_keeps_finished_episodes_and_memory(tmp_path):
    from embodied_mind.run import run
    full = run("harvest_wood", model="rule", env_type="mock", num_episodes=3, max_steps=300,
               seed=7, output_dir=str(tmp_path / "full"))
    run("harvest_wood", model="rule", env_type="mock", num_episodes=2, max_steps=300,
        seed=7, output_dir=str(tmp_path / "resumed"))
    resumed = run("harvest_wood", model="rule", env_type="mock", num_episodes=3, max_steps=300,
                  seed=7, output_dir=str(tmp_path / "resumed"), start_episode=2)
    assert [e["episode"] for e in resumed["episodes"]] == [0, 1, 2]
    assert [e["steps"] for e in resumed["episodes"]] == [e["steps"] for e in full["episodes"]]
    assert EpisodicMemory(persist_path=str(next((tmp_path / "resumed").glob("*_memory.json")))).size == 3
