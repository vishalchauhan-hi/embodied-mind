"""
EmbodiedMind single-task runner.

Usage:
    # Real MineDojo run (needs MineDojo + an API key, see docs/RUNNING.md)
    python -m embodied_mind.run --task harvest_wool --model gemini-3.8-flash --num_episodes 3

    # Offline pipeline check, no Minecraft or API key needed
    python -m embodied_mind.run --env mock --model rule --task harvest_wood
"""

import argparse
import copy
import json
import logging
import re
import time
from pathlib import Path

from .agent import EmbodiedMindAgent, AgentConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("embodied_mind")

# Short task names -> MineDojo programmatic task ids
TASK_MAP = {
    "harvest_wool": "harvest_wool_with_shears_and_sheep",
    "harvest_milk": "harvest_milk_with_empty_bucket_and_cow",
    "harvest_wood": "harvest_1_log",
    "harvest_cobblestone": "harvest_1_cobblestone_with_wooden_pickaxe",
    # MineDojo has no 5-day survival id; "survival" is the bare-handed one (its built-in target is 100 days)
    "survive_5_days": "survival",
    "craft_wooden_pickaxe": "techtree_from_barehand_to_wooden_pickaxe",
    "craft_stone_pickaxe": "techtree_from_barehand_to_stone_pickaxe",
}


def create_env(task: str, seed: int = 0, env_type: str = "minedojo", max_steps: int = 3000):
    """Create an environment. 'mock' needs no Minecraft; 'minedojo' needs MineDojo installed."""
    if env_type == "mock":
        from .mock_env import MockMinecraftEnv
        env = MockMinecraftEnv(task=task, seed=seed, max_steps=max_steps)
        logger.info(f"Created mock env | task={task} seed={seed}")
        return env

    import minedojo  # imported here so mock runs work without MineDojo
    from .minedojo_adapter import MineDojoWrapper

    minedojo_task = TASK_MAP.get(task, task)
    valid = _minedojo_task_ids()
    if valid and minedojo_task not in valid:
        import difflib
        close = difflib.get_close_matches(minedojo_task, valid, n=5, cutoff=0.3)
        raise ValueError(f"'{minedojo_task}' is not a MineDojo task id. Close matches: {close}. "
                         f"Run with --list_tasks to see all ids, then pass one with --task.")
    # minedojo.make pops keys out of its shared task table, so a second make() of the same
    # task in one process fails with KeyError: 'prompt'. Put the entry back afterwards.
    from minedojo.tasks import ALL_TASKS_SPECS
    saved = copy.deepcopy(ALL_TASKS_SPECS.get(minedojo_task))
    try:
        raw = minedojo.make(task_id=minedojo_task, image_size=(256, 256), seed=seed)
    finally:
        if saved is not None:
            ALL_TASKS_SPECS[minedojo_task] = saved
    logger.info(f"Created MineDojo env | task={minedojo_task} seed={seed}")
    return MineDojoWrapper(raw)


def _minedojo_task_ids() -> list:
    """All MineDojo task ids, or [] if the installed version does not expose them."""
    try:
        import minedojo.tasks as T
        for name in ("ALL_PROGRAMMATIC_TASK_IDS", "ALL_TASK_IDS"):
            if hasattr(T, name):
                return list(getattr(T, name))
    except Exception:
        pass
    return []


def run(task, model="gemini-3.8-flash", env_type="minedojo", num_episodes=1, max_steps=3000,
        memory=True, seed=42, output_dir="results", verbose=False, start_episode=0):
    """Run episodes of one task and return a summary dict. Memory carries over between episodes.

    start_episode > 0 resumes an interrupted run: it keeps the saved memory, skills and
    finished episodes and runs the remaining ones with the same seeds.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Model names like "ollama/qwen2.5vl:7b" contain characters Windows filenames cannot hold
    tag = re.sub(r"[^A-Za-z0-9._-]", "-", f"{env_type}_{task}_{model}_{'mem' if memory else 'nomem'}")

    config = AgentConfig(
        model=model,
        max_steps=max_steps,
        memory_enabled=memory,
        verbose=verbose,
        memory_path=str(out / f"{tag}_memory.json") if memory else None,
        skills_path=str(out / f"{tag}_skills.json"),
    )
    summary_path = out / f"{tag}_summary.json"
    results = []
    if start_episode:
        if summary_path.exists():
            results = [r for r in json.loads(summary_path.read_text())["episodes"]
                       if r["episode"] < start_episode]
        if len(results) != start_episode:
            raise ValueError(f"Cannot resume at episode {start_episode}: {summary_path} "
                             f"has {len(results)} finished episodes")
    else:
        # Start each run with empty memory/skills so runs are comparable
        for p in (config.memory_path, config.skills_path):
            if p and Path(p).exists():
                Path(p).unlink()

    summary = _summarize(results, env_type, task, model, memory) if results else None
    for ep in range(start_episode, num_episodes):
        env = create_env(task, seed=seed + ep, env_type=env_type, max_steps=max_steps)
        try:
            agent = EmbodiedMindAgent(env, config)
            start = time.time()
            result = agent.run_episode(task=task, seed=seed + ep)
            result["episode"] = ep
            result["elapsed_seconds"] = round(time.time() - start, 2)
            results.append({k: v for k, v in result.items() if k != "log"})
            logger.info(f"Episode {ep + 1}/{num_episodes} | success={result['success']} "
                        f"steps={result['steps']} | {result['summary']}")
        finally:
            env.close()
        # Rewrite the summary after every episode so a crash in a long run keeps what finished
        summary = _summarize(results, env_type, task, model, memory)
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _summarize(results, env_type, task, model, memory):
    n = len(results)
    return {
        "env": env_type,
        "task": task,
        "model": model,
        "memory_enabled": memory,
        "num_episodes": n,
        "success_rate": sum(r["success"] for r in results) / n,
        "avg_steps": sum(r["steps"] for r in results) / n,
        "avg_steps_success": (sum(r["steps"] for r in results if r["success"]) /
                              max(1, sum(r["success"] for r in results))),
        "episodes": results,
    }


def main():
    parser = argparse.ArgumentParser(description="EmbodiedMind: run a single task")
    parser.add_argument("--task", help=f"One of {sorted(TASK_MAP)} or a MineDojo task id")
    parser.add_argument("--list_tasks", action="store_true", help="Print MineDojo task ids and exit")
    parser.add_argument("--env", choices=["minedojo", "mock"], default="minedojo")
    parser.add_argument("--model", default="gemini-3.8-flash",
                        help="gemini-*, gpt-*, ollama/llava, or 'rule' for offline runs")
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--no_memory", action="store_true", help="Disable episodic memory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--start_episode", type=int, default=0,
                        help="Resume an interrupted run at this episode, keeping its memory")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.list_tasks:
        ids = _minedojo_task_ids()
        print("\n".join(ids) if ids else "MineDojo is not installed or does not list its task ids.")
        return
    if not args.task:
        parser.error("--task is required")

    s = run(args.task, args.model, args.env, args.num_episodes, args.max_steps,
            not args.no_memory, args.seed, args.output_dir, args.verbose, args.start_episode)
    print(f"\n{args.task} | env={s['env']} model={s['model']} memory={s['memory_enabled']}")
    print(f"  success rate {s['success_rate']:.0%} over {s['num_episodes']} episodes, "
          f"avg steps {s['avg_steps']:.0f}")
    print(f"Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
