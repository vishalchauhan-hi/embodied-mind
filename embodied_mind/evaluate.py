"""
EmbodiedMind benchmark: runs a task suite with and without episodic memory and
writes results/benchmark.csv and results/benchmark.md.

Usage:
    python -m embodied_mind.evaluate --task_suite harvest --model gemini-3.8-flash --num_episodes 5
    python -m embodied_mind.evaluate --env mock --model rule --task_suite all   # offline pipeline check
"""

import argparse
import csv
from pathlib import Path

from .run import run

SUITES = {
    "harvest": ["harvest_wood", "harvest_wool", "harvest_cobblestone"],
    "techtree": ["craft_wooden_pickaxe", "craft_stone_pickaxe"],
}
SUITES["all"] = SUITES["harvest"] + SUITES["techtree"]


def main():
    ap = argparse.ArgumentParser(description="EmbodiedMind benchmark")
    ap.add_argument("--task_suite", choices=sorted(SUITES), default="harvest")
    ap.add_argument("--env", choices=["minedojo", "mock"], default="minedojo")
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--num_episodes", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="results")
    args = ap.parse_args()

    rows = []
    for task in SUITES[args.task_suite]:
        for memory in (False, True):
            s = run(task, args.model, args.env, args.num_episodes, args.max_steps,
                    memory, args.seed, args.output_dir)
            rows.append({
                "env": s["env"], "task": task, "model": s["model"], "memory": memory,
                "episodes": s["num_episodes"], "success_rate": round(s["success_rate"], 3),
                "avg_steps": round(s["avg_steps"], 1),
                "avg_steps_when_successful": round(s["avg_steps_success"], 1),
            })

    out = Path(args.output_dir)
    with open(out / "benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lines = [
        f"Environment: {args.env}. Model: {args.model}. Episodes per cell: {args.num_episodes}. "
        f"Max steps: {args.max_steps}. Seeds start at {args.seed}.",
        "",
        "| Task | Memory | Success rate | Avg steps | Avg steps (successful) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['task']} | {'on' if r['memory'] else 'off'} | {r['success_rate']:.0%} | "
                     f"{r['avg_steps']:.0f} | {r['avg_steps_when_successful']:.0f} |")
    if args.env == "mock":
        lines += ["", "These numbers come from the offline mock environment. They check that the "
                      "pipeline runs end to end and are not MineDojo results."]
    (out / "benchmark.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
