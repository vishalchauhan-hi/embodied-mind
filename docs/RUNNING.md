# Running EmbodiedMind on MineDojo

MineDojo runs a real Minecraft instance, so it needs Java 8, a few GB of RAM and a Linux machine works best. A GPU is not needed when the VLM and LLM are API models.

## 1. Machine

Use an Ubuntu machine (20.04 or 22.04) with at least 8 GB of RAM. Apple Silicon Macs are hard to set up for MineDojo because it needs an x86 Java 8 build of Minecraft; a Linux lab machine or a cloud VM is easier.

## 2. Install

```bash
sudo apt update
sudo apt install -y openjdk-8-jdk xvfb
sudo update-alternatives --config java      # pick Java 8

conda create -n emind python=3.9 -y         # MineDojo needs Python 3.9
conda activate emind
pip install setuptools==65.5.0 wheel==0.38.4   # older gym 0.21 needs these
pip install minedojo
python -m minedojo.scripts.validate_install    # first run builds Minecraft, takes 10 to 20 minutes

git clone https://github.com/vishalchauhan-hi/embodied-mind.git
cd embodied-mind
pip install -e ".[minedojo,dev]"
pytest -q
```

If `validate_install` fails while building Minecraft, check the MineDojo GitHub issues for your error message before changing anything else. Build problems there are usually about Java version or Gradle downloads.

## 3. API key

```bash
cp .env.example .env
# put GOOGLE_API_KEY=... or OPENAI_API_KEY=... in .env
set -a; source .env; set +a
```

## 4. Check task ids

MineDojo task ids differ between versions. List them and make sure the ones in `TASK_MAP` in `embodied_mind/run.py` exist:

```bash
python -m embodied_mind.run --list_tasks | grep -E "harvest|techtree"
```

If an id is missing, the runner stops and suggests close matches. You can also pass any MineDojo id directly with `--task`.

## 5. Run

Headless, with a virtual display:

```bash
export MINEDOJO_HEADLESS=1
python -m embodied_mind.run --task harvest_wood --model gemini-2.5-flash --num_episodes 1 --max_steps 500 --verbose
```

When one short episode works, run the benchmark:

```bash
python -m embodied_mind.evaluate --task_suite harvest --model gemini-2.5-flash --num_episodes 5 --max_steps 3000
```

Results go to `results/benchmark.md` and `results/benchmark.csv`, plus one JSON summary per task and setting.

## Cost and time

With the default settings the agent calls the VLM about every 5 steps and the planner every 10 steps, so a 3,000-step episode makes roughly 600 VLM calls and 300 planner calls. Start with `--max_steps 500` and a few episodes to see the cost before running the full benchmark. Raise `perception_interval` and `replan_interval` in `AgentConfig` to make runs cheaper.
