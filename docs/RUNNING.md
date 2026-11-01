# Running EmbodiedMind on MineDojo

MineDojo runs a real Minecraft instance, so it needs Java 8, Python 3.9 and a few GB of RAM. A GPU is not needed when the VLM and LLM are API models. Both Linux and Apple Silicon Macs are covered below.

## 1. Install on Linux

Use an Ubuntu machine (20.04 or 22.04) with at least 8 GB of RAM.

```bash
sudo apt update
sudo apt install -y openjdk-8-jdk xvfb
sudo update-alternatives --config java      # pick Java 8

conda create -n emind python=3.9 -y         # MineDojo needs Python 3.9
conda activate emind
pip install pip==24.0                        # gym 0.21 has metadata newer pip rejects
pip install setuptools==65.5.0 wheel==0.38.4
pip install minedojo "numpy<2"

git clone https://github.com/vishalchauhan-hi/embodied-mind.git
cd embodied-mind
pip install -e ".[minedojo,gemini,dev]"
pytest -q
```

Then check that Minecraft builds and starts (section 3).

## 2. Install on Apple Silicon (M1 and later)

MineDojo's Minecraft needs an x86_64 Java 8, so the whole stack runs as Intel code under Rosetta. Tested on macOS 27 with an M-series MacBook Pro.

```bash
softwareupdate --install-rosetta --agree-to-license   # needs sudo
brew install --cask temurin@8                         # Temurin 8, macOS x64 build
brew install --cask miniforge

CONDA_SUBDIR=osx-64 conda create -n emind python=3.9 -y
conda activate emind
conda config --env --set subdir osx-64               # keep this env Intel-only
conda env config vars set JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home
conda deactivate && conda activate emind
python -c "import platform; print(platform.machine())"   # must print x86_64

pip install pip==24.0
pip install setuptools==65.5.0 wheel==0.38.4
pip install minedojo "numpy<2"

git clone https://github.com/vishalchauhan-hi/embodied-mind.git
cd embodied-mind
pip install "cryptography==48.0.0"    # newer releases have no Intel-Mac wheel and cannot be compiled for x86_64 here
pip install -e ".[minedojo,gemini,dev]"
pytest -q
```

## 3. Check that Minecraft builds

The pip release of MineDojo does not ship `minedojo.scripts.validate_install`. This runs the same check:

```bash
python - <<'EOF'
import minedojo
env = minedojo.make(task_id="combat_spider_plains_leather_armors_diamond_sword_shield",
                    image_size=(288, 512), world_seed=123, seed=42)
env.reset()
for _ in range(20):
    env.step(env.action_space.no_op())
env.close()
print("[INFO] Installation Success")
EOF
```

The first run builds Minecraft with Gradle and takes 10 to 20 minutes.

**`Could not find com.github.SpongePowered:MixinGradle:dcfaf61`.** jitpack no longer serves this build plugin ([MineDojo#113](https://github.com/MineDojo/MineDojo/issues/113)). Point MineDojo's `minedojo/sim/Malmo/Minecraft/build.gradle` (inside your environment's `site-packages`) at SpongePowered's own repository: add

```groovy
maven {
    name = "sponge"
    url = "https://repo.spongepowered.org/repository/maven-public/"
}
```

to the `buildscript { repositories { ... } }` block, and change the MixinGradle line to

```groovy
classpath('org.spongepowered:mixingradle:0.6-SNAPSHOT'){ // 0.6
```

Reinstalling MineDojo undoes this edit.

## 4. API key

```bash
cp .env.example .env
# put GOOGLE_API_KEY=... or OPENAI_API_KEY=... in .env
set -a; source .env; set +a
```

## 5. Check task ids

MineDojo task ids differ between versions. List them and make sure the ones in `TASK_MAP` in `embodied_mind/run.py` exist:

```bash
python -m embodied_mind.run --list_tasks | grep -E "harvest|techtree|survival"
```

If an id is missing, the runner stops and suggests close matches. You can also pass any MineDojo id directly with `--task`.

## 6. Run

On a headless Linux machine, use a virtual display:

```bash
export MINEDOJO_HEADLESS=1
```

One short episode:

```bash
python -m embodied_mind.run --task harvest_wood --model gemini-3.8-flash --num_episodes 1 --max_steps 500 --verbose
```

When one short episode works, run the benchmark:

```bash
python -m embodied_mind.evaluate --task_suite harvest --model gemini-3.8-flash --num_episodes 5 --max_steps 3000
```

Results go to `results/benchmark.md` and `results/benchmark.csv`, plus one JSON summary per task and setting.

## Cost and time

With the default settings the agent calls the VLM about every 5 steps and the planner every 10 steps, so a 3,000-step episode makes roughly 600 VLM calls and 300 planner calls, and a 500-step episode about 420 calls in all. Start with `--max_steps 500` and a few episodes to see the cost before running the full benchmark. Raise `perception_interval` and `replan_interval` in `AgentConfig` to make runs cheaper.

The Gemini free tier allows only 20 requests per model per day, which is not enough for one 500-step episode; enable billing on the key first. When an API call fails, the agent falls back to its rule planner and keeps going, so check the log for `429` or `404` before trusting a result.
