# ESE 6510: Physical Intelligence — Drone Racing Project

A from-scratch PPO drone-racing policy, trained in NVIDIA Isaac Lab / Isaac Sim on a custom
7-gate "Powerloop" track (including a vertical-loop maneuver and a high-speed chicane), then
deployed on a real Crazyflie at PERCH @ Pennovation. Built by Josh Kim and Kevin Song for ESE 6510
at Penn. See [`../CLAUDE.md`](../CLAUDE.md) for full project context, environment details, and
development history, and [`WRITEUP.md`](WRITEUP.md) for the reward-design rationale.

## What's implemented here

- **PPO** (`src/third_parties/rsl_rl_local/rsl_rl/algorithms/ppo.py`): clipped surrogate loss,
  clipped value loss, per-minibatch advantage normalization, adaptive KL-based learning-rate
  schedule.
- **Reward, observations, and reset strategy**
  (`src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py`):
  - Reward = progress-to-gate (Δ-distance, clipped) + sparse gate-pass bonus + crash penalty
    (contact sensor) + command-smoothness cost, following the formulation in
    [arXiv 2406.12505](https://arxiv.org/abs/2406.12505). Gate-pass detection uses a sign-change
    crossing of the gate plane (not a naive distance threshold), which also lets it reject and
    terminate on a backwards traversal.
  - Observations are 20-dim and entirely body-frame: linear/angular velocity, vectors to the
    current and next gate, relative gate orientation, previous action — so the same policy weights
    generalize across every gate on the course.
  - Resets randomize starting gate (uniform over all 7 — full-course curriculum from iteration 0),
    position, heading, and approach speed, plus full domain randomization of thrust-to-weight,
    aerodynamic drag, and PID gains.

## Results

Final training run (2026-04-23, 1000 iterations, `num_envs=8192`, ~72 min on a GCP
`g2-standard-8` + NVIDIA L4): mean episode reward **220**, **~4 laps completed per episode** on
average (28/33 mean gates passed), low steady-state crash rate. See `wandb/run-*/files/output.log`
for full per-iteration logs (local only — not tracked in git, see note below).

Phase 2 (sim2real): a Circle Track policy was flown at Pennovation on the real `crazy_jirl_b3`
Crazyflie — see `plots/rosbag/` for the recorded trajectory (multiple completed loops, gate passes
registered at all 4 waypoints) and `rosbag/` for the raw ROS2 bag.

**Note:** `wandb/`, `logs/`, and `outputs/` are gitignored — this data (15 training runs, all
checkpoints, the real flight recording) exists only in this working copy, recovered from the GCP
training VM. See `../CLAUDE.md` for why that matters.

## Setup

This cannot run on a Mac — Isaac Sim 4.5 requires an NVIDIA GPU + Linux. See `../CLAUDE.md` for
the exact verified environment (Python/torch/CUDA/Isaac Sim versions) and
[`requirements-verified.txt`](requirements-verified.txt) for pinned package versions confirmed to
work together.

* Enter your **home** directory, then clone this repo and Isaac Lab as siblings:

```bash
git clone <this repo>
git clone git@github.com:vineetpasumarti/IsaacLab.git
```

## Training

```bash
python scripts/rsl_rl/train_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs 8192 \
    --max_iterations 1000 \
    --headless \
    --logger wandb
```

## Evaluation

```bash
python scripts/rsl_rl/play_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs 1 \
    --load_run [YYYY-MM-DD_XX-XX-XX] \  # run directory under logs/rsl_rl/quadcopter_direct/
    --checkpoint best_model.pt \
    --headless \
    --video \
    --video_length 800
```

## Directory structure

- `src/isaac_quad_sim2real/tasks/race/config/crazyflie/` — environment, reward/obs/reset strategy,
  PPO hyperparameters (`agents/rsl_rl_ppo_cfg.py`)
- `src/third_parties/rsl_rl_local/` — local PPO implementation
- `ese651_sim2real/` — vendored copy of the Phase 2 ROS2 deployment code (see the standalone
  `ese651_sim2real` repo alongside this one for the canonical version)
- `wandb/`, `logs/`, `outputs/` — local training artifacts (gitignored, see note above)
- `rosbag/`, `plots/rosbag/`, `analyze_bag.py` — real-world Phase 2 flight data and analysis
