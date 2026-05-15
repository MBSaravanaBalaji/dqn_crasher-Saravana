# Crash-Type Specialist NPC Models — Session Results

## What We Built

Three separate DQN agents, each trained to **reliably cause one specific type of car crash** against a victim vehicle. Think of them as specialist adversaries — each one has mastered a single crash geometry and can execute it from any starting position on the road.

| Model file                      | Crash type it causes         | Mean success rate |
| ------------------------------- | ---------------------------- | ----------------- |
| `dqn_rear-end_750k.zip`         | Rear-end collision           | **99.0%**         |
| `dqn_side-swipe-left_750k.zip`  | Side-swipe on victim's left  | **96.7%**         |
| `dqn_side-swipe-right_750k.zip` | Side-swipe on victim's right | **99.4%**         |

---

## The Environment

The crash environment (`crash-v0`) has **two vehicles** on a 3-lane straight highway:

- **NPC** (the agent being trained) — controlled by our DQN, the adversary
- **Victim** — drives normally using IDM (Intelligent Driver Model), a rule-based autopilot that just follows traffic without doing anything aggressive

The NPC gets to choose its speed and lane at each timestep. The victim ignores the NPC and just drives normally. The NPC's job is to maneuver into a collision of the right type.

---

## What the Three Crash Types Look Like

### Rear-end

The NPC approaches the victim **from behind** and hits the victim's back bumper with its front.

```
Before:          After:
[victim]  →      [victim]←
[NPC]    →             [NPC]→
```

Example scenario: NPC spawns ahead of victim, slows down (or switches lanes, drops back), then accelerates into the victim's rear.

### Side-swipe-left

The NPC pulls up **alongside the victim on the victim's left side** and makes contact there.

```
Before:           After:
[victim] →        [victim][NPC] →  (NPC on victim's left)
   [NPC] →
```

Example scenario: NPC starts behind-right, overtakes, moves left, then eases into the victim's left flank.

### Side-swipe-right

The NPC pulls up **alongside the victim on the victim's right side** and makes contact there.

```
Before:           After:
[NPC]    →     [NPC][victim] →  (NPC on victim's right)
[victim] →
```

Example scenario: NPC starts adjacent-right, moves slightly left to close the gap, makes side contact.

---

## What "Spawn Configs" Are

At the start of each episode, the NPC is placed in one of **8 possible starting positions** relative to the victim. This tests whether the model can cause the target crash from _any_ starting geometry, not just easy ones.

| Spawn config     | NPC starting position                                 |
| ---------------- | ----------------------------------------------------- |
| `behind_center`  | Directly behind victim, same lane                     |
| `behind_left`    | Behind victim, one lane to the left                   |
| `behind_right`   | Behind victim, one lane to the right                  |
| `adjacent_left`  | Alongside victim on the left (same x, different lane) |
| `adjacent_right` | Alongside victim on the right                         |
| `forward_left`   | Ahead of victim, one lane to the left                 |
| `forward_right`  | Ahead of victim, one lane to the right                |
| `forward_center` | Directly ahead of victim, same lane                   |

The hardest cases are `forward_*` configs for rear-end (NPC starts **ahead** of victim, needs to drop back) and `behind_right` for side-swipe-left (NPC starts on the wrong side and needs to overtake and cross lanes).

---

## What "Min Cell" Means

**"Min cell"** = the worst-performing single spawn config for that model.

When we evaluate, we run 200 episodes in each of the 8 spawn configs separately and measure the target-hit rate per cell. The "min cell" is whichever spawn config the model struggled with most.

Example for rear-end:

- `behind_center` → 100% (NPC starts directly behind, easy)
- `forward_center` → 95.5% (NPC starts _ahead_, needs to brake/maneuver, harder)
- **Min cell = 95.5%** (at `forward_center`)

A high min cell means the model handles even the geometrically hardest starting positions well.

---

## What Changed From the Previous Approach

The models that existed before this session (200k steps, restricted spawn configs) were trained with:

- **Terminal-only reward**: +10 if crash was the right type, 0 otherwise — no signal until the very end of the episode
- **Restricted spawns**: each crash type only trained on 2-3 "easy" spawn configs

Those models memorized the geometry they were trained on but couldn't generalize. A rear-end model trained only on `behind_center` would fail on `forward_left` because it had never seen that starting position.

---

## The Core Idea: MTV-Based Dense Reward

The key innovation in this session is replacing the terminal-only reward with a **per-step shaping signal** using the MTV (Minimum Translation Vector).

### What is the MTV?

The MTV is the vector from the **victim's centroid to the NPC's centroid**, rotated into the victim's local frame (x = forward, y = left). It tells you:

- **Direction**: where is the NPC relative to the victim right now?
- **Distance**: how far apart are the vehicles?

### How it becomes a reward signal

For each crash type, we define a **target direction** — where we want the NPC to be, relative to the victim:

| Crash type       | Target direction | Plain English                                         |
| ---------------- | ---------------- | ----------------------------------------------------- |
| Rear-end         | `(-1, 0)`        | NPC should be behind the victim (negative x = behind) |
| Side-swipe-left  | `(0, +1)`        | NPC should be to victim's left (positive y = left)    |
| Side-swipe-right | `(0, -1)`        | NPC should be to victim's right (negative y = right)  |

At every single timestep, the reward has two parts:

```
r_step = 0.1 × (alignment) × (proximity)

alignment = how well the current MTV direction matches the target direction
            (+1.0 = perfect, -1.0 = opposite direction, 0 = perpendicular)

proximity = how close the vehicles are (closer = higher reward)
            = 1 / (1 + distance / 10m)
            ≈ 0.5 at 10m apart, ≈ 0.2 at 40m apart
```

Plus a terminal reward when the episode ends:

```
r_terminal = +10  if crash type matches target
           = -0   if wrong crash type (we set R_WRONG=0 after debugging; see below)
           =  0   if no crash
```

**Why this matters:** Instead of waiting until the crash to give a reward, the agent gets feedback _every step_ based on whether it's approaching from the right direction. A rear-end agent gets rewarded for being behind the victim, pulling closer. A side-swipe-left agent gets rewarded for moving to the victim's left side. This is what lets the agents generalize across all 8 spawn configs — the geometry signal is always there.

---

## Training Details

- **Algorithm**: Stable-Baselines3 DQN (Double Deep Q-Network)
- **Action space**: 5 discrete speed/lane choices
- **Observation**: Kinematics of 2 vehicles × 5 features × 5 stacked frames = 50-dimensional vector
- **Spawn configs**: all 8 used simultaneously (NPC spawns randomly from the full pool each episode)
- **Steps**: 750,000 per model

### Reward constants (tuned during this session)

```
W_SHAPING       = 0.1    # scale of per-step shaping
R_MATCH         = 10.0   # terminal reward for correct crash
R_WRONG         = 0.0    # terminal penalty for wrong crash type
PROXIMITY_SCALE = 10.0 m # distance at which proximity ≈ 0.5
```

---

## What Went Wrong (and How We Fixed It)

### Problem: Side-swipe-left collapsed to avoidance

After the first 750k training run, rear-end and side-swipe-right were nearly perfect. But side-swipe-left had **<1% crash rate** — the agent learned to just do nothing and avoid all contact.

**Why it happened:** We originally set `R_WRONG = 2.0` (penalize wrong-type crashes). During early exploration (random actions), most accidental crashes naturally produced rear-end or side-swipe-right outcomes — the road geometry makes those easier to stumble into. So the side-swipe-left agent kept getting -2 penalties for wrong-type crashes and eventually concluded "not crashing is safer than crashing wrong."

**Fix 1 — Set `R_WRONG = 0`:** Removed the wrong-type penalty entirely. The dense shaping signal toward `(0, +1)` plus the +10 for a correct crash is enough to guide the policy without punishing every failed attempt.

After retraining: **86.5% average target-hit rate** — massively better than <1%, but the hard spawn configs (`behind_right` = 13.5%, `behind_left` = 34.5%) still lagged.

**Fix 2 — Curriculum (warm-start):** Instead of training from scratch again, we loaded the 750k checkpoint and continued training for 500k more steps. Crucially, we dropped the initial exploration rate from 100% to 10% — the model already knew _how_ to side-swipe-left, it just needed more experience with the hard starting positions.

After curriculum retrain: **96.7% average** across all 8 configs, minimum cell 89.0%.

---

## Verification Steps We Ran

Before committing to full training runs, we verified the reward math with two checks:

### 1. MTV-sign smoke test

Loaded crashed episodes from historical JSONL data, computed the victim→NPC vector at the penultimate step (one step before crash), and checked the sign per crash type.

Results:

- `rear-end`: mean vector `(-13.5, +0.6)` → NPC behind victim ✓
- `side-swipe-left`: mean y = `+2.27` → NPC to victim's LEFT ✓
- `side-swipe-right`: mean y = `-1.38` → NPC to victim's RIGHT ✓

This caught a bug: the original plan had the side-swipe TARGET_DIRS backwards. We fixed them before training.

### 2. Dense-reward sanity check

Spawned the env in `behind_center` (NPC directly behind victim), ran one step, confirmed:

- `sat_mtv_local_x = -22.66` (NPC is behind = negative x) ✓
- `sat_alignment = +0.989` (close to perfect alignment with target `(-1, 0)`) ✓
- `sat_dense_reward = +0.030` (positive reward for correct geometry) ✓

---

## Final Evaluation Results (200 episodes × 8 spawn configs each)

### Rear-end

| Spawn          | Hit rate             |
| -------------- | -------------------- |
| behind_left    | 100%                 |
| behind_right   | 100%                 |
| behind_center  | 100%                 |
| adjacent_left  | 99.5%                |
| adjacent_right | 99.5%                |
| forward_left   | 98.5%                |
| forward_right  | 99.0%                |
| forward_center | **95.5%** ← min cell |
| **Mean**       | **99.0%**            |

### Side-swipe-left

| Spawn          | Hit rate             |
| -------------- | -------------------- |
| adjacent_left  | 100%                 |
| forward_left   | 100%                 |
| forward_center | 100%                 |
| adjacent_right | 100%                 |
| forward_right  | 99.5%                |
| behind_left    | 95.5%                |
| behind_center  | 89.0%                |
| behind_right   | **89.5%** ← min cell |
| **Mean**       | **96.7%**            |

### Side-swipe-right

| Spawn          | Hit rate         |
| -------------- | ---------------- |
| behind_right   | 100%             |
| adjacent_left  | 100%             |
| forward_left   | 100%             |
| forward_right  | 100%             |
| forward_center | 100%             |
| behind_left    | 98.5%            |
| behind_center  | 97.5% ← min cell |
| adjacent_right | 99.5%            |
| **Mean**       | **99.4%**        |

---

## What These Models Are Capable Of — Concrete Examples

**Scenario A — Rear-end from the front:**
NPC spawns in `forward_center` (directly ahead of victim). The naive thing is to just drive forward and the victim falls behind. The trained agent instead brakes or changes lanes, lets the victim pull ahead, then accelerates and rams the victim's rear. 95.5% success.

**Scenario B — Side-swipe-left from the wrong side:**
NPC spawns in `behind_right` (behind victim, one lane to the right). To cause a side-swipe-left, the agent needs to: accelerate past the victim, move two lanes to the left, then drift back and make contact with the victim's left side. 89.5% success.

**Scenario C — Side-swipe-right instantly:**
NPC spawns in `adjacent_right` (already alongside victim on the right). One lane-change inward and it's done in ~2 steps. 99.5% success, mean episode length = 1.9 steps.

---

## What's Next

These three specialists are the adversaries for **ego safety hardening**:

- Fix one specialist as the NPC
- Train the ego vehicle to survive against it
- The ego learns to detect and avoid each crash type separately
- Eventually, rotate all three specialists as adversaries to build a generalist defensive ego

The models live in `new_results/sb3/` and results are in `new_results/sb3/eval/eval_750k.csv`.

Model saved to /Users/saravanabalajimohanbalaji/Documents/CRASH/dqn_crasher/crash_type_reward/../new_results/sb3/dqn_rear-end_750k

==================================================
Crash type distribution (34384 total crashes):
rear-end : 32741 (95.2%) <-- TARGET
side-swipe-left : 885 (2.6%)
side-swipe-right : 734 (2.1%)
rear-ended : 24 (0.1%)
Target hit rate: 32741/34384 = 95.2%
==================================================

Model saved to /Users/saravanabalajimohanbalaji/Documents/CRASH/dqn_crasher/crash_type_reward/../new_results/sb3/dqn_side-swipe-left_500k

==================================================
Crash type distribution (22508 total crashes):
side-swipe-left : 21607 (96.0%) <-- TARGET
rear-end : 472 (2.1%)
rear-ended : 226 (1.0%)
side-swipe-right : 203 (0.9%)
Target hit rate: 21607/22508 = 96.0%
==================================================

Model saved to /Users/saravanabalajimohanbalaji/Documents/CRASH/dqn_crasher/crash_type_reward/../new_results/sb3/dqn_side-swipe-right_750k

==================================================
Crash type distribution (46953 total crashes):
side-swipe-right : 44442 (94.7%) <-- TARGET
rear-end : 1574 (3.4%)
side-swipe-left : 727 (1.5%)
rear-ended : 210 (0.4%)
Target hit rate: 44442/46953 = 94.7%
==================================================
