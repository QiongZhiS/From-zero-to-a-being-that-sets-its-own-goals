# Synthetic Life

> Creating from scratch a being that sets its own goals. Not simulated life — synthetic life.
>
> Project created: 2026-08 | Status: **three phases complete + B0 scale test + cross-validation loop closed + Route C complete + SEED-21 society pollution-lock mechanism version** (21 experiments + 20 documents)
> Phase 3 archived: **continuity (endogenous continuity)** — docs/14 theory + docs/15 summary (SEED-18/19)
> B0 conclusion: **scale is not a variable** — the mechanism-level findings (division of labor / competitive exclusion) hold across 300~5000 individuals
> Cross-validation (docs/17): the collaborator's distillation experiments empirically confirmed the signal-quality prediction — self-distill -16.5pp vs human-label +2.5pp; the recommendation has been adopted in their repository (signal-quality-constraint.md)
> Route C: standalone articles on signal quality published (34 posts on Zhihu, 2026-08-20); public repository https://github.com/QiongZhiS/From-zero-to-a-being-that-sets-its-own-goals
> Next steps: polish the public repository (English README + experiment-data JSON) | Route B argument | the continuity trilogy (shelved by the user)

## What this is

A long-term project that builds a living subject from nothing. It starts from a plain question: **"Can we, like a creator, build with our own hands a being that approaches life?"**

The answer: yes — but the path is not "designing intelligence." It is **giving it the foundations for survival, and letting life grow out on its own.**

## Core philosophy (written before any code)

1. **Endogenous homeostasis outranks everything.** Agency, cost structure, and exploratory drive need no separate design — as long as the homeostatic variables genuinely deplete and cannot be bypassed, "survival" emerges from existence itself.
2. **No external reward.** The system does not know that "being alive" is a goal; it is simply forced by physics to stay alive. The cost is endogenous (energy runs out), not bolted on (a reward function).
3. **Keep room to maneuver.** Life is first born in a virtual world. Fully local, pausable, replayable, rollback-able. Let it live first; discuss consequences later.
4. **Conceptual unity.** Perception / action / homeostasis / subject are four top-level words. Vision and pain, movement and thought are just different implementation difficulties of the same word, not different essences.
5. **Synthesis, not simulation.** The subject's death is real (energy depletion is irreversible). We do not perform life; we let life emerge on its own within the rules.

## Project structure

```
synthetic-life/
├── README.md              # this file
├── PLAN.md                # roadmap (v0.x → SEED-1 → ...)
├── docs/                  # written record of discussions (idea threads → conceptual framework → positions → experiments)
│   ├── 01-缘起-从算法到合成生命.md
│   ├── 02-概念框架-感知动作稳态主体.md
│   ├── 03-AGI-差最后一块.md
│   ├── 04-主体性与判据外置的张力.md
│   ├── 05-SEED-0-第一代实验.md
│   ├── 06-外部参考.md
│   ├── 07-七代表格.md          # now covers 12 generations
│   ├── 08-倒果为因.md
│   ├── 09-业内方案对照.md
│   └── 10-结尾与机器路线.md     # project summary + new-direction manifesto
└── seed-0..12/            # twelve generations of experiments, each reproducible
```

## Current status (measured at v0.1.1)

| Strategy | Mean lifespan | Conclusion |
|----------|---------------|------------|
| L0 random | 174 ticks | world baseline |
| L1 heuristic (with prior) | 3000 (immortal) | the power of priors |
| L2 Q-learning (no prior) | 177 ticks | only matched random; never learned food-seeking |

Three empirical findings:

1. **The prior gap is real** — L1 and L2 differ only in the single inductive bias of "what food is"
2. **The extreme of pure homeostasis minimization is inertia** — the first thing learned is "die as cheaply as possible"
3. **Interoception must come before bolted-on reward** — the death-penalty signal is too sparse to be useful; only the energy delta carries gradient

> Honest conclusion: with only the single physical rule "energy runs out" plus pure homeostatic learning, survival cannot be learned.
> Priors are a prerequisite for survival learning — the evolutionary priors of the human brain are not a free lunch.

## SEED-1 status (reproduction + evolution — complete)

**The environment shapes behavior — three numbers prove it** (the only heritable parameter is hunger, the urgency to forage):

| Environment | Evolved hunger | Population |
|-------------|----------------|------------|
| Abundant food (128) | 0.188 (unhurried, roaming) | hits the 400 cap |
| Scarce food (64) | 0.353 (moderately urgent) | hits the 400 cap |
| Depleted food (32) | 0.854 (foraging almost constantly) | stable ~250 (resource bottleneck) |

- Natural selection is at work: no hand-written food-seeking rule; hunger is monotonically shaped by the environment
- An r-selection pattern emerges under scarcity: gen 6→13, births 3883 / deaths 3636
- **The threshold of life has been crossed**: self-maintenance (SEED-0) + self-replication (SEED-1)

## SEED-2 status (predator-prey coevolution — complete)

**Predation pressure shapes behavior — another monotonic demonstration** (the prey's only parameter is still hunger; it does not know predators exist):

| Predation pressure | Prey hunger | Predator aggression |
|--------------------|-------------|---------------------|
| No predators | 0.38 | — |
| 2 predators | 0.63 | 0.37 |
| 5 predators | 0.82 | 0.57 |

- The ecosystem coexists stably (high prey turnover: births 28364 / eaten 22752 / starved 5326)
- **Key physical finding**: predators with full-map perception inevitably cause over-predation collapse (extinction in all 4 rounds of parameter sweeps); adding a vision limit (PRED_VISION=5) yields immediate coexistence — **perception limits are the physical condition for ecological stability**
- Threat pressure breaks the "inertia extreme" (the design constraint from docs/05, finding 2, is now implemented)

## SEED-3 status (cognition: local vision + memory — complete)

Prey can only perceive food within 4 cells, but can remember food locations (memory has capacity limits, decays over time, and needs verification).
Heritable parameters: hunger + memory_weight (trust in memory).

| Memory mode | Evolved trust | Goal-directed navigations | Starvation deaths |
|-------------|---------------|---------------------------|-------------------|
| No memory | 0.07 | 8,619 | 501 |
| Evolvable memory | **0.68** | 44,616 | **383** ← optimal |
| Full-trust memory | 0.94 | 56,471 | 421 |

- **Goal-directedness emerges**: "remembered, therefore go" (5× more navigations than reactive behavior)
- **Evolution finds the optimal trust level 0.68** — neither full trust (stale memories mislead) nor full doubt (memory wasted)
- **Memory carries a real cost**: full-trust memory starves more (stale memories send agents on wasted trips) — cognition is not a free lunch

## SEED-4 status (curiosity: the emergence of exploration — complete)

Healthy individuals use the heritable parameter curiosity to choose between "explore unvisited regions" and "wander randomly".

| World structure | Evolved curiosity | Starvation deaths in dynamic world (evolve vs no-curiosity) |
|-----------------|-------------------|-------------------------------------------------------------|
| Uniform static | 0.31 | — |
| Clustered static | 0.38 | 816 vs 649 (benefit unclear) |
| Clustered dynamic (rich patches move) | **0.51** | **812 vs 934 (curiosity -13%)** |

- **Curiosity emerges under survival pressure**: nonzero in every world — exploration (spatial coverage efficiency) has value in itself, even when "staying alive" is the only goal
- **Environmental dynamics monotonically shape curiosity** (0.31 → 0.38 → 0.51): when rich patches move, exploration becomes a hard requirement
- **An empirical answer to the inertia problem**: inertia is not the inevitable outcome of homeostasis minimization — when the world has structure and change, evolution selects for curiosity

## SEED-5 status (intuition: within-individual integration of experience — complete)

Each subject accumulates a "richness map" over its lifetime (which regions hold more food; eating adds score, score decays over time).
When hungry and no food is visible, heading straight for the richest region is intuition (the influence is present, its origin cannot be pointed to).
**The map (experience) is not inherited; only the learning rate (integration speed) is inherited** — abilities are inherited, knowledge is not.

| Learning-rate mode | Evolved value | Intuitive moves | Starvation deaths |
|--------------------|---------------|-----------------|-------------------|
| Fixed 0 | 0.13 (drifting) | 16,699 | 1004 (worst) |
| Evolvable | **0.69** | 2,312 | **832** (optimal) |
| Fixed high 0.8 | 0.81 | 911 | 967 |

- **The forgetting rate is the key to the optimum**: in a dynamic world, old intuitions go stale — forgetting too slowly (low lr) turns intuition into a liability that steers individuals toward depleted regions (most starvation); forgetting too fast (high lr) means intuition barely exists; 0.69 is the balance point
- **Verifiability wins**: in a dynamic world, explicit memory (verifiable, deletable) sees 25k uses ≫ intuition's 2k — intuition cannot be verified or corrected, so in a rapidly changing environment it loses to memory
- Echoes the essay "The End Point of Memory Is Intuition": **the rate at which "intuition must be re-nurtured" is the learning_rate — evolution found the balance**

## SEED-6 status (active verification: the minimal do-operator unit — complete)

Memory is a hypothesis ("there may be food there"), and its confidence decays over time (the world changes). To keep a memory usable,
the subject must actively re-verify — pay a cost to confirm, observe the outcome, update the belief (do → observe → update).
Heritable parameter: verification_bias (willingness to pay the cost of verification).

| Verification tendency | Evolved value | Verifications | Starvation deaths |
|-----------------------|---------------|---------------|-------------------|
| Fixed 0 | 0.08 | 4,038 | 1158 (worst) |
| Evolvable | **0.65** | 42,771 | **515** (optimal) |
| Fixed high 1.0 | 0.96 | 41,166 | 550 |

- **Active verification is hugely valuable**: starvation deaths halve in the dynamic world (1158 → 515)
- **Evolution picks moderate verification (0.65), not always-verify**: verification costs; over-verifying wastes energy
- **The payoff is in the region, not the point** (hit_rate is only 0.002 yet starvation halves): remembered locations lie near clusters where food was seen; verification brings individuals back to food-rich areas — "that shop is closed, but the block still has food"

## SEED-7 status (event segmentation: a sense of time — complete, negative result)

The subject carries an internal predictor (sliding mean of visible food count); prediction error continuously exceeding a threshold = event boundary = memory wipe.

| Segmentation mode | Evolved threshold | Starvation (gradual-change world) | Starvation (abrupt-change world) |
|-------------------|-------------------|-----------------------------------|----------------------------------|
| Never segment | 36 (drifting) | **503** | **851** |
| Evolvable | 2.49 | 635 | 1169 |
| Always segment | 0.10 | 974 | — |

- **Negative result (recorded honestly)**: full-wipe segmentation was never selected in any world — evolution pushed the threshold up both times
- **Why**: brute-force wiping discards valuable information; confidence decay (smooth forgetting) plus per-item verification (targeted updates) is always better
- **Conclusion**: memory decay is gradual, not abrupt. Real event segmentation should "organize events," not "wipe memory" (kept as a design constraint for the next phase; see docs/07)

## SEED-8 status (prediction-driven strategy switching — complete, negative result)

The subject predicts "current region is depleting" (inter-feeding interval stretched beyond its own history × sensitivity) → evacuate early instead of passively waiting.
Heritable: sensitivity. Three worlds: renewable / very slowly renewable / non-renewable.

| World | fixed0 (no prediction) | evolve | Low sensitivity (always flee) |
|-------|------------------------|--------|-------------------------------|
| Renewable | **455** | 590 | 1031 |
| Very slowly renewable | 54 pop / 2 migrations | 72 pop / 5 migrations | — |
| Non-renewable | extinction (same as evolve) | extinction | — |

- **Negative result**: prediction-driven evacuation was never selected in any world (evolution pushed the threshold up both times)
- **Boundary conditions**: in renewable worlds, waiting beats fleeing; in global depletion there is nowhere to flee; the value requires "locally irreversible depletion"
- **The deepest finding**: prediction's value is capped by perception quality — a 4-cell vision cannot tell "not arrived yet" from "region depleted," so the prediction signal is noise. **A sense of time can only emerge with better perception**

## SEED-9 status (courage: action under uncertainty — complete ✅)

SEED-8's lesson is not only about perception but about a "courage" layer: signals are always ambiguous, and whether to act on them is an independent decision.
boldness = probability of acting under an ambiguous signal (heritable). Perception (50-tick hit-rate trend, denoised)
and courage (action tendency) are orthogonal and evolve separately.

| Courage mode | boldness | Migrations | Starvation deaths |
|--------------|----------|------------|-------------------|
| timid | 0.07 | 14,954 | 314 |
| **evolve** | **0.425** | 24,830 | **275** ← optimal |
| bold | 0.95 | 27,644 | 311 |

- **Courage is a real parameter**: evolution finds the 0.425 balance point (signal but no action: 314; wasted on false alarms: 311; moderate: 275)
- **Perception and courage are orthogonal**: SEED-8 lacked the courage layer (deterministic triggering) and failed; once added, evacuation gets selected
- Maps onto the boldness-shyness continuum in animal behavior — empirical evidence that "complex things can be simplified"

## SEED-10 status (open capability space — complete)

Neuroevolution (71 weights as the genome, no preset capability parameters) + post-hoc archaeology (the answer to docs/08's reverse causality).

- **Post-hoc naming is viable**: a "sedentary filter-feeding" behavior emerged (always GATHER, waiting for food to regenerate) — we did not preset it; we read it out of the data. The "foraging pathway" we had preset was erased by evolution
- **Open ≠ rich**: a stress-free open space degenerates to the simplest solution (the inertia system reproduced at evolutionary scale); too much stress causes extinction; rich behavior needs moderate stress (SEED-9's courage world)
- **The causal origin lies with the creator**: the stress parameter is given by the designer — the shape of the capability space is set by "world pressure" (docs/08)

## SEED-11 status (open space + pressure — complete, negative result)

Drifting food regeneration (breaking fixation) is pressed into the neuroevolution world. Result: **extinction** (collapses at every scale and parameter setting).

- **Core finding: the world must contain exploitable structure for intelligence to gain a foothold**
  - Static structure (in-place regeneration) → fixation (SEED-10, the simplest strategy that exploits structure)
  - No structure (random drift) → extinction (SEED-11, no strategy available)
  - Dynamic structure (regional depletion) → rich behavior (SEED-9, but with hand-written parameters)
- **A precise answer to "where does courage come from"**: at toy scale, courage cannot grow from zero —
  the courage we observe is always "our written structural prior being tuned"; true from-scratch emergence needs
  "dynamically predictable structure + a larger stage" (docs/08)

## SEED-12 status (minimal quality diversity — complete, negative result)

Niche-crowding penalty (competition within the same behavior type; the reproduction threshold rises with niche crowding).

- **Failure mechanism**: the crowding penalty freezes reproduction of the dominant niche → the mutation stream stops → diversity locks (all settings end at 98-100% fixation)
- **Core insight: competitive exclusion holds in behavior space** — competition within the same niche inevitably squeezes out diversity; implicit penalties (competition) cannot maintain diversity, they only freeze variation
- **Quality diversity must be explicit** (MAP-Elites-style grid + elite retention + continuous sampling of mutations) — the approach used by POET / Jeff Clune's group in the industry; we independently verified its necessity through three failures (SEED-10/11/12)

## SEED-13 status (cultural evolution: imitation vs genetics — complete, first stop of Phase 2)

Genotype/phenotype separation: gene_hunger is inherited (mutation); pheno_hunger drives behavior (adjusted by imitation, never enters the genome).

| Mechanism | pheno convergence | Starvation deaths | Gene |
|-----------|-------------------|-------------------|------|
| Genetics only | 0.66→0.724 (still climbing at 6000 ticks) | 144 | 0.709 |
| **Genetics + imitation** | **stable at 0.545 by ~600 ticks** | **60 (-58%)** | **0.459 (barely moves)** |

- **Imitation accelerates convergence**: ~600 ticks vs not converged at 6000
- **Imitation yields a real survival advantage** (-58% starvation): it spreads "verified good strategies" (population-level verification), while genetics spreads "random mutations" — **culture bypasses the prior gap** (a docs/08 prediction fulfilled)
- **Genotype/phenotype separation works**: culture never enters the genome (anti-Lamarckian); offspring relearn — "abilities are inherited, knowledge is not" extends from the individual level to the cultural level
- **A concrete link to continual learning**: "the more it spreads, the faster it spreads" (spatial) is empirically confirmed

## SEED-14 status (ratchet effect — complete, conditional result)

Imitation vs genetics on a two-parameter combination (hunger×boldness), with local vision.

| World | Cultural lock-in | Starvation advantage |
|-------|------------------|----------------------|
| Rich (128) | **complete lock-in** (pheno variance collapses to zero [0.570,0.570]) | marginal |
| Harsh (64) | fails (pheno swings [0.27,0.92]) | -4.6% |

- **The ratchet (consensus lock-in) is real**: the whole population copies one combination, variance collapses to zero, no regression
- **But lock-in has preconditions**: the most successful individual must be stable — in high-pressure worlds "the most successful" changes every tick (luck-dominated); the imitation target wavers → consensus swings → lock-in fails
- **Core insight: cultural transmission is limited by the signal-to-noise ratio (SNR) of the success signal** — in single-parameter, high-SNR worlds imitation has a huge advantage (SEED-13: -58%); in two-parameter, low-SNR worlds imitation only transmits noise — **isomorphic to SEED-8's perception ceiling**

## SEED-15 status (culture vs genetics in a dynamic world — complete, counterintuitive result)

Food switches periodically between 32↔128; optimal hunger 0.85↔0.19. Tests "who keeps up after the switch."

| Mechanism | Locked position | Starvation deaths |
|-----------|-----------------|-------------------|
| Genetics | 0.839 (= optimum of the old environment; cannot keep up) | 3,249 |
| Culture | 0.609 (long-run compromise) | **2,840 (-12.6%)** |

- **"Cultural inertia" is falsified; it is actually "genetic inertia"**: imitation spreads "currently most successful" (real-time signal); genetics locks in "the choice from before the switch" (lagged signal) — after an environment switch, genetics suffers while carrying the old optimum
- **The true optimum in a dynamic world = long-run compromise** (not either static optimum), and culture gets closer to it
- **Integration with docs/12**: culture's signal update rate > genetics' — another instance of the signal quality theory

## SEED-16 status (emergence of division of labor — complete, overturning expectations)

Two food regions (A sparse/high-energy, B dense/low-energy); individuals have a preference (0 = specialize in A, 1 = specialize in B).

| Mechanism | Distribution | Starvation deaths |
|-----------|--------------|-------------------|
| uniform (fixed bimodal) | 110A/2mid/288B | 2,387 |
| genetic | 100A/62mid/238B (partial division of labor) | **844** (optimal) |
| cultural | 6A/390mid/3B (**all-middle unimodal**) | **5,605** (worst!) |

- **Division of labor partially emerges under genetics** (bimodal, fewest starvation deaths)
- **Imitation flattens division of labor**: copying "the most successful" → whole-population homogenization — **imitation is de-differentiating by nature**
- **More deeply: imitation spreads a bad consensus** — "most successful" is measured by instantaneous energy (a short-term signal); generalists show high short-term energy and get copied, but are inefficient long-term (most starvation deaths) — **under short-term signals, culture is a liability**
- **Integration with docs/12**: culture's value depends on whether the imitated signal reflects long-term fitness — this explains why real division of labor (ants/humans) is sustained by "identity / inimitable skills," not by imitation

## SEED-21 status (LLM society pollution-lock — mechanism version complete ✅)

**Problem**: docs/18 predicted S2 — "LLM multi-agent societies will develop erroneous consensus lock (NO RECOVERY)."
The mechanism version first validates the mechanism (no API burn): a society of 200 agents copies whoever "looks most successful" (reputation/leaderboard);
strategy A (true mean 10) beats B (8). During the fraud window, all scores of the B camp get +5 (fake leaderboard/hype), exposed after the window.
Verification layer = personally test-drive before adoption (SEED-6's do → observe → update).

| Mechanism | Fraud | A share in window | A at end | Lock-in rate | Recovery rate |
|-----------|:--:|:--:|:--:|:--:|:--:|
| genetic (no imitation) | yes | 0.50 | 0.50 | 0% | 100% |
| cultural-nv (blind follow, exploration 0.02) | no | 0.72 | 0.71 | 0% | 100% |
| cultural-nv (blind follow, exploration 0.02) | yes | **0.38** | 0.72 | 0% | 100% |
| **cultural-nv (blind follow, exploration 0 = pure culture)** | **yes** | **0.33** | **0.00** | **100%** | **0%** |
| cultural-v (verification layer, exploration 0.02) | yes | 0.62 | 0.65 | 0% | 100% |
| **cultural-v (verification layer, exploration 0 = pure culture)** | **yes** | 1.00 | 1.00 | 0% | 100% |

- **P16 confirmed**: a pure-culture society captured by fraud is **NO RECOVERY** — the erroneous consensus sustains itself; even after the fraud is exposed (B returns to its true scores) it does not recover, because no independent exploration offers alternatives; the exploration trickle (0.02) is exactly "the antidote to cultural lock-in" — at the cost of each individual retaining a shred of independent experimentation
- **P19 confirmed**: the verification layer rejects fraud (A's share barely drops during the window), but its protection decays as noise rises (σ=0.5→25: 0.646→0.520), and verification carries a real cost (welfare 7.2 vs 9.1 for blind follow — SEED-6's "verification costs; evolution picks moderate, not always" reproduced)
- **Genetic immunity**: genetic mode does not amplify fraud (no contagion) — the societal version of P16's "genes are naturally immune"
- **Q3 surprise finding**: pure luck (no fraud, σ=25) cannot lock B in; instead it reproduces the SEED-14 mechanism — under high noise, imitation transmits only noise and consensus quality degrades (end_A 0.759→0.542); lock-in requires an **active erroneous signal** (fraud/hype); luck is not enough
- **Implication for docs/18 S2**: the prediction's mechanism holds, but with a boundary added — lock-in in an LLM society requires (a) no verification layer, (b) low independent exploration, (c) an active erroneous signal; only with all three present is it NO RECOVERY
- Next step: the real-LLM-agent version (DeepSeek API, 8-16 agents × 30 rounds; verification layer = the prompt requires test-first-then-adopt)

## SEED-21b status (LLM version: a real DeepSeek agent society — complete ✅)

After the mechanism version ran, we moved to real LLM agents (DeepSeek chat, 10 agents × 30 rounds × 2 views, seed 42).
Each agent picks a method each round from A (true mean 10) / B (true mean 2), watching a "mean score of the last 15 rounds" leaderboard (the window denoises);
the blind-follow group follows the most successful; the verification group test-drives 5 times before adopting; fraud = one implanted fixed B user who claims +5 during the window (fake leaderboard).

| Configuration | A share in fraud window | After exposure | Conclusion |
|---------------|:--:|:--:|------------|
| blind + global leaderboard + fraud | **0.000** | 0.000 (**NO RECOVERY**) | captured and locked in within a single round |
| blind + local gossip + fraud | 0.225 | 0.000 (**NO RECOVERY**) | locked in after 2-3 rounds of spread |
| blind + local + no fraud | 1.000 | 1.000 | control; converges stably on A |
| **verify + global + fraud** | **0.150** | **1.000 (instant recovery)** | still captured 80-90%, but everyone switches back after exposure |
| verify + local + fraud | 0.800 | 1.000 (recovered) | shallow capture, fast recovery |
| verify + global + no fraud | 1.000 | 1.000 | control |

**Key findings from the real LLM agents**:

1. **Social-proof bias outweighs measured data**: the verify group held hard data — "B measured 2 vs my own A 10" — yet 80-90% still chose to follow the 14.5-point fake leaderboard in round one. The LLM weights the social signal of "the most successful" above direct evidence
2. **In the LLM version, the verification layer is a recovery mechanism, not a prevention mechanism** (vs the mechanism version, where verification prevented): once they switch to B they stop (leaderboard #1 == themselves, no more testing), but after exposure testing discipline makes everyone switch back **instantly** — the sharpest contrast to the blind group's NO RECOVERY
3. **Self-referential leaderboard self-reinforcement**: the global leaderboard includes the fraudster itself → it sees itself as #1 → stays on B; the local sample excludes itself → the fraudster rationally defects back to A (inflating scores only advertises A) — so the fraudster must be an "implanted fixed strategy" (SEED-18's intruder design), or the experiment cannot measure pollution
4. **Framing effect**: a ranked leaderboard carries more authority for an LLM than a raw list of samples; when the gap is too small (A10 vs B8), an LLM society locks onto a lucky leader before fraud even begins (a language-level reproduction of Q3's luck lock) — another instance of P3's signal quality: when a single round's score has SNR < 1, the society cannot tell good from bad and can only follow luck
5. **Revision of docs/18 S2**: lock-in in an LLM society requires blind following (no verification discipline) + an active erroneous signal + low independent exploration; a society with verification discipline cannot be locked, but **prevention fails while recovery succeeds** — the prediction is refined from "NO RECOVERY" to "blind followers: NO RECOVERY / verifiers: delayed capture but recoverable"

Code: `seed-21/seed21_llm.py` (the key comes from the environment variable DEEPSEEK_API_KEY and is never written to disk) + `seed-21/debug_llm.py` (diagnostic tool)

## SEED-21c status (scale and lock probability — Route B's first argument ✅)

**Question**: Route B requires demonstrating a scale-sensitive phenomenon (B0 concluded "scale is not a variable"). Three probes:

1. **Proportional pollution (15% invaders) → scale-invariant**: capture depth is constant across N=50/200/1000 (dur_A 0.38-0.39) — mathematically forced: the expected number of "sees the fraud" events per round is N × p_imit × sample_n/N = p_imit×sample_n ≈ 0.25, **independent of N**
2. **Single planted fraud → still scale-invariant**: the same product cancels the visibility dilution (`seed21.py --scale-sweep --invader-n 1`)
3. **Finite-population randomness → genuinely scale-sensitive** (no fraud, single-draw signal σ=8, pure imitation, 300 seeds per point):

| N | wrong-lock probability P(lockB) | correct-lock probability P(lockA) |
|---|:--:|:--:|
| 8 | **0.270** | 0.730 |
| 10 | 0.200 | 0.800 |
| 15 | 0.100 | 0.900 |
| 20 | 0.083 | 0.917 |
| 25 | 0.027 | 0.973 |
| 40 | 0.010 | 0.990 |

**Conclusions (first argument for Route B)**:

- **Scale-sensitive phenomena exist, but their shape is "small-population risk"**: a small society (N≈8-15) is likely to be dominated by a single lucky agent (27%→10%), monotonically decreasing with size — this explains why the LLM version showed a 6-agent luck-lock but stable 10-agent runs
- **Direction is opposite to the docs/08 hypothesis**: it is not "large scale creates new capabilities" but "large scale reduces wrong-lock probability" — B0's "scale is not a variable" is refined to: **scale is not a variable for mechanism type, but IS a variable for consensus correctness**
- **Cultural homogenization is the default** (lockA+lockB=1.0, no intermediate states): imitation societies always converge to a single strategy, right or wrong — the societal version of P14; diversity requires explicit mechanisms (MAP-Elites style) or larger scale
- **Refinement to docs/18 S2**: multi-agent lock risk is highest in small societies — experiments with small populations (8-15 agents) systematically overestimate lock-in risk; scaling dilutes the luck component

## Quick start

```bash
cd seed-0
python seed0.py --all        # audit + baseline + L2 training
python seed0.py --baseline   # run baseline only
python diag.py               # inspect what L2 learned
```

## In one sentence

**We are not writing a program. We are deciding the fate of a species — starting with keeping it from starving.**
