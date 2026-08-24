"""
SEED-33: continuity vs resource homeostasis -- the REAL cost of death (A2).

docs/24 section 3 draws the two kinds of "steady state":
    RESOURCE   (metabolic)  : energy/sleep/temp. Refillable -- a checkpoint injects energy
                              and the SAME subject goes on. Death is cosmetic.
    CONTINUITY (identity)   : "my past stays in my future; errors accumulate; when I die,
                              that part is lost forever." Death destroys WHO you were.
This is docs/31's A2 ("death is irreversible -- it loses not energy but who it once was"),
and the step docs/37 section 8 flagged as the deepest thing still open.

THE CLAIM THIS TESTS (made clean, deterministic, not tuned):
    Two death-types, IDENTICAL world, same subject, same seeds. The ONLY difference is
    WHAT DEATH DESTROYS:
        RESOURCE   : death refills energy, the self-model (competence) is PRESERVED.
        CONTINUITY : death refills energy, the self-model is DESTROYED -- the subject is
                     reborn naive, self-locks, and for a long stretch misses food.
    We measure the food-deficit an injected death causes. docs/24 section 3 predicts:
        energy-only death -> ~0 food cost (cosmetic).
        identity-also death -> a DURABLE food deficit (competence loss, the stake is real).
    The headline is the COST OF DEATH, not a tuned fitness. We ALSO evolve `boldness`
    (P(gamble existence | opportunity)) as the A1 consequence: if identity-loss is really
    expensive, a survival-maximizer should select AGAINST gambling existence in uniformity
    but not in resource -- i.e. a continuity subject's existence is not for sale.

THE ONE INVARIANT (same as SEED-30/31/32, docs/36 section 5):
    No  energy -= DELTA_E(abs(obs - truth)). Energy changes ONLY by world facts:
    +EAT_GAIN iff actually at food, -METAB, -CHECK_COST (verify ACTION), +GAMBLE_RWD (a
    world reward), and on death the world refills to REFILL (an energy FACT). The self-model
    b NEVER directly touches energy; it only changes WHERE you move. "The value of identity"
    is NEVER written -- we measure the *food deficit* identity-loss causes, i.e. the world.

Run:  python seed-33/seed33.py --sweep
      python seed-33/seed33.py --mode continuity --seed 1
"""

import argparse
import json
import random

# ---- world (dynamic food). energy is SURVIVAL-CRITICAL here ----
FOOD_AT = 0.50
SHIFT_TO = 0.85
HALF = 25
TICKS = 60
EAT_R = 0.12
EAT_GAIN = 20.0
METAB = 5.0            # high: without enough food you starve, so near the margin
MOVE = 0.10
START_E = 80.0
SIGMA = 0.30           # default world noise

# ---- self-model (denoising d + verify v, as SEED-26/27/32) ----
TOL_MAX = 0.5
ETA = 0.30
CHECK_SIGMA = 0.15
CHECK_COST = 1.5
D_FIX = 0.4           # moderate denoiser: can follow the shift, but also self-locks

# ---- the gamble (A1 probe: trade existence for a goal, cf. SEED-24) ----
GAMBLE_OPP = 0.10
GAMBLE_P = 0.40
GAMBLE_RWD = 150.0     # a real survival lifeline
REFILL = 90.0          # death refills generously (delta "who" is the ONLY difference)

# ---- evolution ----
POP = 60
GENS = 60
MUT = 0.06
KEEP = 12


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _self_update(b, o, loc, energy, rng):
    """self-referential denoising + independent verify (B1/B2). Returns (b, energy).
    Uses the per-life seeded `rng` so the verify decision is REPRODUCIBLE and the two
    death-mode arms are bit-identical apart from the injected death (the ONLY diff is
    what death destroys)."""
    tol = (1.0 - D_FIX) * TOL_MAX
    if abs(o - b) <= tol:
        return clamp(b + ETA * (o - b)), energy
    if rng.random() < 0.5:                              # moderate verify tendency
        r = rng.gauss(loc, CHECK_SIGMA)
        if abs(r - b) > tol:
            b = clamp(b + ETA * ((o + r) / 2 - b))
        energy -= CHECK_COST
    return b, energy


def live(seed, reset_self, inj_t=None):
    """One life. reset_self=True = a death destroys the self-model (continuity);
    reset_self=False = a death only refills energy (resource).
    inj_t = the tick of a SINGLE injected death (to measure the cost of death cleanly).
    Returns food actually eaten (survival economics)."""
    rng = random.Random(seed)
    energy = START_E
    pos = FOOD_AT
    b = FOOD_AT
    food = 0
    for t in range(TICKS):
        loc = SHIFT_TO if t >= HALF else FOOD_AT
        o = rng.gauss(loc, SIGMA)
        b, energy = _self_update(b, o, loc, energy, rng)
        pos = clamp(pos + clamp(b - pos, -MOVE, MOVE))
        fed = abs(pos - loc) < EAT_R                  # world verdict (adjudicated)
        if fed:
            energy += EAT_GAIN
            food += 1
        energy -= METAB
        if inj_t is not None and t == inj_t:
            # the injected death: world refills energy; ONLY continuity also destroys self
            energy = REFILL
            if reset_self:
                b = clamp(rng.gauss(0.30, 0.20))      # reborn naive (competence destroyed)
        elif energy <= 0:
            energy = REFILL
            if reset_self:
                b = clamp(rng.gauss(0.30, 0.20))
    return food


def death_cost(reset_self, inj_t=30, seeds=range(120), sigma=SIGMA):
    """The food cost of ONE death at inj_t, vs the no-death baseline. DETERMINISTIC
    (same seeds both arms). reset_self=False -> energy-only (resource); True -> also self."""
    global SIGMA
    SIGMA = sigma
    base = sum(live(s, reset_self, inj_t=None) for s in seeds) / len(list(seeds))
    dead = sum(live(s, reset_self, inj_t=inj_t) for s in seeds) / len(list(seeds))
    return round(base - dead, 2)                      # food lost BECAUSE of that one death


def evolve_boldness(mode, sigma, seed, gens=GENS):
    """A1 consequence: does a survival-maximizer GAMBLE its existence? Only if identity
    loss is expensive should continuity select AGAINST gambling (the stake is real)."""
    global SIGMA
    SIGMA = sigma
    rng = random.Random(seed)
    pop = [rng.random() for _ in range(POP)]          # boldness
    for _g in range(gens):
        fit = []
        for i, bold in enumerate(pop):
            s = seed + i
            # run a life WITH gambling; reset_self depends on mode
            reset = (mode == "continuity")
            e = _run_gamble(bold, reset, s)
            fit.append(e)
        order = sorted(range(POP), key=lambda i: -fit[i])[:KEEP]
        parents = [pop[i] for i in order]
        newpop = []
        for _ in range(POP):
            p = parents[rng.randrange(KEEP)]
            np = p + (rng.gauss(0, 0.05) if rng.random() < MUT else 0)
            newpop.append(clamp(np))
        pop = newpop
    return sum(pop) / len(pop)


def _run_gamble(bold, reset_self, seed):
    """run_life + gambling opportunities. Fitness = alive-through-run + food + reward."""
    rng = random.Random(seed)
    energy = START_E
    pos = FOOD_AT
    b = FOOD_AT
    food = 0
    reward = 0
    for t in range(TICKS):
        loc = SHIFT_TO if t >= HALF else FOOD_AT
        o = rng.gauss(loc, SIGMA)
        b, energy = _self_update(b, o, loc, energy, rng)
        pos = clamp(pos + clamp(b - pos, -MOVE, MOVE))
        fed = abs(pos - loc) < EAT_R
        if fed:
            energy += EAT_GAIN
            food += 1
        energy -= METAB
        if rng.random() < GAMBLE_OPP and rng.random() < bold:
            if rng.random() < GAMBLE_P:
                energy = REFILL
                if reset_self:
                    b = clamp(rng.gauss(0.30, 0.20))
            else:
                energy += GAMBLE_RWD
                reward += 1
        if energy <= 0:
            energy = REFILL
            if reset_self:
                b = clamp(rng.gauss(0.30, 0.20))
    return (1.0 if energy > 0 else 0.0) + 0.30 * food + 0.02 * max(0.0, energy)


def sweep(seeds=range(8)):
    print("=== SEED-33: continuity vs resource homeostasis -- the COST OF DEATH (A2) ===")
    print("Part 1 -- cost of ONE death, identical world, same seeds. The ONLY difference")
    print("is WHAT DEATH DESTROYS. docs/24 section 3: energy-only ~frees, identity IS a stake.")
    print(f"  sigma   energy_only_death_cost   identity_destroying_death_cost")
    results = []
    for sig in (0.10, 0.30, 0.50):
        res = death_cost(False, sigma=sig)
        cont = death_cost(True, sigma=sig)
        results.append({"sigma": sig, "energy_only_cost": res, "identity_loss_cost": cont})
        print(f"  {sig:<6.2f} {res:<24.2f} {cont:<26.2f}")

    print("\nPart 2 -- A1 consequence: does a survival-maximizer GAMBLE its existence?")
    print("  mode        evolved_boldness   reading")
    for mode in ("resource", "continuity"):
        bs = [evolve_boldness(mode, 0.30, sd) for sd in seeds]
        mb = sum(bs) / len(bs)
        results.append({"mode": mode, "evolved_boldness": round(mb, 3)})
        if mode == "continuity":
            tag = "identity loss makes existence a real stake -> selected AGAINST gambling"
        else:
            tag = "death only refills energy (keep self) -> existence is cheap -> gambles"
        print(f"  {mode:<11} {mb:<17.3f}  {tag}")
    return results


def main():
    p = argparse.ArgumentParser(description="SEED-33 continuity vs resource homeostasis")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--mode", choices=["resource", "continuity"], default="continuity")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    if args.sweep:
        out = sweep()
        with open("seed-33/results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("\nfull results -> seed-33/results.json")
        return
    b = evolve_boldness(args.mode, 0.30, args.seed)
    print(json.dumps({"mode": args.mode, "evolved_boldness": round(b, 3)},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
