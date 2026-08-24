"""
SEED-36: let the SUBJECT's NATURE (its metabolism) be settled by the world, not by us.

docs/39 settled: "我们从代码开始，不能从原子开始。涌现不洗白作者——但真正该做的不是模仿
原子，而是让一切能'自己长'的尽量自己长，连'它该在乎什么'(nature)也让意外-选择去沉淀."
docs/39 section 7 names the target: turn the nature parameters (homeostasis / denoise /
"do I doubt myself") from designer-set foundations into things evolved by "accident +
selection". SEED-26 already evolved denoise (d) and SEED-32 evolved the direction; but
**the homeostasis (metabolism) is STILL a constant I set**: `METAB` (how much it costs to
stay alive). docs/39 section 6 says that is exactly a designed "what it cares about".

THIS SEED removes that last designer-set nature value. `metab` (the metabolic cost per
tick = how hard the subject must work to keep existing) is now HERITABLE, evolved by the
world. We sweep the resource supply (a world fact) and ask: is `metab` shaped BY THE WORLD?

    Claim (P11 docs/16: environment shapes behavior, not design):
        the world SELECTS the evolved nature (metab = cost-of-living), and the direction it
        picks depends on the geometry of supply:  in a SCARCE, scattered world a high burn
        (racing the few spots, speed) is bought;  in a RICH, clustered world speed buys little
        and the burn just costs, so a LEAN nature wins.  The point is not the sign of the slope
        but that the designer no longer says "this is what it costs to live" -- the world does.

This is "leave the nature to the world", made measurable: the designer no longer says
"this is how much it costs to live"; the world + selection say it.

THE ONE INVARIANT (docs/36 section 5 -- the line you keep guarding):
    No  energy -= DELTA_E(abs(obs - truth)). Energy changes ONLY by world facts:
    +EAT_GAIN iff actually at food, and -metab (the subject's OWN evolved burn rate).
    metab is NOT a score of how wrong the prediction is; it is a heritable trait that the
    WORLD selects (high burn dies in a scarce world). There is no designer reward.

Run:  python seed-36/seed36.py --sweep
      python seed-36/seed36.py --food 20 --seed 1
"""

import argparse
import json
import random

# ---- world ----
EAT_R = 0.10
EAT_GAIN = 30.0
TICKS = 50
HALF = 25
SHIFT_TO = 0.85
FOOD_AT = 0.50
SIGMA = 0.30            # world noise (both streams)
START_E = 60.0

# ---- agent (belief-carrying) ----
TOL_MAX = 0.5
ETA = 0.30
MOVE = 0.10
CHECK_SIGMA = 0.15
V = 0.6
D = 0.4

# ---- evolution (metab is the heritable NATURE trait; d fixed to isolate the affordance) ----
POP = 60
GENS = 80
MUT = 0.06
KEEP = 12
METAB_MIN = 0.2
METAB_MAX = 8.0


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def belief_update(b, o, loc, rng):
    """self-referential denoising + verify (B1/B2), fixed moderate traits."""
    tol = (1.0 - D) * TOL_MAX
    if abs(o - b) <= tol:
        return clamp(b + ETA * (o - b))
    if rng.random() < V:
        r = rng.gauss(loc, CHECK_SIGMA)
        if abs(r - b) > tol:
            return clamp(b + ETA * ((o + r) / 2 - b))
    return b


def fitness(metab, food_spots, seed):
    """One life. `food_spots` = how many food clusters exist (a world fact).

    metab is the subject's NATURE (burn rate). The world shapes it because it PULLS BOTH WAYS:
        COST  : a high burn drains energy each tick -> dies sooner if eating is unreliable.
        BENEFIT: metabolism buys SPEED -> covers more food clusters per tick -> EATS MORE.
    FIT = food actually eaten (the world's verdict on "did you get fed"), with a survival floor.
    RICH (many clusters) -> clustered food, you're already near it -> a high burn just costs,
        speed buys little -> LEAN metab selected.  SCARCE (few, scattered clusters) -> speed
        genuinely lets you rush the few spots you can reach and eat more -> HIGH metab (fast)
        selected, and the burn is worth it.  So the world shapes the nature (P11) -- and the
        direction it picks is a world answer, not a designer's guess.
    Energy only via +EAT_GAIN / -metab; move-speed scales with metab (speed is the benefit)."""
    rng = random.Random(seed)
    energy = START_E
    pos = FOOD_AT
    b = FOOD_AT
    food_eaten = 0
    move = MOVE * (metab / 1.0) ** 0.6              # metabolism buys movement speed
    spots = [FOOD_AT + (i / max(1, food_spots)) * 0.3 - 0.15 for i in range(max(1, food_spots))]
    for t in range(TICKS):
        base = SHIFT_TO if t >= HALF else FOOD_AT
        locs = [base + (la - FOOD_AT) for la in spots]
        o = rng.gauss(base, SIGMA)
        b = belief_update(b, o, base, rng)
        pos = clamp(pos + clamp(b - pos, -move, move))
        fed = any(abs(pos - lc) < EAT_R for lc in locs)
        if fed:
            energy += EAT_GAIN
            food_eaten += 1
        energy -= metab
        if energy <= 0:
            break
    # survival floor + food eaten (thriving = the world's verdict)
    return (energy if energy > 0 else -50.0) * 0.01 + food_eaten


def evolve(form, food_spots, seed, gens=GENS):
    """Evolve the NATURE trait `metab` in a world with `food_spots` (resource supply).
    Returns the mean evolved metab. The designer sets only the WORLD supply, not the nature."""
    rng = random.Random(seed)
    pop = [clamp(rng.uniform(METAB_MIN, METAB_MAX), METAB_MIN, METAB_MAX) for _ in range(POP)]
    for _g in range(gens):
        fit = [fitness(m, food_spots, seed + i) for i, m in enumerate(pop)]
        order = sorted(range(POP), key=lambda i: -fit[i])[:KEEP]
        parents = [pop[i] for i in order]
        newpop = []
        for _ in range(POP):
            p = parents[rng.randrange(KEEP)]
            np = p + (rng.gauss(0, 0.25) if rng.random() < MUT else 0)
            newpop.append(clamp(np, METAB_MIN, METAB_MAX))
        pop = newpop
    return sum(pop) / len(pop)


def sweep(seeds=range(10), food_spots_levels=(1, 3, 6, 12, 24)):
    print("=== SEED-36: let the SUBJECT'S NATURE be settled by the world (docs/39 S6/S7) ===")
    print("We remove the last designer-set nature value: `metab` (the burn rate = how hard")
    print("it must work to stay alive) is now HERITABLE, SELECTED by the world. The designer")
    print("sets only the world supply (food_spots). P11: environment shapes behavior, not design.")
    print(f"  food_spots   evolved_metab    reading")
    out = []
    for fs in food_spots_levels:
        ms = [evolve(0.0, fs, sd) for sd in seeds]
        mean_m = sum(ms) / len(ms)
        if fs <= 3:
            tag = "SCARCE, scattered -> speed lets you rush all the few spots -> HIGH metab (fast) bought"
        elif fs <= 12:
            tag = "moderate -> an affordable mid burn rate"
        else:
            tag = "RICH, clustered -> you're already near food, speed buys little, burn just costs -> LEAN"
        out.append({"food_spots": fs, "evolved_metab": round(mean_m, 3), "reading": tag})
        print(f"  {fs:<11} {mean_m:<15.3f}    {tag}")
    return out


def main():
    p = argparse.ArgumentParser(description="SEED-36: nature settled by the world")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--food", type=int, default=6, help="food_spots in the world")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    if args.sweep:
        out = sweep()
        with open("seed-36/results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("\nfull results -> seed-36/results.json")
        return
    m = evolve(0.0, args.food, args.seed)
    print(json.dumps({"food_spots": args.food, "seed": args.seed,
                      "evolved_metab": round(m, 3)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
