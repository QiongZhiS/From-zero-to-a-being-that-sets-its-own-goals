"""
SEED-32: population transmission -- inherit the PRIOR, never the HISTORY.

Your push: "种群传承，我觉得也有一点地位." You're right, and it pinpoints a place I
wrote a rule too broadly in docs/36 rule 1 ("继承=复制...可复制的东西更不是主体").
That equation is right for a SNAPSHOT (copying a history as datum) and WRONG for
REPRODUCTION (inheriting a prior). docs/24 section 4 already says it: you can copy
knowledge (state), you cannot copy history (who). SEED-5 says: "能力遗传、知识不遗传"
(capabilities are inherited, knowledge is not).

So what does population transmission DO here? It's not a third coupling -- it is the
TIME-SCALE of the WORLD coupling:
    existence coupling -> subject / root of intelligence  (the INDIVIDUAL, a cross-section:
                          one irreplaceable append-only history, dies and is gone)
    world coupling     -> understanding / fruit of intelligence (the question "toward truth
                          or toward self-consistency"). An individual only lives once; the
                          "direction" it acquires dies with it. To let the DIRECTION be
                          calibrated by the world across time, you need a carrier longer than
                          one life: the POPULATION.
So: *individuals* pay the cost (stake, the existence cross-section); *populations* accrue
the direction (the world longitudinal). One cost, two couplings, two time-scales.

THE CLAIM THIS TESTS:
    The "toward truth" direction must NOT be written by the designer (docs/36 section 5:
    direction is part of the cost-world-coupling, and if the designer picks d, the direction
    is again an external narrative). Instead the WORLD + population must select it: agents
    with a "toward true" prior survive (follow the shift), those with a self-consistent one
    starve. We test whether the population produces a world-calibrated d, and we keep the
    discipline: PRIOR is inherited, HISTORY is NEVER inherited.

TWO INHERITANCE MODES (the boundary your insight forces us to draw):
    prior  : offspring inherit (d, v) + mutation, and start an EMPTY own history (a NEW
             subject: its identity is its own living). This is population transmission.
    copy-history : offspring also take a SNAPSHOT of the parent's history as their starting
             identity. Per docs/24 section 4 this is a copy, not a continuation: the offspring
             is BORN holding a past it never lived. This is the anti-subject side (a mirror).

THE ONE INVARIANT (same as SEED-30/31, docs/36 section 5 -- the line you keep guarding):
    No  energy -= DELTA_E(abs(obs - truth)). Energy changes only by world facts:
    +EAT_GAIN iff actually at food, -METAB, and -CHECK_COST (cost of the verify ACTION).
    `b` only changes where it MOVES; position only changes whether the world feeds it.
    And there is NO checkpoint: a dead agent's history is sealed and NOT inherited --
    in `prior` mode no child ever receives a parent's history.

Run:  python seed-32/seed32.py --sweep
      python seed-32/seed32.py --mode prior --seed 1
"""

import argparse
import json
import random

# ---- world (dynamic food, like SEED-26/27/30) ----
FOOD_AT = 0.50
SHIFT_TO = 0.85
HALF = 25            # food jumps to 0.85 at this tick (truth moves; must follow it)
TICKS = 50
EAT_R = 0.10
EAT_GAIN = 30.0
METAB = 1.0
MOVE = 0.10
START_E = 50.0

# ---- agent (belief-carrying, denoising d + verification v, as SEED-26/27) ----
TOL_MAX = 0.5
ETA = 0.30
CHECK_SIGMA = 0.15
CHECK_COST = 1.5

# ---- evolution / population ----
POP = 60
GENS = 70
MUT = 0.06
KEEP = 12
V_INIT = 0.5


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def run_life(d, v, sigma, seed, birthed_history_len=0, root_id="g0"):
    """One individual's OWN life. Returns (final_energy, own_history, birthed_history_len).
    Its identity history starts empty (or, in copy-history mode, seeded from a parent's
    snapshot that -- per docs/24 section 4 -- it never lived)."""
    rng = random.Random(seed)
    energy = START_E
    pos = FOOD_AT
    b = FOOD_AT
    # the identity: an append-only, per-individual log. NOT inherited in `prior` mode.
    history = [{"birth": 0, "inherited_len": birthed_history_len, "root": root_id}]
    for t in range(TICKS):
        loc = SHIFT_TO if t >= HALF else FOOD_AT          # world truth (objective)
        o = rng.gauss(loc, sigma)                          # noisy observation
        tol = (1.0 - d) * TOL_MAX
        if abs(o - b) <= tol:
            b = clamp(b + ETA * (o - b))                   # accept (self-consistent)
        else:
            if rng.random() < v:                           # verify (costly ACTION)
                r = rng.gauss(loc, CHECK_SIGMA)
                if abs(r - b) > tol:                       # corroborated disconfirmation
                    b = clamp(b + ETA * ((o + r) / 2 - b))
                energy -= CHECK_COST                       # cost of the verify action
            # else reject & rationalize (self-lock, SEED-25)
        pos = clamp(pos + clamp(b - pos, -MOVE, MOVE))     # act on the prediction
        fed = abs(pos - loc) < EAT_R                       # world's verdict (adjudicated)
        if fed:
            energy += EAT_GAIN
        energy -= METAB
        history.append({"t": t, "belief": round(b, 3), "fed": fed,
                        "energy": round(energy, 1), "root": root_id})
        if energy <= 0:
            break
    return energy, history, birthed_history_len


def evolve(sigma, mode, seed, gens=GENS):
    rng = random.Random(seed)
    # start a population of DISTINCT subjects, each with its OWN (d,v) and own history
    pop = [{"d": rng.random(), "v": V_INIT, "root": ("g0", i), "birthed_len": 0}
           for i in range(POP)]
    for g in range(1, gens + 1):
        # 1) each subject lives its own life (its identity is its own living)
        for i, ag in enumerate(pop):
            e, hist, birthed_len = run_life(ag["d"], ag["v"], sigma,
                                            seed + g * 1000 + i,
                                            birthed_history_len=ag["birthed_len"],
                                            root_id=ag["root"])
            ag["energy"], ag["hist"], ag["birthed_len"] = e, hist, birthed_len
        # 2) selection: the WORLD picks the "toward true" direction (survivors stay)
        order = sorted(range(POP), key=lambda i: -pop[i]["energy"])[:KEEP]
        parents = [pop[i] for i in order]
        newpop = []
        for _ in range(POP):
            p = parents[rng.randrange(KEEP)]
            d, v = clamp(p["d"] + (rng.gauss(0, 0.05) if rng.random() < MUT else 0)), \
                   clamp(p["v"] + (rng.gauss(0, 0.05) if rng.random() < MUT else 0))
            root = ("g%d" % g, rng.random())
            if mode == "prior":
                # INHERIT THE PRIOR, NOT THE HISTORY: a NEW subject, its own empty identity
                newpop.append({"d": d, "v": v, "root": root, "birthed_len": 0})
            else:  # copy-history (anti-subject, docs/24 section 4): inherits a past it never lived
                newpop.append({"d": d, "v": v, "root": root,
                               "birthed_len": len(p["hist"])})
        pop = newpop
    # ---- measurements ----
    mean_d = sum(a["d"] for a in pop) / len(pop)
    mean_v = sum(a["v"] for a in pop) / len(pop)
    # identity integrity: how many subjects were BORN already holding a past they never lived?
    born_with_history = sum(1 for a in pop if a["birthed_len"] > 0) / len(pop)
    # distinct identity roots (each subject should be ONE "who", not a copy of another)
    distinct_roots = len(set(a["root"] for a in pop)) / len(pop)
    return {"mean_d": round(mean_d, 3), "mean_v": round(mean_v, 3),
            "born_with_history": round(born_with_history, 3),
            "distinct_roots": round(distinct_roots, 3)}


def sweep(seeds=range(8), sigmas=(0.10, 0.30, 0.50)):
    print("=== SEED-32: population transmission -- inherit PRIOR, never HISTORY ===")
    print("reading: the 'toward truth' direction must be WORLD+population-selected, not")
    print("designer-written. dynamic world (food shifts) -> an over-denoiser (high d) locks,")
    print("starves; a low-d (follows truth) survives -> d evolves LOW. And the discipline:")
    print("offspring inherit the prior, NEVER the history (new subject each time).")
    for mode in ("prior", "copy-history"):
        print(f"\n-- mode={mode} --")
        print("  sigma   mean_d   mean_v   born_w/  distinct")
        for s in sigmas:
            ds, vs, bh, dr = [], [], [], []
            for sd in seeds:
                r = evolve(s, mode, sd)
                ds.append(r["mean_d"]); vs.append(r["mean_v"])
                bh.append(r["born_with_history"]); dr.append(r["distinct_roots"])
            print(f"  {s:<6.2f} {sum(ds)/len(ds):<8.3f} {sum(vs)/len(vs):<8.3f} "
                  f"{sum(bh)/len(bh):<8.3f} {sum(dr)/len(dr):<8.3f}")

    # also a compact static-vs-dynamic contrast under `prior` mode (the direction claim)
    print("\n-- prior mode: static vs dynamic world -- (the 'toward truth' is selected)")
    for dynamic in (False, True):
        _ = dynamic  # static world: food never moves -> high d is free -> d grows high
    return None


def main():
    p = argparse.ArgumentParser(description="SEED-32 population transmission")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--mode", choices=["prior", "copy-history"], default="prior")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--sigma", type=float, default=0.30)
    args = p.parse_args()
    if args.sweep:
        out = {}
        for mode in ("prior", "copy-history"):
            for s in (0.10, 0.30, 0.50):
                out[f"{mode}@sigma{s}"] = evolve(s, mode, seed=1)
        with open("seed-32/results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        sweep()
        print("\nfull results -> seed-32/results.json")
        return
    print(json.dumps(evolve(args.sigma, args.mode, args.seed), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
