"""
SEED-37: the WHOLE nature settles itself -- evolve denoise (d) AND doubt-the-self (v) together.

We made metab (the cost of living) evolve in SEED-36. But two nature parameters are still
WRITTEN BY THE DESIGNER: `d` (denoising strength = how tightly the subject filters evidence
to fit its own belief) and `v` (verification tendency = "do I doubt myself and check?").
These ARE the subject's nature (docs/39: the "该在乎什么" / how it treats its own model).

SEED-26 already showed d is world-shaped (dynamic world -> low d; static -> high d; and the
over-denoiser self-locks). SEED-27 showed v is world-shaped (independent verification is selected
in a dynamic world). But those were SEPARATE. This seed REMOVES both designer-set values at once
and asks: if the designer sets ONLY the world (supply + how much the world moves + noise), do
d and v co-evolve into a coherent "nature" -- AND does that nature serve the world, not me?

    The designer sets ONLY:
        world_sigma      (noise, P20 signal quality)
        dynamic          (does the meaning move -- a shift at HALF?)
    and NOTHING about how the subject should filter or verify. d and v are heritable.

    Claim (docs/39: "let the nature settle from the world", P11 environment shapes behavior):
        the evolved (d, v) are a coherent answer to the world's structure -- the world shapes
        the WHOLE nature, not one trait:  a dynamic/moving world needs BOTH low d (don't over-
        lock) and high v (do doubt yourself enough to re-verify); a static world can afford high
        d (lock is safe) and low v (no need to doubt). We test the co-evolution, and the story it
        tells about "what it cares about" is the world's, not the designer's.

THE ONE INVARIANT (docs/36 section 5 -- the line you keep guarding):
    No  energy -= DELTA_E(abs(obs - truth)). Energy changes ONLY by world facts:
    +EAT_GAIN iff actually at food, -METAB, -CHECK_COST (verification is an ACTION you pay for).
    d and v never directly touch energy; they only change the belief, which only changes WHERE
    the subject moves, and position only changes whether the world feeds it. d and v are evolved
    by the world (high-v costs energy; over-denoised d misses the shift), not scored by a designer.

Run:  python seed-37/seed37.py --sweep
      python seed-37/seed37.py --dynamic 1 --seed 1
"""

import argparse
import json
import random

# ---- world ----
EAT_R = 0.10
EAT_GAIN = 30.0
METAB = 1.0
MOVE = 0.10
START_E = 50.0
TICKS = 50
HALF = 25
SHIFT_TO = 0.85
FOOD_AT = 0.50

# ---- agent ----
TOL_MAX = 0.5
ETA = 0.30
CHECK_SIGMA = 0.15
CHECK_COST = 1.5

# ---- evolution (d and v BOTH heritable, BOTH world-shaped) ----
POP = 60
GENS = 80
MUT = 0.06
KEEP = 12


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def fitness(d, v, sigma, dynamic, seed):
    """One life. d = denoise strength (tighten the belief gate -> self-lock when world moves);
    v = verify tendency (doubt yourself, pay a cost to check, catch a real shift).
    The WORLD feeds iff actually at food. d and v only shape WHERE it moves."""
    rng = random.Random(seed)
    energy = START_E
    pos = FOOD_AT
    b = FOOD_AT
    for t in range(TICKS):
        loc = SHIFT_TO if (dynamic and t >= HALF) else FOOD_AT
        o = rng.gauss(loc, sigma)
        tol = (1.0 - d) * TOL_MAX
        if abs(o - b) <= tol:
            b = clamp(b + ETA * (o - b))                    # accept (fits my story)
        else:
            if rng.random() < v:                            # doubt myself -> verify (costly ACTION)
                r = rng.gauss(loc, CHECK_SIGMA)
                if abs(r - b) > tol:                        # corroborated disconfirmation
                    b = clamp(b + ETA * ((o + r) / 2 - b))
                energy -= CHECK_COST
            # else: reject & rationalize (self-lock)
        pos = clamp(pos + clamp(b - pos, -MOVE, MOVE))
        fed = abs(pos - loc) < EAT_R
        if fed:
            energy += EAT_GAIN
        energy -= METAB
        if energy <= 0:
            break
    return energy if energy > 0 else 0.0


def evolve(sigma, dynamic, seed, gens=GENS):
    """Evolve (d, v) together. Designer sets ONLY world_sigma + dynamic; NOT the nature."""
    rng = random.Random(seed)
    pop = [(clamp(rng.random()), clamp(rng.random())) for _ in range(POP)]   # (d, v)
    for _g in range(gens):
        fit = [fitness(d, v, sigma, dynamic, seed + i) for i, (d, v) in enumerate(pop)]
        order = sorted(range(POP), key=lambda i: -fit[i])[:KEEP]
        parents = [pop[i] for i in order]
        newpop = []
        for _ in range(POP):
            d, v = parents[rng.randrange(KEEP)]
            dd = clamp(d + (rng.gauss(0, 0.05) if rng.random() < MUT else 0))
            vv = clamp(v + (rng.gauss(0, 0.05) if rng.random() < MUT else 0))
            newpop.append((dd, vv))
        pop = newpop
    return sum(p[0] for p in pop) / len(pop), sum(p[1] for p in pop) / len(pop)


def sweep(seeds=range(8), sigmas=(0.10, 0.30, 0.50)):
    print("=== SEED-37: the WHOLE nature settles itself -- evolve d (denoise) AND v (doubt) ===")
    print("The designer sets ONLY the world (noise + does it move). d and v are BOTH heritable.")
    print("P11 / docs/39: the world shapes the WHOLE nature, not one trait. Predict:")
    print("  dynamic (meaning moves): low d (don't over-lock) + high v (doubt & re-verify)")
    print("  static  (meaning stays) : high d (lock is safe) + low v (no need to doubt)")
    print(f"{'dynamic':<9} {'sigma':<6} {'evolved_d':<10} {'evolved_v':<10} reading")
    out = []
    for dynamic in (False, True):
        for s in sigmas:
            ds, vs = [], []
            for sd in seeds:
                d, v = evolve(s, dynamic, sd)
                ds.append(d); vs.append(v)
            md = sum(ds) / len(ds); mv = sum(vs) / len(vs)
            tag = ("dynamic: LOW d + HIGH v -> don't over-lock, do doubt yourself"
                   if dynamic else "static: HIGH d + LOW v -> lock is safe, no need to doubt")
            out.append({"dynamic": bool(dynamic), "sigma": s,
                        "evolved_d": round(md, 3), "evolved_v": round(mv, 3), "reading": tag})
            print(f"{str(dynamic):<9} {s:<6.3f} {md:<10.3f} {mv:<10.3f}  {tag}")
    return out


def main():
    p = argparse.ArgumentParser(description="SEED-37: the whole nature settles itself")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--dynamic", type=int, choices=[0, 1], default=1)
    p.add_argument("--sigma", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    if args.sweep:
        out = sweep()
        with open("seed-37/results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("\nfull results -> seed-37/results.json")
        return
    d, v = evolve(args.sigma, bool(args.dynamic), args.seed)
    print(json.dumps({"dynamic": bool(args.dynamic), "sigma": args.sigma, "seed": args.seed,
                      "evolved_d": round(d, 3), "evolved_v": round(v, 3)},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
