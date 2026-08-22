"""
SEED-26: evolve DENOISING ability (docs/26 section 5 falsifiable proposition).

docs/26 proposes denoising ability as a SECOND AXIS, dual to signal quality
(P20/docs/12): signal quality = the WORLD's ceiling, denoising = the SUBJECT's
skill. Proposition: ① denoising ability is monotonically shaped by the world's
signal-to-noise (P11: environment shapes behavior); ② over-denosing has a
SELF-REFLEXIVE cost -- filtering too hard by your own belief rejects the true
signal when reality diverges from your belief, so you LOCK onto a stale model.

World: food at loc on [0,1]. Each tick the agent observes o ~ N(food_loc, sigma_w)
(sigma_w = world noise = 1/SNR knob). Its belief b tracks food; it moves toward b
and eats if within EAT_R of food_loc.

The heritable param is d = DENOISING STRENGTH (how tightly the agent filters
observations by its own belief -- a confirmation gate):
    tol(d) = (1 - d) * TOL_MAX      # higher d = tighter gate = stronger denoise
    accept o into b iff |o - b| <= tol(d)
    b += ETA * (o - b)  when accepted; else b stays (the "noise" is rejected)

So d=1 rejects almost everything (over-denoiser: clean but rigid -> locks),
d=0 accepts everything (tracks raw noise -> no denoise but adapts).

Environments (swept): STATIC food loc=0.5, vs DYNAMIC (food jumps to 0.85 at
HALF_T) x sigma_w in {low=0.10, mid=0.30, high=0.50}.
Predictions:
  ① SNR shapes d: in high-noise static world, strong denoising is essential to
     survive (low-d follows noise away from food and dies) -> d evolves HIGH;
     in low-noise world, denoising is unneeded -> d drifts LOW/neutral.
  ② over-denosing locks: in DYNAMIC world, high-d rejects the shifted food
     signal (far from stale belief) -> stays put -> dies; low-d tracks the shift
     -> survives -> d evolves LOW.

Run:  python seed-26/seed26.py --sweep
      python seed-26/seed26.py --sigma 0.4 --dynamic 1
"""

import argparse
import json
import random

# world
EAT_R = 0.10
EAT_GAIN = 30.0
METAB = 1.0
TICKS = 50
HALF = 25              # dynamic world: food jumps at this tick
SHIFT_TO = 0.85
FOOD_AT = 0.50

# agent
TOL_MAX = 0.5
ETA = 0.30
MOVE = 0.10
START_E = 50.0

# evolution
POP = 60
GENS = 70
MUT = 0.06
KEEP = 12


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def fitness(d, sigma_w, dynamic, seed, ticks=TICKS):
    rng = random.Random(seed)
    energy = START_E
    pos = FOOD_AT
    belief = FOOD_AT
    for t in range(ticks):
        loc = SHIFT_TO if (dynamic and t >= HALF) else FOOD_AT
        o = rng.gauss(loc, sigma_w)
        tol = (1 - d) * TOL_MAX
        if abs(o - belief) <= tol:
            belief = clamp(belief + ETA * (o - belief))
        pos = clamp(pos + clamp(belief - pos, -MOVE, MOVE))
        if abs(pos - loc) < EAT_R:
            energy += EAT_GAIN
        energy -= METAB
        if energy <= 0:
            return 0.0
    return energy


def evolve(sigma_w, dynamic, seed, gens=GENS):
    rng = random.Random(seed)
    pop = [rng.random() for _ in range(POP)]        # random initial d
    history = []
    for _g in range(gens):
        fit = [fitness(d, sigma_w, dynamic, seed + i) for i, d in enumerate(pop)]
        # select top KEEP, clone + mutate
        order = sorted(range(POP), key=lambda i: -fit[i])[:KEEP]
        parents = [pop[i] for i in order]
        newpop = []
        for _ in range(POP):
            p = parents[rng.randrange(KEEP)]
            if rng.random() < MUT:
                p = clamp(p + rng.gauss(0, 0.05))
            newpop.append(p)
        pop = newpop
        history.append(sum(pop) / len(pop))
    return sum(pop) / len(pop), history


def sweep(seeds=range(5), sigmas=(0.10, 0.30, 0.50), dynamics=(False, True)):
    print("=== SEED-26: evolve denoising ability (mean evolved d per world) ===")
    print("(d=denoising strength; higher d = tighter belief-gate = stronger denoise)")
    print(f"{'dynamic':<9} {'sigma':<7} {'evolved_d':<10} {'interpretation'}")
    results = []
    for dynamic in dynamics:
        for s in sigmas:
            ds = [evolve(s, dynamic, sd)[0] for sd in seeds]
            mean_d = sum(ds) / len(ds)
            tag = "DYNAMIC" if dynamic else "static "
            results.append({"dynamic": bool(dynamic), "sigma": s, "evolved_d": round(mean_d, 3)})
            print(f"{tag:<9} {s:<7.3f} {mean_d:<10.3f}")
    return results


def main():
    p = argparse.ArgumentParser(description="SEED-26 evolve denoising ability")
    p.add_argument("--sigma", type=float, default=0.4)
    p.add_argument("--dynamic", type=int, choices=[0, 1], default=0)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--seeds", type=int, default=5)
    args = p.parse_args()
    if args.sweep:
        out = sweep(seeds=range(1, args.seeds + 1))
        with open("seed-26/results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("\nfull results -> seed-26/results.json")
        return
    d, hist = evolve(args.sigma, bool(args.dynamic), seed=1)
    print(json.dumps({"sigma": args.sigma, "dynamic": bool(args.dynamic),
                      "evolved_d": round(d, 3), "traj": [round(x, 3) for x in hist]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
