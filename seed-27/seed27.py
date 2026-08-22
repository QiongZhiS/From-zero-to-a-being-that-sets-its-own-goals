"""
SEED-27: verification bridge vs self-review -- does INDEPENDENT verification
break the over-denosing self-consistency lock? (docs/25 section 5.3, P19)

SEED-25 + docs/25 section 5.3 claim: a self-model filter closes into a stubborn
self-referential illusion; the "anti-rigidity anchor" is a verification bridge
(SEED-6 do->observe->update, P19) -- BUT only if it is INDEPENDENT of the
self-model. If the "verification" is just self-review (re-affirming the belief),
it keeps filtering by the model and cannot break self-deception.

SEED-26 found over-denoisers (high d) lock onto a stale belief when the world
shifts (they filter the true signal as noise). So the question here:
  does a verification bridge let a high-d agent follow the shift (break the lock),
  and does it have to be INDEPENDENT, or does self-review also work?

VERIFY_MODE (the knob, compared in a DYNAMIC world -- food jumps to SHIFT_TO at HALF):
  none        : no verification -> disconfirming obs are rejected (rationalized).
  independent : on a disconfirming obs, get a 2nd independent read r; accept the
                shift iff BOTH o and r disconfirm the belief (corroboration).
                -> a real shift shows up in both reads -> followed.
  self-review : on a disconfirming obs, get r but ONLY accept if r CONFIRMS the
                belief (re-affirm). Disconfirming r is dismissed as noise.
                -> never breaks the lock (just re-affirms the model).

Part A (mechanism): a fixed high-d over-denoiser in a DYNAMIC world, verify v=1.0,
  VERIFY_MODE in {none, independent, self-review} -> survival & final position.
  Expect: none = lock (stays at old loc, dies); independent = follows shift
  (survives); self-review = locks (dies). -> verification must be INDEPENDENT.

Part B (evolution): evolve d and v (both heritable) in a DYNAMIC world, same three
  VERIFY_MODE -> evolved d. Expect: none = low d (must be adaptive); independent =
  higher d (verification lets it stay clean AND adaptive); self-review = low d
  (verification doesn't help adaptivity).

Run:  python seed-27/seed27.py --sweep
      python seed-27/seed27.py --mode independent --seed 1
"""

import argparse
import json
import random

# world
EAT_R = 0.10
EAT_GAIN = 30.0
METAB = 1.0
TICKS = 50
HALF = 25
SHIFT_TO = 0.85
FOOD_AT = 0.50

# agent
TOL_MAX = 0.5
ETA = 0.30
MOVE = 0.10
START_E = 50.0
CHECK_SIGMA = 0.15
CHECK_COST = 1.5

# evolution (part B)
POP = 60
GENS = 70
MUT = 0.06
KEEP = 12
V_INIT = 0.5


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def fitness(d, v, sigma, verify_mode, seed, ticks=TICKS):
    rng = random.Random(seed)
    energy = START_E
    pos = FOOD_AT
    b = FOOD_AT
    verified = 0
    for t in range(ticks):
        loc = SHIFT_TO if t >= HALF else FOOD_AT    # world is objective, food always shifts
        o = rng.gauss(loc, sigma)
        tol = (1 - d) * TOL_MAX
        if abs(o - b) <= tol:
            b = clamp(b + ETA * (o - b))          # accept (confirms)
        else:
            # disconfirming candidate
            if verify_mode != "none" and rng.random() < v:
                r = rng.gauss(loc, CHECK_SIGMA)   # 2nd independent read
                verified += 1
                if verify_mode == "independent":
                    # corroborate: accept shift iff r ALSO disconfirms b
                    if abs(r - b) > tol:
                        b = clamp(b + ETA * ((o + r) / 2 - b))
                else:  # self-review: only re-affirm if r CONFIRMS b
                    if abs(r - b) <= tol:
                        b = clamp(b + ETA * (r - b))
                energy -= CHECK_COST
            # else: reject (rationalize) -- lock-prone
        pos = clamp(pos + clamp(b - pos, -MOVE, MOVE))
        if abs(pos - loc) < EAT_R:
            energy += EAT_GAIN
        energy -= METAB
        if energy <= 0:
            return 0.0, pos, verified
    return energy, pos, verified


def partA(sigma=0.30, d=0.90, v=1.0, seeds=range(200)):
    print("=== SEED-27 Part A: high-d over-denoiser in DYNAMIC world, does verification break the lock? ===")
    print(f"(d={d}, v={v}, sigma={sigma}; fin_pos ~0.85 = followed the shift, ~0.5 = locked)")
    print(f"{'verify_mode':<12} {'meanEn':<8} {'fin_pos':<8} {'lock%':<7}")
    out = []
    n = len(list(seeds))
    for mode in ("none", "independent", "self-review"):
        en = 0.0; pos = 0.0; locks = 0
        for s in seeds:
            e, fp, _ = fitness(d, v, sigma, mode, s)
            en += e; pos += fp
            if fp < 0.70:     # didn't reach the shifted food (locked)
                locks += 1
        r = {"mode": mode, "mean_energy": round(en / n, 1),
             "mean_fin_pos": round(pos / n, 3), "lock_rate": round(locks / n, 3)}
        out.append(r)
        print(f"{mode:<12} {r['mean_energy']:<8.1f} {r['mean_fin_pos']:<8.3f} {r['lock_rate']:<7.3f}")
    return out


def evolve(sigma, verify_mode, seed, gens=GENS):
    rng = random.Random(seed)
    pop = [(rng.random(), V_INIT) for _ in range(POP)]      # (d, v)
    for _g in range(gens):
        fit = [fitness(d, v, sigma, verify_mode, seed + i)[0] for i, (d, v) in enumerate(pop)]
        order = sorted(range(POP), key=lambda i: -fit[i])[:KEEP]
        parents = [pop[i] for i in order]
        newpop = []
        for _ in range(POP):
            d, v = parents[rng.randrange(KEEP)]
            dd, vv = d, v
            if rng.random() < MUT:
                dd = clamp(dd + rng.gauss(0, 0.05))
            if rng.random() < MUT:
                vv = clamp(vv + rng.gauss(0, 0.05))
            newpop.append((dd, vv))
        pop = newpop
    mean_d = sum(p[0] for p in pop) / len(pop)
    mean_v = sum(p[1] for p in pop) / len(pop)
    return mean_d, mean_v


def partB(sigma=0.30, seeds=range(5)):
    print("\n=== SEED-27 Part B: evolve (d=denoise, v=verify) in DYNAMIC world ===")
    print(f"(sigma={sigma}; none=no channel, independent=catch real shift, self-review=censored)")
    print(f"{'verify_mode':<12} {'evolved_d':<10} {'evolved_v':<10}")
    out = []
    for mode in ("none", "independent", "self-review"):
        ds, vs = [], []
        for sd in seeds:
            d, v = evolve(sigma, mode, sd)
            ds.append(d); vs.append(v)
        row = {"mode": mode, "evolved_d": round(sum(ds) / len(ds), 3),
               "evolved_v": round(sum(vs) / len(vs), 3)}
        out.append(row)
        print(f"{mode:<12} {row['evolved_d']:<10.3f} {row['evolved_v']:<10.3f}")
    return out


def main():
    p = argparse.ArgumentParser(description="SEED-27 verification bridge vs self-review")
    p.add_argument("--mode", choices=["none", "independent", "self-review"], default="independent")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--seeds", type=int, default=5)
    args = p.parse_args()
    if args.sweep:
        a = partA(seeds=range(1, 201))
        b = partB(sigma=0.30, seeds=range(1, args.seeds + 1))
        with open("seed-27/results.json", "w", encoding="utf-8") as f:
            json.dump({"partA": a, "partB": b}, f, ensure_ascii=False, indent=1)
        print("\nfull results -> seed-27/results.json")
        return
    e, ver = fitness(0.9, 1.0, 0.30, args.mode, 1)
    print(json.dumps({"survival": e > 0, "energy": round(e, 1)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
