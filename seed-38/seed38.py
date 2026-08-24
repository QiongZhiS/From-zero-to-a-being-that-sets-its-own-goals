"""
SEED-38: can the subject's PURPOSE toward another GROW out of the encounter -- not be set by me?

The hardest bone docs/42 left open. Every purpose ever observed was a designer-set G
(SEED-24's treasure). And SEED-24 showed the trap: a terminal goal the subject values more
than living makes it TRADE its continuity (instrumentalization). So "purpose" cannot be a
designer target -- that's exactly the mirror/instrumental thing docs/22 and 36 rejected.

Just as docs/36/37 let the NATURE parameters (metab / denoise / doubt-self) be settled by the
world, SEED-38 asks: can the subject's PURPOSE toward an OTHER be settled by the world too?

THE CLAIM THIS TESTS (made measureable, no designer reward):
    There is a second like-being in the world (docs/36 rule 4: COUPLED existence -- the shared
    cache is abundant ONLY if BOTH are alive, 互咬合). The ONLY designer knob is "is the other
    even there" (with_other). The subject's `give` (how much of its budget it spends on the
    other) is HERITABLE, EVOLVED BY THE WORLD. Predict: with the interlock, keeping the other
    alive is genuinely survival-relevant -- so the world SELECTS a NONZERO give: a real purpose
    toward the other GROWS out of the encounter. In NO-OTHER there is nothing to give to, so
    give is irrelevant and settles ~0. We measure the EVOLVED give and both survival rates.

THE ONE INVARIANT (docs/36 section 5 -- the line you keep guarding):
    No  energy -= DELTA_E(abs(obs - truth)). Energy changes ONLY by world facts:
    +BOTH_YIELD iff BOTH alive / +SOLO_YIELD if the other is gone (a world fact, the interlock),
    -METAB, -give (cost of transferring budget to the other -- an ACTION cost, like SEED-22).
    There is NO designer function "score how much you care about the other". `give` is EVOLVED,
    never scored; the WORLD decides who lives. 'Purpose toward the other' is READ, never rewarded
    by a designer.

Run:  python seed-38/seed38.py --sweep
      python seed-38/seed38.py --with-other 1 --seed 1
"""

import argparse
import json
import random

# ---- world ----
START_A = 40.0
METAB = 6.0            # high: solo (interlock broken) the subject DIES before the run ends
TICKS = 40

# ---- the other (like-being, REAL interlock per docs/36 rule 4) ----
# The shared cache is abundant ONLY if BOTH are alive (互咬合): if the other dies, your food
# yield collapses so hard you STARVE before the run ends. So keeping the other alive is
# genuinely survival-critical, and a purpose toward it can GROW out of the encounter.
B_START = 30.0
B_METAB = 2.0
GIVE_UNIT = 3.0        # per-tick budget a subject can transfer to the other (an ACTION cost)
BOTH_YIELD = 8.0       # if BOTH alive: net +2/tick -> survive IF it lasts the whole run
SOLO_YIELD = 2.0       # if other gone: net -4/tick -> A drains and dies before the end

# ---- evolution (the `give` fraction is the heritable PURPOSE-toward-other trait) ----
POP = 60
GENS = 80
MUT = 0.06
KEEP = 12
BONUS = 0.0            # (kept for backward-compat references; the interlock replaces it)


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def fitness(give, with_other, seed):
    """One life. `give` in [0,1] = fraction of the per-tick give the subject is WILLING to
    spend on the other (its purpose-toward-other). with_other = is the other even there.
    REAL INTERLOCK (docs/36 rule 4): if the other is alive, you can eat enough to survive the
    whole run; if it is dead, you drain and DIE before the end. So keeping the other alive is
    survival-critical, and a purpose toward it can GROW. Fit = survive to the end (the world's
    verdict on 'did you live') + a small tie-break on energy."""
    rng = random.Random(seed)
    A = START_A
    B = B_START if with_other else None
    give_spent = 0
    survived = False
    for t in range(TICKS):
        interlock_ok = (not with_other) or (B is not None and B > 0)
        if A > 0:
            A += BOTH_YIELD if interlock_ok else SOLO_YIELD
        A -= METAB
        if with_other and B is not None and B > 0:
            g = min(GIVE_UNIT * give, A - 1.0)
            A -= g
            B += g
            give_spent += g
        if with_other and B is not None:
            B -= B_METAB
        if A <= 0:
            break
    survived = A > 0
    return (1.0 if survived else 0.0) + 0.001 * max(0.0, A), give_spent


def evolve(with_other, seed, gens=GENS):
    """Evolve the PURPOSE-toward-other trait `give`. Designer sets ONLY whether the other is there."""
    rng = random.Random(seed)
    pop = [clamp(rng.random()) for _ in range(POP)]       # heritable give fraction
    for _g in range(gens):
        fit = [fitness(p, with_other, seed + i)[0] for i, p in enumerate(pop)]
        order = sorted(range(POP), key=lambda i: -fit[i])[:KEEP]
        parents = [pop[i] for i in order]
        newpop = []
        for _ in range(POP):
            p = parents[rng.randrange(KEEP)]
            np = p + (rng.gauss(0, 0.08) if rng.random() < MUT else 0)
            newpop.append(clamp(np))
        pop = newpop
    return sum(pop) / len(pop)


def survive(give, with_other, seeds=range(300)):
    a_alive, b_alive = 0, 0
    n = 0
    for s in seeds:
        r = _run_stats(give, with_other, s)
        a_alive += int(r[0]); b_alive += int(r[1]); n += 1
    return round(a_alive / n, 3), round(b_alive / n, 3)


def _run_stats(give, with_other, seed):
    rng = random.Random(seed)
    A = START_A
    B = B_START if with_other else None
    for t in range(TICKS):
        interlock_ok = (not with_other) or (B is not None and B > 0)
        if A > 0:
            A += BOTH_YIELD if interlock_ok else SOLO_YIELD
        A -= METAB
        if with_other and B is not None and B > 0:
            g = min(GIVE_UNIT * give, A - 1.0)
            A -= g
            B += g
        if with_other and B is not None:
            B -= B_METAB
        if A <= 0:
            break
    return (A > 0, (B is not None and B > 0))


def sweep(seeds=range(8)):
    print("=== SEED-38: can the subject's PURPOSE toward an other GROW -- not be set by me? ===")
    print("No designer 'love the other' goal. The `give` trait (budget spent on the other) is")
    print("HERITABLE, evolved by the world. Only knob = is the other even there (coupled exist).")
    print(f"{'with_other':<11} {'evolved_give':<14} {'A_alive%':<9} {'B_alive%':<9} reading")
    out = []
    for with_other in (False, True):
        gs = [evolve(with_other, sd) for sd in seeds]
        mean_give = sum(gs) / len(gs)
        a_alive, b_alive = survive(mean_give, with_other)
        if with_other:
            tag = "interlock makes the other's survival=MINE -> a non-zero give GROWS (give=0 is fatal, give>=0.5 keeps both alive)"
        else:
            tag = "no other -> nothing to give to -> give is irrelevant (drift, ~uniform)"
        out.append({"with_other": bool(with_other), "evolved_give": round(mean_give, 3),
                    "A_alive_rate": a_alive, "B_alive_rate": b_alive, "reading": tag})
        print(f"{str(with_other):<11} {mean_give:<14.3f} {a_alive:<9.3f} {b_alive:<9.3f}  {tag}")
    return out


def main():
    p = argparse.ArgumentParser(description="SEED-38: can the other's purpose grow?")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--with-other", type=int, choices=[0, 1], default=1)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    if args.sweep:
        out = sweep()
        with open("seed-38/results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("\nfull results -> seed-38/results.json")
        return
    g = evolve(bool(args.with_other), args.seed)
    print(json.dumps({"with_other": bool(args.with_other), "seed": args.seed,
                      "evolved_give": round(g, 3)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
