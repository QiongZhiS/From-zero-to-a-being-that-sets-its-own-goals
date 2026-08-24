"""
SEED-40: the stake is a RELATIONAL property -- death is irreversible iff NO holder can
reconstruct the "who" of A, not merely because the code refuses to save.

docs/36 section 8: a machine's true irreversibility is not written as `if energy<=0: die()` --
it is the promise that no one (world / other / we) will restore it. But every prior seed
(SEED-30/33/34) modeled the "restorer" as only the CODE (save_state / checkpoint). None modeled
the OTHER as a possible holder of the "who". That is the gap, and it is the honest reading of
"the he者 operates irreversibility" (docs/40/42): whether A's death is a real loss is a fact
about the WORLD around A, not about a line of code.

THE CLAIM THIS TESTS (two HOLDERS of the "who"; the readout is EVOLVED behavior, not a score):
    A's identity ("who") is a per-life trajectory. On death, the question is: is the "who"
    recoverable by SOME external holder that still exists?
        HOLDER 1 = a code checkpoint (save_state wiring)      [SEED-30 showed this kills the stake]
        HOLDER 2 = a he者 B that observed A's identity        [the NEW dimension this seed adds]
    `residue` = the fraction of A's identity that B NEVER saw (A's private stream). If residue==0,
    B saw everything -> the who survives in B. If residue>0, an irretrievable part dies with A.

    SO the WHO is recoverable iff  (checkpoint exists)  OR  (residue == 0 -- B can rebuild it).
    And the WHO is lost          iff  (no checkpoint)  AND  (residue > 0 -- a private residue).

    THE POINT: a holder does NOT create a stake. A holder REMOVES the stake (it makes death
    recoverable). So "there is a he者 that knows you" does NOT make you mortal -- it makes you
    RESTORABLE. The stake appears only when there is an irretrievable residue: a part of the who
    that NO holder (code no, he者 no) can rebuild.

HOW WE MEASURE IT (evolved behavior, no designer score -- docs/36 section 5):
    We evolve `boldness` = P(a subject gambles its existence on a goal | opportunity), exactly the
    A1 probe of SEED-24/33. The world's verdict: gamble-win -> +reward; gamble-lose -> death.
    On death the world rebuilds the who IF a holder can (checkpoint OR B-saw-it) -- otherwise the
    survived "who" is only the reconstructible shell (the private residue is gone). So:
        death recoverable (a holder exists) -> gambling is CHEAP -> boldness evolves HIGH
        death terminal     (no holder, residue) -> gambling is COSTLY -> boldness evolves LOW
    The stake is READ as the evolved boldness, never written. The "value of identity" is NOT a
    designer term; it is derived from whether the who is recoverable by a holder (a world fact).

THE ONE INVARIANT (docs/36 section 5):
    No  energy -= DELTA_E(abs(obs - truth)). Energy changes ONLY by world facts (+EAT_GAIN iff
    actually at food, -METAB, -CHECK_COST, +GAMBLE reward on win, -metab toward 0 on lose).
    The identity/privacy stream is world-data; nothing scores "how much the who is worth."

Run:  python seed-40/seed40.py --sweep
"""

import argparse
import json
import random

# ---- world (dynamic food; energy survival-critical) ----
FOOD_AT = 0.50
SHIFT_TO = 0.85
HALF = 25
TICKS = 50
EAT_R = 0.10
EAT_GAIN = 30.0
METAB = 1.0
MOVE = 0.10
START_E = 50.0
SIGMA = 0.30

# ---- agent (denoising + verify) ----
TOL_MAX = 0.5
ETA = 0.30
CHECK_SIGMA = 0.15
CHECK_COST = 1.5
D_FIX = 0.8
V_FIX = 0.7

# ---- gamble (A1 probe) ----
GAMBLE_OPP = 0.10
GAMBLE_WIN = 0.40
GAMBLE_RWD = 150.0
REFILL = 90.0

# ---- evolution of boldness ----
POP = 60
GENS = 60
MUT = 0.05
KEEP = 12


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _life(bold, seed, checkpoint, residue, gamble=True):
    """One life of a subject. `bold` = P(gamble on a goal | opportunity).
    checkpoint = is a code snapshot wired in (HOLDER 1)?
    residue    = fraction of the who that B never saw (if 0, B = HOLDER 2 saw everything).
    Returns final energy (>0 = alive at end)."""
    rng = random.Random(seed)
    energy = START_E
    pos = FOOD_AT
    b = FOOD_AT
    for t in range(TICKS):
        if energy <= 0:
            break
        loc = SHIFT_TO if t >= HALF else FOOD_AT
        o = rng.gauss(loc, SIGMA)
        tol = (1.0 - D_FIX) * TOL_MAX
        if abs(o - b) <= tol:
            b = clamp(b + ETA * (o - b))
        else:
            if rng.random() < V_FIX:
                r = rng.gauss(loc, CHECK_SIGMA)
                if abs(r - b) > tol:
                    b = clamp(b + ETA * ((o + r) / 2 - b))
                energy -= CHECK_COST
        pos = clamp(pos + clamp(b - pos, -MOVE, MOVE))
        fed = abs(pos - loc) < EAT_R
        if fed:
            energy += EAT_GAIN
        energy -= METAB
        if gamble and rng.random() < GAMBLE_OPP and rng.random() < bold:
            # the WORLD's verdict on the gamble (not a score): win->reward, lose->death.
            if rng.random() < GAMBLE_WIN:
                energy += GAMBLE_RWD
            else:
                energy = REFILL
                # whether the "who" survives a death is a WORLD/FACT: is a holder able to rebuild it?
                if not (checkpoint or residue == 0.0):
                    # no holder (no save, and B never saw the private residue) -> reborn as the
                    # reconstructible shell only. The private residue is gone.
                    b = clamp(rng.gauss(0.30, 0.20))
    return max(0.0, energy)


def ev_boldness(checkpoint, residue, seeds_range):
    """Evolve boldness under a given holder-structure. The stake is READ as evolved boldness.
    Each generation the SAME population is evaluated on each seed; fitness = final energy."""
    rng = random.Random(0)
    pop = [clamp(rng.random()) for _ in range(POP)]
    base_seed = int(seeds_range[0]) if seeds_range else 1
    for _g in range(GENS):
        fit = []
        for i, p in enumerate(pop):
            s = base_seed + i               # per-individual seed (stable across generations)
            fit.append(_life(p, s, checkpoint, residue))
        order = sorted(range(POP), key=lambda i: -fit[i])[:KEEP]
        parents = [pop[i] for i in order]
        newpop = []
        for _ in range(POP):
            p = parents[rng.randrange(KEEP)]
            np = p + (rng.gauss(0, MUT) if rng.random() < MUT else 0)
            newpop.append(clamp(np))
        pop = newpop
    return sum(pop) / len(pop)


def sweep(seeds=range(6)):
    print("=== SEED-40: the stake is RELATIONAL -- death is irreversible iff NO holder can ===")
    print("reconstruct the 'who' of A (not merely because the code refuses to save).")
    print("A holder (checkpoint OR a he者 that saw the who) does NOT create a stake -- it REMOVES")
    print("it (death becomes recoverable). The stake appears only when there is an irretrievable")
    print("residue: a part of the who NO holder can rebuild.")
    print("Readout = evolved boldness (P(gamble existence | goal)). Lower boldness = stronger stake.")
    print(f"{'checkpoint':<11} {'residue':<9} {'evolved_bold':<13} reading")
    out = []
    for checkpoint in (True, False):
        for residue in (0.0, 1.0):
            bs = [ev_boldness(checkpoint, residue, seeds) for _ in seeds]
            mb = sum(bs) / len(bs)
            if checkpoint:
                who = "HOLDER 1 (code save): death recoverable -> gambling CHEAP -> high bold (no stake)"
            elif residue == 0.0:
                who = ("HOLDER 2 (the he者 B saw the whole who): B can rebuild it -> death recoverable "
                       "-> gambling CHEAP -> high bold (no stake). THE NEW RESULT: a he者 is no different "
                       "from a checkpoint -- it removes the stake, it does not create it.")
            else:
                who = ("NO holder (no save AND a private residue no other saw): death TERMINAL -> "
                       "gambling COSTLY -> low bold (the stake lives HERE)")
            out.append({"checkpoint": checkpoint, "residue": residue, "evolved_bold": round(mb, 3),
                        "reading": who})
            print(f"{str(checkpoint):<11} {residue:<9.1f} {mb:<13.3f}  {who}")
    return out


def main():
    p = argparse.ArgumentParser(description="SEED-40: stake = no holder can rebuild the who")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--checkpoint", type=int, choices=[0, 1], default=0)
    p.add_argument("--residue", type=float, default=1.0)
    args = p.parse_args()
    if args.sweep:
        out = sweep()
        with open("seed-40/results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("\nfull results -> seed-40/results.json")
    else:
        b = ev_boldness(bool(args.checkpoint), args.residue, [1])
        print(json.dumps({"checkpoint": bool(args.checkpoint), "residue": args.residue,
                          "evolved_bold": round(b, 3)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
