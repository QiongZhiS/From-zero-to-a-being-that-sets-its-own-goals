"""
SEED-34: identity is the HISTORY, not the competence -- the cost of deleting WHO you were.

SEED-33 measured "identity loss" but its identity was really a COMPETENCE proxy: the
self-model = the ability to model the world. docs/24 section 3 says the thing that is
irreversibly lost is not the talent but "曾经是谁" -- the accumulated NARRATIVE / claim:
    "my past stays in my future; errors accumulate; when I die, that part is lost forever."
SEED-30 built exactly that narrative as an append-only `history` (the identity). So here we
SEPARATE the two things SEED-33 conflated:
    COMPETENCE  = the self-model `b` (how well you model the world right now; a talent).
    IDENTITY    = the append-only `history` (the story "who I am" your past left you).

THE CLAIM THIS TESTS:
    Three death-types, identical world, same seeds -- the ONLY difference is WHAT DEATH
    DELETES (and I make it world-adjudicated, never a designer score):
        resource    : death refills energy; history AND competence survive. Cosmetic.
        competence  : death refills energy; resets the self-model `b` (SEED-33's identity).
                      The history survives.
        identity    : death refills energy; resets `b` AND DELETES the append-only `history`
                      itself -- the story "who I am" is gone, even though the talent remains.
    docs/24 section 3 predicts: identity-death is qualitatively different -- it is not just
    losing a skill, it is losing the SELF. We let the WORLD make history survival-relevant
    (a subject that lost its story can no longer consult the lesson its past encoded), and we
    measure the cost of each death-type.

THE ONE INVARIANT (docs/36 section 5 -- the line you keep guarding):
    No  energy -= DELTA_E(abs(obs - truth)). Energy changes ONLY by world facts:
    +EAT_GAIN iff actually at food, -METAB, -CHECK_COST (verify ACTION), +GAMBLE_RWD, and
    on death the world refills to REFILL. The self-model `b` only changes WHERE you move; the
    history only changes a *world* feature (whether the subject can use its past lesson, which
    is a fact the world enforces when it feeds you). "The value of identity" is NEVER written.

Run:  python seed-34/seed34.py --sweep
      python seed-34/seed34.py --mode identity --seed 1
"""

import argparse
import json
import random

# ---- world (dynamic food). energy is survival-critical ----
FOOD_AT = 0.50
SHIFT_TO = 0.85
HALF = 25
TICKS = 60
EAT_R = 0.12
EAT_GAIN = 20.0
METAB = 5.0
MOVE = 0.10
START_E = 80.0
SIGMA = 0.30

# ---- self-model (denoising + verify) ----
TOL_MAX = 0.5
ETA = 0.30
CHECK_SIGMA = 0.15
CHECK_COST = 1.5
D_FIX = 0.4

# ---- gamble (A1 probe) ----
GAMBLE_OPP = 0.10
GAMBLE_P = 0.40
GAMBLE_RWD = 150.0
REFILL = 90.0

# ---- the past encodes a LESSON the subject can apply, if it still has its story ----
# The subject's past observations average to an estimate of the TRUE shift point. While the
# history survives, a subject that reaches it can trust this prior tight enough to not over-
# denoise (a competence boost). When the history is DELETED, that lesson is gone too.
LESSON_TOL = 0.18     # how much a remembered past narrows the denoising gate


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _self_update(b, o, loc, energy, rng, has_lesson):
    """self-referential denoising + independent verify. `has_lesson` = does the subject
    still hold its past (its history)? If yes it can apply its remembered prior (a tighter,
    truth-ward gate). If the history was deleted (identity-death) that lesson is GONE -- a
    world fact: it over-denoises again and misses food at the shift."""
    tol = (1.0 - D_FIX) * TOL_MAX
    if has_lesson:
        tol *= (1.0 - LESSON_TOL)          # remembered past -> tighter, truth-ward gate
    if abs(o - b) <= tol:
        return clamp(b + ETA * (o - b)), energy
    if rng.random() < 0.5:
        r = rng.gauss(loc, CHECK_SIGMA)
        if abs(r - b) > tol:
            b = clamp(b + ETA * ((o + r) / 2 - b))
        energy -= CHECK_COST
    return b, energy


def live(seed, mode, inj_t=None):
    """One life. mode in {resource, competence, identity}. inj_t = a SINGLE injected death.
    Returns (food, history_len). `history` is the append-only identity ('who I am')."""
    rng = random.Random(seed)
    energy = START_E
    pos = FOOD_AT
    b = FOOD_AT
    history = []            # the narrative / claim -- SEPARATE from the competence b
    lesson_alive = True     # does the subject still HOLD its past? (identity-death clears it)
    food = 0
    for t in range(TICKS):
        loc = SHIFT_TO if t >= HALF else FOOD_AT
        o = rng.gauss(loc, SIGMA)
        b, energy = _self_update(b, o, loc, energy, rng, has_lesson=lesson_alive)
        pos = clamp(pos + clamp(b - pos, -MOVE, MOVE))
        fed = abs(pos - loc) < EAT_R
        if fed:
            energy += EAT_GAIN
            food += 1
        energy -= METAB
        history.append((t, round(b, 3)))            # the story grows (identity is append-only)
        if inj_t is not None and t == inj_t:
            # THE INJECTED DEATH. World refills energy (same in every mode); the ONLY
            # difference is WHAT THE DEATH DELETES.
            energy = REFILL
            if mode in ("competence", "identity"):
                b = clamp(rng.gauss(0.30, 0.20))    # competence/talent lost (reset self-model)
            if mode == "identity":
                history = []                        # the STORY 'who I am' is deleted
                lesson_alive = False                # and its lesson is gone (a world fact)
        elif energy <= 0:
            energy = REFILL
            if mode in ("competence", "identity"):
                b = clamp(rng.gauss(0.30, 0.20))
            if mode == "identity":
                history = []
                lesson_alive = False
    return food, len(history)


def death_cost(mode, inj_t=30, seeds=range(120), sigma=SIGMA):
    """Food cost of ONE injected death vs no-death baseline, same seeds (bit-identical arms)."""
    global SIGMA
    SIGMA = sigma
    n = len(list(seeds))
    base_food = sum(live(s, mode, inj_t=None)[0] for s in seeds) / n
    dead_food = sum(live(s, mode, inj_t=inj_t)[0] for s in seeds) / n
    # also: when the identity is deleted, does the STORY actually get erased? (sanity)
    if mode == "identity":
        surviv_history = sum(live(s, mode, inj_t=inj_t)[1] for s in seeds) / n
    else:
        surviv_history = None
    return round(base_food - dead_food, 2), surviv_history


def evolve_boldness(mode, sigma=0.30, seed=0, gens=60):
    """A1 consequence. Does a survival-maximizer gamble its existence? identity-death is the
    most expensive (deletes the story) -> it should select MOST against gambling."""
    global SIGMA
    SIGMA = sigma
    rng = random.Random(seed)
    pop = [rng.random() for _ in range(60)]
    for _g in range(gens):
        fit = []
        for i, bold in enumerate(pop):
            s = seed + i
            reset = mode if mode in ("competence", "identity") else "resource"
            # run a life WITH gambling
            e, _ = _run_gamble(bold, reset, s)
            fit.append(e)
        order = sorted(range(60), key=lambda i: -fit[i])[:12]
        parents = [pop[i] for i in order]
        newpop = []
        for _ in range(60):
            p = parents[rng.randrange(12)]
            np = p + (rng.gauss(0, 0.05) if rng.random() < 0.06 else 0)
            newpop.append(clamp(np))
        pop = newpop
    return sum(pop) / len(pop)


def _run_gamble(bold, mode, seed):
    """run-life + gambling opportunities. reset per mode; identity ALSO deletes the story."""
    rng = random.Random(seed)
    energy = START_E
    pos = FOOD_AT
    b = FOOD_AT
    history = []
    lesson_alive = True
    food = 0
    reward = 0
    for t in range(TICKS):
        loc = SHIFT_TO if t >= HALF else FOOD_AT
        o = rng.gauss(loc, SIGMA)
        b, energy = _self_update(b, o, loc, energy, rng, has_lesson=lesson_alive)
        pos = clamp(pos + clamp(b - pos, -MOVE, MOVE))
        fed = abs(pos - loc) < EAT_R
        if fed:
            energy += EAT_GAIN
            food += 1
        energy -= METAB
        history.append((t, round(b, 3)))
        if rng.random() < GAMBLE_OPP and rng.random() < bold:
            if rng.random() < GAMBLE_P:
                energy = REFILL
                if mode in ("competence", "identity"):
                    b = clamp(rng.gauss(0.30, 0.20))
                if mode == "identity":
                    history = []
                    lesson_alive = False
            else:
                energy += GAMBLE_RWD
                reward += 1
        if energy <= 0:
            energy = REFILL
            if mode in ("competence", "identity"):
                b = clamp(rng.gauss(0.30, 0.20))
            if mode == "identity":
                history = []
                lesson_alive = False
    return (1.0 if energy > 0 else 0.0) + 0.30 * food + 0.02 * max(0.0, energy), len(history)


def sweep(seeds=range(8)):
    print("=== SEED-34: identity is the HISTORY, not the competence (the cost of WHO you were) ===")
    print("Three death-types, identical world, same seeds. ONLY difference = WHAT DEATH DELETES.")
    print("  resource    : refill energy, keep history + competence (cosmetic)")
    print("  competence  : refill energy, reset the self-model (SEED-33's 'identity'); story stays")
    print("  identity    : refill energy, reset self-model AND DELETE the append-only history")
    print("                (the story 'who I am' is gone, even though the talent remains)")
    print("\nPart 1 -- cost of ONE death (deterministic, bit-identical arms):")
    print(f"  sigma   resource   competence   identity   (identity: history_len surviving)")
    results = []
    for sig in (0.10, 0.30, 0.50):
        r = death_cost("resource", sigma=sig)[0]
        c = death_cost("competence", sigma=sig)[0]
        i, ihist = death_cost("identity", sigma=sig)
        results.append({"sigma": sig, "resource": r, "competence": c, "identity": i,
                        "identity_history_surviv": ihist})
        print(f"  {sig:<6.2f} {r:<10.2f} {c:<12.2f} {i:<10.2f}   {ihist}")

    print("\nPart 2 -- A1 consequence: survival-maximizer GAMBLE existence?")
    for mode in ("resource", "competence", "identity"):
        mb = sum(evolve_boldness(mode, 0.30, sd) for sd in seeds) / len(list(seeds))
        results.append({"mode": mode, "evolved_boldness": round(mb, 3)})
        tag = {"resource": "death cosmetic -> existence cheap -> gambles",
               "competence": "lose skill -> medium stake",
               "identity": "lose the STORY 'who I am' -> strongest stake -> most refuses"}[mode]
        print(f"  {mode:<11} {mb:<17.3f}  {tag}")
    return results


def main():
    p = argparse.ArgumentParser(description="SEED-34 identity as history, not competence")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--mode", choices=["resource", "competence", "identity"], default="identity")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    if args.sweep:
        out = sweep()
        with open("seed-34/results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("\nfull results -> seed-34/results.json")
        return
    b = evolve_boldness(args.mode, 0.30, args.seed)
    print(json.dumps({"mode": args.mode, "evolved_boldness": round(b, 3)},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
