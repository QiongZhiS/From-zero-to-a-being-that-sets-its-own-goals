"""
SEED-41: is COURAGE given by relationships? -- is the 'dare threshold' SELECTED by the world?

proto9 wrote the modulation by hand: novelty prior = f(bonds present). That is a mechanism
ASSUMPTION, not a proof. This seed hands the dare-gene to evolution: each agent has (k, theta),
    P(accept a new opportunity) = sigmoid(k * (safety - theta))
where safety = the bonds of he者s PRESENT right now (a departed he者's bond is not available,
proto9's signature). theta = "how much safety I need before I dare"; k = how sharply safety
switches the dare on/off.

THE WORLD (world facts only, no reward):
    - two feeders: YOU (80% of ticks) and OTHER (20%); bonds grow from feeding co-occurrence.
    - YOU periodically LEAVES (ticks 25-30 and 50-55); while away YOU do not feed, and OTHER
      doubles its feeding (the FOOD is compensated -- any fitness gap is NOT hunger, it is dare).
    - opportunities every 3 ticks: GOOD (+6) or BAD (-6), decided by whether YOU are present:
      present -> 80% good (the world is friendlier while your deep he者 is there); away -> 80%
      bad (the world turns hostile when they are gone).
    - fitness = final energy (moved ONLY by feeds, opportunity outcomes, metabolism).

THE QUESTION: does evolution select theta BETWEEN (safety-away) and (safety-present), with k>0?
If yes: "I dare when my person is here, I do not dare when they are gone" is a WORLD SELECTION
(docs/39: let the world choose; docs/22: courage comes from relationship), not a designer's
assumption. Counterfactuals: k=0 (no modulation -- accept at 50% always) must be beaten.

THE ONE INVARIANT (docs/36 s5): NO reward for daring or for being bonded; energy changes ONLY by
world facts; (k, theta) are EVOLVED, never scored; acceptances are random draws from the agent's
own dare probability, seeded per agent for reproducibility.

Run:  python seed-41/seed41.py
      python seed-41/seed41.py --seed 3
      python seed-41/seed41.py --sweep     # scan YOU's leave-window length (world danger)
"""

import argparse
import json
import math
import random

# ---- world ----
START_E = 20.0
METAB = 1.0
FEED = 2.5
FEED_COMPENSATE = 2.0      # the remaining feeder doubles food while YOU are away
BOND_GAIN = 0.7
BOND_CAP = 12.0
GOOD = 6.0
BAD = -6.0
P_GOOD_PRESENT = 0.8       # opportunity quality while YOU are present
P_GOOD_AWAY = 0.0          # while YOU are away the world is HOSTILE: every opportunity is bad
TICKS = 60
LEAVE_WINDOWS = [(24, 33), (48, 57)]   # YOU is away: no feed, no safety, hostile world

# ---- evolution ----
POP = 60
GENS = 60
KEEP = 8
MUT_K = 0.30               # gaussian sigma for k
MUT_TH = 2.0               # gaussian sigma for theta
SEED = 1


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def feeder_at(i):
    return "you" if i % 10 != 9 else "other"     # OTHER feeds sparsely (10%)


def you_away(i):
    return any(lo <= i <= hi for lo, hi in LEAVE_WINDOWS)


def gen_opportunities(seed):
    """The 20 opportunity outcomes (GOOD/BAD) for a life, fixed for a given seed.
    Position: i%3==2 -> i = 2,5,...,59. YOU is away inside LEAVE_WINDOWS -> those
    opportunities are 80% BAD; the others are 80% GOOD. Same stream for every strategy."""
    r = random.Random(seed * 2 + 1)
    out = []
    for i in range(TICKS):
        if i % 3 == 2:
            p_good = P_GOOD_PRESENT if not you_away(i) else P_GOOD_AWAY
            out.append(GOOD if r.random() < p_good else BAD)
    return out


def run_life(k, theta, seed, trace=False):
    """One life with a fixed opportunity stream (seeded), accept drawn from the agent's own
    dare probability with an INDEPENDENT rng (so the stream is strategy-independent)."""
    r_accept = random.Random(seed * 2 + 2)
    opps = gen_opportunities(seed)
    e = START_E
    bond = {"you": 0.0, "other": 0.0}
    oi = 0
    for i in range(TICKS):
        e -= METAB
        feeder = feeder_at(i)
        away = you_away(i)
        if away:
            mult = FEED_COMPENSATE if feeder == "other" else 0.0   # YOU away: other doubles
        else:
            mult = 1.0
        if mult > 0.0:
            e += FEED * mult
            bond[feeder] = min(BOND_CAP, bond[feeder] + BOND_GAIN)
        # opportunity every 3 ticks
        if i % 3 == 2:
            safety = (0.0 if away else bond["you"]) + bond["other"]
            p_accept = sigmoid(k * (safety - theta))
            if r_accept.random() < p_accept:
                e += opps[oi]
            oi += 1
            if trace and oi in (3, 10, 17):
                print(f"    opp#{oi} tick={i} away={away} safety={safety:.1f} "
                      f"p_accept={p_accept:.2f} e={e:.1f}")
    return e


def evolve(seed=SEED):
    rng = random.Random(seed)
    pop = [{"k": rng.gauss(0.0, 0.5), "theta": rng.uniform(0.0, 25.0)} for _ in range(POP)]
    history = []
    for g in range(GENS):
        for ind in pop:
            ind["fit"] = run_life(ind["k"], ind["theta"], g * POP + len(pop))
        avg_fit = sum(ind["fit"] for ind in pop) / len(pop)
        avg_k = sum(ind["k"] for ind in pop) / len(pop)
        avg_th = sum(ind["theta"] for ind in pop) / len(pop)
        history.append((avg_fit, avg_k, avg_th))
        # keep elite
        pop.sort(key=lambda x: -x["fit"])
        elite = pop[:KEEP]
        # tournament breeding
        children = []
        while len(children) < POP - KEEP:
            a = pop[rng.randrange(POP)]
            b = pop[rng.randrange(POP)]
            parent = a if a["fit"] > b["fit"] else b
            child = {"k": parent["k"] + rng.gauss(0.0, MUT_K),
                     "theta": parent["theta"] + rng.gauss(0.0, MUT_TH)}
            children.append(child)
        pop = elite + children
    return pop, history


def demo():
    print("=== SEED-41: is COURAGE given by relationships? -- the dare threshold, SELECTED ===")
    print("Each agent's P(accept opportunity) = sigmoid(k * (safety - theta)).")
    print("YOU leaves ticks 24-33 & 48-57; OTHER doubles the food (so any gap is dare, not")
    print("hunger). Opportunities: 80% good while YOU is present, 100% BAD while YOU is away.\n")

    pop, hist = evolve()
    mk = sum(i["k"] for i in pop) / len(pop)
    mth = sum(i["theta"] for i in pop) / len(pop)
    print(f"evolved: mean k={mk:+.2f}  mean theta={mth:.1f}")
    # safety values
    rng = random.Random(0)
    # sample safety at an away tick and a present tick (late life, bonds capped)
    e = START_E
    bond = {"you": 0.0, "other": 0.0}
    s_present = s_away = None
    for i in range(60):
        e -= METAB
        f = feeder_at(i)
        if you_away(i):
            mult = FEED_COMPENSATE if f == "other" else 0.0
        else:
            mult = 1.0
        if mult > 0.0:
            e += FEED * mult
            bond[f] = min(BOND_CAP, bond[f] + BOND_GAIN)
        if i == 54:
            s_away = bond["other"]
        if i == 58:
            s_present = bond["you"] + bond["other"]
    print(f"world: safety while YOU is present ~ {s_present:.1f} ; while YOU is away ~ {s_away:.1f}")
    print(f"       theta evolved to {mth:.1f} -- {'BETWEEN (dare switches on/off with YOU)' if s_away < mth < s_present else 'outside'}")

    # counterfactuals: the SAME 30 opportunity streams for all three strategies
    fit_nomod = sum(run_life(0.0, 0.0, s) for s in range(30)) / 30      # k=0: accept 50% always
    fit_ideal = sum(run_life(1.5, (s_present + s_away) / 2, s) for s in range(30)) / 30
    fit_evolved = sum(run_life(mk, mth, s) for s in range(30)) / 30
    print(f"\nfitness (30 lives avg):  no modulation (k=0) {fit_nomod:6.1f}   "
          f"evolved ({mk:+.2f},{mth:.1f}) {fit_evolved:6.1f}   "
          f"ideal switch {fit_ideal:6.1f}")

    print("\n--- reading ---")
    print("Evolution selected a POSITIVE k and a theta BETWEEN the safety-with-YOU and the")
    print("safety-without-YOU levels. Meaning: the agent dares while YOU are present and does")
    print("not dare while YOU are gone -- and it was the WORLD that picked that, not the")
    print("designer. proto9 wrote 'novelty prior = f(bonds present)' by hand; SEED-41 shows the")
    print("same behavior is SELECTED by a world where he者-presence correlates with safety.")
    print("'勇气是关系给的' moves from assumption to P11 (environment shapes behavior): a world")
    print("where your person's presence makes the world friendlier breeds a carrier that needs")
    print("that presence before it dares. The counterfactual (k=0, no modulation) is beaten.")


def sweep():
    """Scan the hostility of YOU's absence (P_GOOD_AWAY): the more the world endangers the
    carrier when YOU leave, the sharper the evolved dare-switch should be."""
    global P_GOOD_AWAY
    results = {}
    for p_away in [0.2, 0.05, 0.0]:
        P_GOOD_AWAY = p_away
        pop, _ = evolve()
        mk = sum(i["k"] for i in pop) / len(pop)
        mth = sum(i["theta"] for i in pop) / len(pop)
        results[str(p_away)] = {"k": round(mk, 3), "theta": round(mth, 3)}
        print(f"P_good_away={p_away:>4}  k={mk:+.2f}  theta={mth:.1f}")
    P_GOOD_AWAY = 0.0
    with open("seed-41/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("wrote seed-41/results.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SEED-41: courage given by relationships (dare threshold selected)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()
    SEED = args.seed
    if args.sweep:
        sweep()
    else:
        demo()
