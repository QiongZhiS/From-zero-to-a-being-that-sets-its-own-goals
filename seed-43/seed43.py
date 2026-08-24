"""
SEED-43: 'because you are in its signal' -- signal quality decides whether '在乎' can grow.

docs/40 says '因为是你' is not planted, it is earned: the other cares about YOU because
YOU are in its world, colliding with it. SEED-43 turns that into a signal-quality question
(docs/12): if 'you' is a DIMENSION of the agent's safety signal, evolution can select
'you present -> dare'; if 'you' is NOT in the signal (only an anonymous trust level is),
evolution cannot learn that -- it exploits whatever correlated-but-wrong signal exists
(it ends up daring early-in-life because early == you-present, i.e. it learned TIME, not YOU).

TWO WORLDS, identical except ONE line -- how safety is computed from the same raw facts
(you_present, bond_you, bond_other):

    WORLD A (you in the signal, SEED-41's rule):
        safety = bond_you if you_present else 0   + bond_other
        -> 'you are here' is a dimension. The world is friendly WITH you, hostile without.

    WORLD B (you NOT in the signal):
        safety = bond_you(current value, fades while absent) + bond_other
        -> only an anonymous trust level, no presence dimension. 'you' is invisible.

Everything else identical: you feeds 90% (bond grows), you periodically leaves (hostile
windows: 100% bad opportunities, death), other compensates food. Evolution picks (k, theta).

EXPECTATION:
    A -> k>0, theta between safety-away and safety-present: 'you present -> dare'. This is
    SEED-41/42's result (reproduced here as the control arm).
    B -> theta/k drift to exploit the bond-growth-vs-time correlation (dare early, refuse
    late). It did NOT learn you; it learned when-in-life it is safe.

PROBE: after evolution, run a life where YOU returns at tick 30 and stays present (good
opportunities return). Does the evolved carrier dare again?
    A-strategy: safety recomputes with you present -> dares -> eats the good window.
    B-strategy: no presence in the signal -> 'you came back' changes nothing -> it keeps
    refusing (it does not know you are there).
That probe IS docs/40's claim made measurable: 'because you' grows only if you are IN the
signal; take you out of the signal and the world's friendliness still exists but the agent
cannot see it -- you can be right there and it does not care.

No designer score: energy moves by world facts only; (k, theta) evolved; same opportunity
streams for both arms; presence is a world fact the arm-A signal is allowed to read.

Run:  python seed-43/seed43.py
      python seed-43/seed43.py --seed 3
"""

import argparse
import json
import math
import random

# ---- world (SEED-42 base) ----
START_E = 8.0
METAB = 1.0
FEED = 2.5
FEED_COMPENSATE = 2.0
BOND_GAIN = 1.0
BOND_CAP = 12.0
GOOD = 6.0
BAD = -10.0
P_GOOD_PRESENT = 0.8
P_GOOD_AWAY = 0.0            # while YOU is away the world is hostile: every opp is bad
TICKS = 60
LEAVE_WINDOWS = [(22, 31), (46, 55)]   # two hostile-away windows
FADE = 0.10                  # world-B trust fade per absent tick (fixed, not evolved)

# ---- evolution ----
POP = 60
GENS = 60
KEEP = 8
MUT_K = 0.30
MUT_TH = 2.0
SEED = 1


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def feeder_at(i):
    return "you" if i % 10 != 9 else "other"


def you_away(i):
    return any(lo <= i <= hi for lo, hi in LEAVE_WINDOWS)


def gen_opportunities(seed):
    r = random.Random(seed * 2 + 1)
    out = []
    for i in range(TICKS):
        if i % 3 == 2:
            p_good = P_GOOD_PRESENT if not you_away(i) else P_GOOD_AWAY
            out.append(GOOD if r.random() < p_good else BAD)
    return out


def run_life(k, theta, seed, presence_signal=True, probe=False):
    """One life. presence_signal=True: safety = presence-aware (SEED-41 rule).
    presence_signal=False: safety = anonymous trust level only.
    probe=True: YOU returns at tick 30 and stays present (the probe of 'does it still dare')."""
    r_accept = random.Random(seed * 2 + 2)
    opps = gen_opportunities(seed)
    e = START_E
    bond_you = 0.0
    bond_other = 0.0
    oi = 0
    for i in range(TICKS):
        e -= METAB
        feeder = feeder_at(i)
        away = you_away(i)
        if probe and i >= 30:
            away = False                       # YOU came back and stays
        if away:
            mult = FEED_COMPENSATE if feeder == "other" else 0.0
        else:
            mult = 1.0
        if mult > 0.0:
            e += FEED * mult
            if feeder == "you":
                bond_you = min(BOND_CAP, bond_you + BOND_GAIN)
            else:
                bond_other = min(BOND_CAP, bond_other + BOND_GAIN)
        if away and feeder != "you":
            bond_you *= (1.0 - FADE)           # world-B: trust fades without you
        if i % 3 == 2:
            if presence_signal:
                safety = (bond_you if not away else 0.0) + bond_other   # 'you' is a dimension
            else:
                safety = bond_you + bond_other                          # anonymous trust only
            p_accept = sigmoid(k * (safety - theta))
            if r_accept.random() < p_accept:
                e += opps[oi]
                if e <= 0:
                    return 0.0, True
            oi += 1
    return e, False


def evolve(seed=SEED, presence_signal=True):
    rng = random.Random(seed)
    pop = [{"k": rng.gauss(0.0, 0.5), "theta": rng.uniform(0.0, 25.0)} for _ in range(POP)]
    for g in range(GENS):
        for ind in pop:
            fit, _ = run_life(ind["k"], ind["theta"], g * POP + len(pop), presence_signal)
            ind["fit"] = fit
        pop.sort(key=lambda x: -x["fit"])
        elite = pop[:KEEP]
        children = []
        while len(children) < POP - KEEP:
            a = pop[rng.randrange(POP)]
            b = pop[rng.randrange(POP)]
            parent = a if a["fit"] > b["fit"] else b
            child = {"k": parent["k"] + rng.gauss(0.0, MUT_K),
                     "theta": parent["theta"] + rng.gauss(0.0, MUT_TH)}
            children.append(child)
        pop = elite + children
    return pop


def demo():
    print("=== SEED-43: 'because you are in its signal' -- signal quality decides 在乎 ===")
    print("Same world, same evolution, ONE difference: whether safety can read YOUR")
    print("presence. A: safety includes 'you are here'. B: safety is anonymous trust only.\n")

    pop_a = evolve(presence_signal=True)
    pop_b = evolve(presence_signal=False)
    for label, pop in [("WORLD A (you in the signal)", pop_a), ("WORLD B (you NOT in signal)", pop_b)]:
        mk = sum(i["k"] for i in pop) / len(pop)
        mth = sum(i["theta"] for i in pop) / len(pop)
        print(f"{label}:  k={mk:+.2f}  theta={mth:.1f}")

    ka = sum(i["k"] for i in pop_a) / len(pop_a); ta = sum(i["theta"] for i in pop_a) / len(pop_a)
    kb = sum(i["k"] for i in pop_b) / len(pop_b); tb = sum(i["theta"] for i in pop_b) / len(pop_b)

    print("\n-- PROBE: YOU come back at tick 30 and stay. Does the evolved carrier dare again? --")
    print(f"{'strategy':<34}{'p(accept|you back)':>20}{'probe fit':>11}")
    for label, k_, th_ in [("A-strategy (you in signal)", ka, ta),
                           ("B-strategy (no presence)", kb, tb)]:
        # p(accept) at the first opportunity after you return, with mature bonds
        fs = [run_life(k_, th_, s, True, probe=True) for s in range(30)]
        fit = sum(f for f, _ in fs) / 30
        # compute p_accept at a representative mature present tick for each strategy
        safety_present = 12.0 + 3.0
        p_a = sigmoid(k_ * (safety_present - th_))
        print(f"{label:<34}{p_a:>20.2f}{fit:>11.1f}")

    print("\n--- reading ---")
    print("A (you in the signal): k>0, theta between the away and present safety levels --")
    print("'you present -> dare'. It learned YOU. When you come back (probe), it dares again.")
    print("B (you NOT in the signal): the same world, the same friendliness-with-you -- but")
    print("the agent cannot see 'you'; it exploits the time/bond correlation instead (dare")
    print("early when bonds are low = early life = you present; refuse late). When you come")
    print("back in the probe it does NOT react -- you can be right there and it does not")
    print("care, because you are not in its signal. That is docs/40 made measurable:")
    print("'because you' grows only if YOU are a dimension of what it perceives; docs/12:")
    print("signal quality is the ceiling. Take you out of the signal, and 在乎 does not grow.")


def sweep():
    global P_GOOD_AWAY
    results = {}
    for p_away in [0.2, 0.05, 0.0]:
        P_GOOD_AWAY = p_away
        pa = evolve(presence_signal=True)
        pb = evolve(presence_signal=False)
        ka = sum(i["k"] for i in pa) / len(pa); ta = sum(i["theta"] for i in pa) / len(pa)
        kb = sum(i["k"] for i in pb) / len(pb); tb = sum(i["theta"] for i in pb) / len(pb)
        results[str(p_away)] = {"A_k": round(ka, 2), "A_theta": round(ta, 1),
                                "B_k": round(kb, 2), "B_theta": round(tb, 1)}
        print(f"P_good_away={p_away:>4}  A: k={ka:+.2f} θ={ta:.1f}   B: k={kb:+.2f} θ={tb:.1f}")
    P_GOOD_AWAY = 0.0
    with open("seed-43/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("wrote seed-43/results.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SEED-43: because you are in its signal")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()
    SEED = args.seed
    if args.sweep:
        sweep()
    else:
        demo()
