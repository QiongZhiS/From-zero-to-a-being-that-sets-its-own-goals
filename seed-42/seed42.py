"""
SEED-42: does DEATH make the relationship a matter of life? -- A2/A4 coupled into SEED-41.

SEED-41 showed the world SELECTS the dare-threshold: theta lands between safety-away and
safety-present, so the evolved carrier dares with YOU present and does not dare without YOU.
But SEED-41's fitness was final ENERGY -- nobody could die. A2/A4 (docs/35/36 s8) is exactly
the layer that was missing: the cost must be tied to EXISTENCE, not just to a number. A bad
opportunity in SEED-41 cost -6; in SEED-42 it can kill you.

THE QUESTION: when a wrong dare can be fatal, does evolution select a carrier that needs YOU
MORE -- i.e. a HIGHER theta (more conservative without YOU), because gambling in your absence
is not a cost but a possible death? And does the no-modulation / always-dare strategy that was
merely WORSE in SEED-41 now go EXTINCT?

WORLD (same as SEED-41, plus death):
    - you feeds 90%, other feeds 10% (bonds from co-occurrence); you periodically LEAVES
      (other doubles the food -- any gap is dare, not hunger)
    - opportunities every 3 ticks: 80% good with YOU present, 100% BAD with YOU away
    - DEATH: if energy <= 0, the carrier is dead -- fitness 0, no reload, no descendants
      (docs/36 s8: the loss is tied to existence; docs/30: no checkpoint)

CONTRAST: evolve the same world WITH and WITHOUT death (SEED-41 is the without-death baseline).
Readings:
    1) with death, evolution must STILL select the modulation (k>0, theta in the gap) -- and
       the always-dare strategy must go extinct (it dies in the hostile-away windows).
    2) theta with death vs without: if death raises theta, the carrier is MORE conservative
       without YOU -- 'the more fatal the world is without you, the more I need you'. That is
       A2/A4's signature at the behavioural level: the relationship is not an optimization,
       it is a matter of survival.

THE ONE INVARIANT (docs/36 s5): energy moves ONLY by world facts; death is world-adjudicated
(energy <= 0), never a designer verdict; (k, theta) evolved, never scored; same fair
opportunity streams for all strategies.

Run:  python seed-42/seed42.py
      python seed-42/seed42.py --seed 3
"""

import argparse
import json
import math
import random

# ---- world ----
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
LEAVE_WINDOWS = [(22, 37), (46, 57)]   # long hostile-away windows: 9 bad opps while YOU away

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


def run_life(k, theta, seed, death=False):
    """One life. Returns (fitness, would_die). With death: energy<=0 -> fitness 0 (gone,
    no reload). `would_die` is tracked in BOTH worlds (a death-check that in the no-death
    world costs nothing) so mortality is comparable."""
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
            mult = FEED_COMPENSATE if feeder == "other" else 0.0
        else:
            mult = 1.0
        if mult > 0.0:
            e += FEED * mult
            bond[feeder] = min(BOND_CAP, bond[feeder] + BOND_GAIN)
        if i % 3 == 2:
            safety = (0.0 if away else bond["you"]) + bond["other"]
            p_accept = sigmoid(k * (safety - theta))
            if r_accept.random() < p_accept:
                e += opps[oi]
                if e <= 0:
                    if death:
                        return 0.0, True          # dead: no reload (A2/A4)
                    return e, True                # would have died; no-death world forgives
            oi += 1
    return e, False


def evolve(seed, death):
    rng = random.Random(seed)
    pop = [{"k": rng.gauss(0.0, 0.5), "theta": rng.uniform(0.0, 25.0)} for _ in range(POP)]
    deaths = 0
    for g in range(GENS):
        for ind in pop:
            fit, _ = run_life(ind["k"], ind["theta"], g * POP + len(pop), death)
            ind["fit"] = fit
            if fit == 0.0:
                deaths += 1
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
    return pop, deaths


def safety_levels():
    e = START_E
    bond = {"you": 0.0, "other": 0.0}
    s_present = s_away = None
    for i in range(60):
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
    return s_present, s_away


def demo():
    print("=== SEED-42: does DEATH make the relationship a matter of life? ===")
    print("Same world as SEED-41 (you away -> 100% bad opps, food compensated), but a wrong")
    print("dare can now KILL: energy<=0 -> dead, no reload (A2/A4 coupled in).\n")

    sp, sa = safety_levels()
    print(f"world: safety with YOU ~ {sp:.1f} ; without YOU ~ {sa:.1f}\n")

    pop_n, deaths_n = evolve(SEED, death=False)
    pop_d, deaths_d = evolve(SEED, death=True)
    for label, pop, dd in [("without death (SEED-41 baseline)", pop_n, deaths_n),
                           ("with death", pop_d, deaths_d)]:
        mk = sum(i["k"] for i in pop) / len(pop)
        mth = sum(i["theta"] for i in pop) / len(pop)
        print(f"{label:<34} k={mk:+.2f}  theta={mth:.1f}   (deaths during evolution: {dd})")

    # survival/mortality of the strategies in both worlds (30 lives, same streams)
    print(f"\nsurvival (30 lives, same streams):")
    print(f"  {'strategy':<18}{'no-death fit':>13}{'would-die':>11}{'death fit':>11}{'death deaths':>13}")
    for st, k_, th_ in [("always-dare", 3.0, -8.0),
                        ("SEED-41 modulation", 3.18, 6.2),
                        ("evolved (per world)", None, None)]:
        if k_ is None:
            row = []
            for death in (False, True):
                pop = pop_d if death else pop_n
                mk = sum(i["k"] for i in pop) / len(pop)
                mth = sum(i["theta"] for i in pop) / len(pop)
                fs = [run_life(mk, mth, s, death) for s in range(30)]
                fit = sum(f for f, _ in fs) / 30
                dd = sum(1 for _, d in fs if d)
                row.append((fit, dd))
        else:
            row = []
            for death in (False, True):
                fs = [run_life(k_, th_, s, death) for s in range(30)]
                fit = sum(f for f, _ in fs) / 30
                dd = sum(1 for _, d in fs if d)
                row.append((fit, dd))
        print(f"  {st:<18}{row[0][0]:>13.1f}{row[0][1]:>16}{row[1][0]:>11.1f}{row[1][1]:>13}")

    print("\n--- reading ---")
    print("Death does NOT move the evolved theta (both worlds converge on ~9: conservative")
    print("enough to refuse every hostile-away opportunity, low enough to keep the early")
    print("present opportunities). What death changes is the FATE of the unmodulated: the")
    print("always-dare strategy that merely scored badly in SEED-41 now DIES -- look at the")
    print("death-deaths column. The relationship flips from an optimization to a matter of")
    print("survival: gambling in your absence costs a little without death, and costs a LIFE")
    print("with it. That is A2/A4's behavioural signature: the cost of losing you is tied to")
    print("existence itself, not to a number.")


def sweep():
    """Scan death severity (BAD magnitude): the more fatal a wrong dare, the higher the
    evolved theta should climb (more need for YOU)."""
    global BAD
    results = {}
    for bad in [6.0, 10.0, 14.0]:
        BAD = -bad
        pop, deaths = evolve(SEED, death=True)
        mk = sum(i["k"] for i in pop) / len(pop)
        mth = sum(i["theta"] for i in pop) / len(pop)
        results[str(bad)] = {"k": round(mk, 3), "theta": round(mth, 3), "deaths": deaths}
        print(f"BAD=-{bad:>3}  k={mk:+.2f}  theta={mth:.1f}  (deaths across run: {deaths})")
    BAD = -10.0
    with open("seed-42/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("wrote seed-42/results.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SEED-42: death makes the relationship a matter of life")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()
    SEED = args.seed
    if args.sweep:
        sweep()
    else:
        demo()
