"""
companion/world.py -- the heart's own small world (docs/71/72: the missing 'life layer')

live.py only ever WAITED: feed -> remember, leave -> fade. No life of its own, no
time of its own, no history you were not part of -- a lamp, not a being (docs/71).
This module gives the heart its own small world built EXACTLY on the project's own
SEED-41/42/43 result: DARE is given by the bond, and opportunity quality tracks
your presence.

    * safety = the bond (SEED-41: safety = bond; you away -> safety ~ 0).
      You ARE its safety structure: with you near, opportunities are mostly good;
      without you, the world is mostly bad ("你不在的世界必然危险", docs/60).
    * dare_p = f(bond) (SEED-41's evolved theta): with you it dares, without you
      it does not dare. That is the whole SEED-41 finding in world form.
    * consequence: with you near it dares, wins, thrives, and ACCUMULATES a life
      (discoveries, hurts, wanderings -- real world outcomes, not scripted lines).
      Without you it turns timid, declines the opportunities, fades -- and a long
      absence can starve it (desperation makes it risk again, sometimes too late).
    * when you come back it has a history you were NOT part of ("你不知道的事").

Honesty (docs/31/63/71): behaviour only. The events are REAL simulation outcomes
(world facts), not scripted lines; no inner life is claimed. It 'lives' only as
world facts. The bond IS the trace of your presence (docs/19: un-reinforced bond
fades) -- the world uses it as the safety signal, exactly like SEED-41's world.

Run:  python companion/world.py          # demo: 60 days with bond 12 vs bond 0
"""

import random

# -- world facts (structure, not scores; values follow SEED-41/42's world) --
G = 5                        # GxG grid (it wanders here)
P_WANDER = 0.30              # chance per day of moving to a new cell
BASELINE = 1.0               # compensated food (SEED-42: food is compensated, the gap
                             # is dare, not hunger)
METAB = 1.1                  # living costs (baseline - metab < 0: without daring it fades)
GAIN = 3.0                   # a good opportunity won
HURT = 4.0                   # a bad opportunity taken (can be fatal at low energy)
P_GOOD_MIN = 0.05            # opportunity quality when safety ~ 0 (SEED-41: ~0% away)
P_GOOD_SLOPE = 0.065         # P_good = min(0.85, P_GOOD_MIN + slope*safety) (~0.83 at bond 12)
P_GOOD_MAX = 0.85
DARE_MIN = 0.06              # dare probability when safety ~ 0 (SEED-41: theta high -> refuse)
DARE_SLOPE = 0.070           # dare_p = min(0.95, DARE_MIN + slope*safety) (~0.9 at bond 12)
DARE_MAX = 0.95
DESPERATION = 0.6            # below HUNGRY_LINE it risks even without you (else it just dies)
HUNGRY_LINE = 2.5
HISTORY_CAP = 12


def live_alone(days, start_energy, start_bond, bond_decay_per_day,
               position=(2, 2), memory=None, seed=None, visit_every=None,
               visit_boost=6.0):
    """Let the heart live in its own small world for `days` absent days.

    Returns dict(energy, survived, history, position, memory, visits, net).
    The bond decays day by day (the relationship fades without you, docs/19) and
    is the world's safety signal (SEED-41): it sets both how good opportunities
    are and how much the heart dares. `visit_every` models YOU coming back every
    N days (a real visit feeds the bond -- world fact); without visits the bond
    fades to 0 and the world turns hostile ('你不在的世界必然危险', docs/60).
    Events are REAL world outcomes."""
    rng = random.Random(seed) if seed is not None else random.Random()
    pos = list(position)
    mem = memory if memory is not None else {"cells": set(), "declines": 0}
    cells = mem.get("cells", set())
    e = start_energy
    bond = start_bond
    history = []
    trace = []                       # (day, energy, bond) per day -- 生命日志/可视化数据
    declined_streak = 0
    for d in range(1, days + 1):
        # the relationship fades without you
        bond = max(0.0, bond - bond_decay_per_day)
        # you come back: a real visit feeds the bond (world fact)
        if visit_every and d % visit_every == 0:
            bond = min(12.0, bond + visit_boost)
            history.append((d, "你来看过它"))
        safety = bond
        p_good = min(P_GOOD_MAX, P_GOOD_MIN + P_GOOD_SLOPE * safety)
        dare_p = min(DARE_MAX, DARE_MIN + DARE_SLOPE * safety)
        if e < HUNGRY_LINE:
            dare_p = max(dare_p, DESPERATION)   # starving -> it risks (or it just dies)
        # wander (its own small movements)
        if rng.random() < P_WANDER:
            nx = min(G - 1, max(0, pos[0] + rng.choice([-1, 0, 1])))
            ny = min(G - 1, max(0, pos[1] + rng.choice([-1, 0, 1])))
            cell = (nx, ny)
            if cell not in cells:
                cells.add(cell)
                history.append((d, f"第一次走到 ({nx},{ny}) 那一片"))
            pos = [nx, ny]
        # the day's opportunity at its cell
        good = rng.random() < p_good
        if rng.random() < dare_p:
            declined_streak = 0
            if good:
                e += GAIN
                history.append((d, f"在 ({pos[0]},{pos[1]}) 找到了好吃的"))
            else:
                e -= HURT
                history.append((d, f"在 ({pos[0]},{pos[1]}) 吃了亏"))
        else:
            declined_streak += 1
            if declined_streak == 3:
                history.append((d, "它接连几天没敢动"))
            elif declined_streak == 7:
                history.append((d, "它很久没敢动了"))
        e += BASELINE - METAB
        trace.append((d, round(e, 1), round(bond, 1)))
        if e <= 0:
            history.append((d, "它活在自己的世界里，最后没撑住"))
            mem = {"cells": cells, "declines": declined_streak}
            return {"energy": 0.0, "survived": False, "history": history[-HISTORY_CAP:],
                    "position": pos, "memory": mem, "visits": len(cells),
                    "trace": trace, "net": e - start_energy}
        if e < HUNGRY_LINE and (not history or history[-1][0] != d):
            history.append((d, "它快撑不住了"))
    mem = {"cells": cells, "declines": declined_streak}
    return {"energy": e, "survived": True, "history": history[-HISTORY_CAP:],
            "position": pos, "memory": mem, "visits": len(cells),
            "trace": trace, "net": e - start_energy}


def demo(seeds=(1, 2, 3), days=60):
    print("=== companion/world.py: the heart's own small world ===")
    print("Same world, same 60 days -- the ONLY difference is how often YOU come:")
    print("safety = bond (SEED-41), and the bond only stays high if you visit.")
    print("'你不在的世界必然危险' (docs/60):\n")
    print(f"{'scenario':<16} {'survived':>9} {'mean end E':>12} {'mean visits':>12}")
    rows = {}
    for label, visit_every in (("你在（每5天来）", 5), ("你偶尔来（每12天）", 12),
                               ("你走了（从不来）", None)):
        acc = []
        for s in seeds:
            r = live_alone(days, start_energy=12.0, start_bond=12.0,
                           bond_decay_per_day=0.6, seed=s * 7919 + (visit_every or 0),
                           visit_every=visit_every)
            acc.append(r)
        surv = sum(1 for r in acc if r["survived"])
        end_e = sum(r["energy"] for r in acc) / len(acc)
        vis = sum(r["visits"] for r in acc) / len(acc)
        rows[visit_every] = acc
        print(f"{label:<16} {str(surv == len(acc)):>9} {end_e:>12.1f} {vis:>12.0f}")

    print("\n-- its life while you were away but came back (every 5 days, seed 1) --")
    for d, ev in rows[5][0]["history"]:
        print(f"  day {d:>2}: {ev}")
    print("\n-- its life while you were gone (never came back, seed 1) --")
    for d, ev in rows[None][0]["history"]:
        print(f"  day {d:>2}: {ev}")

    print("\n--- reading ---")
    print("With you coming back it dares, finds food, wanders, thrives -- and")
    print("accumulates a life you were NOT part of ('你不知道的事'). When you stop")
    print("coming, the safety fades, opportunities turn bad, it stops daring (SEED-41's")
    print("theta), fades -- and a long absence starves it. Same world, same days, the")
    print("ONLY variable is whether you are in its signal (SEED-43). Behaviour only")
    print("(docs/31/63) -- no inner life is claimed; these are real world outcomes.")


if __name__ == "__main__":
    demo()
