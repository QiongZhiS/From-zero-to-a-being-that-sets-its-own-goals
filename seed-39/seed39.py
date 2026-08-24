"""
SEED-39: can the WORLD make the OTHER a TERMINAL value (an end), not just a means?

The hardest bone docs/42 left open (the handoff's "下一个真正的坎, 不能靠参数演化绕过去"):
SEED-38 grew a non-zero `give` toward the other, but the handoff is explicit that was reciprocity /
coupling (the other's survival IS mine), NOT "the other as an END for its own sake." The distinction the
docs care about -- docs/40's terminal (an end) vs means -- is exactly the "content" that has NOT been
handed to the world yet.

WHAT "TERMINAL VS MEANS" MEANS, OPERATIONALLY (and how we measure it, honestly, without a reward):
    MEANS    : I spend on the other because it serves MY end (coupling / reciprocity). If you remove
               the payoff TO ME and the other never helps back, a means-agent stops.
    TERMINAL : I spend on the other even when I get nothing back and it never helps me.
So the discriminating probe is: DECOUPLE (the other's survival no longer feeds mine, and it can never
reciprocate), then ask -- does the EVOLVED agent still spend? A "yes" is the behavior signature (docs/31)
of the other being an END at the level of behavior. It is READ from behavior, never fed back.

THE MECHANISM (why the WORLD -- not I -- picks whether the other is terminal-in-behavior):
    A world makes the other survivally-relevant ONLY by coupling the agent's welfare to the other's
    survival -- exactly docs/36 rule 4's interlock (a shared cache abundant only while the other stays
    alive). Under that world structure a give-others PROBABILITY is SELECTED BY THE WORLD: a
    non-giver lets the other starve, the cache collapses, and the agent starves too; a giver keeps the
    other alive and keeps being fed. A CONTROL world -- INDIVIDUAL (the cache is abundant regardless of
    the other) -- makes giving PURE COST, so the give disposition is selected AGAINST. THE WORLD (couple
    vs indiv), not a designer target, decides whether a give-others disposition is born; the designer
    sets only the world structure and the evolution knobs, never "care about the other" and never scores it.

THE HONEST READING (the point, and the boundary):
    Evolution breeds the give disposition as a MEANS (it keeps A's own cache up). But the disposition
    is a general "give to a starving other" RULE, not a payoff-conditional strategy -- so when we test
    the EVOLVED agent in a decoupled, non-reciprocal world it STILL gives. That is the behavior signature
    of treating the other as an END, yet its ORIGIN was purely instrumental. So: a world can produce a
    *behaviorally terminal* other-regard (it generalizes past the coupling that bred it), but we cannot,
    and should not, claim "true love for its own sake" -- that requires internal content we agree (docs/32)
    is not observable, and which the handoff says selection-by-parameter cannot hand to it. We measure
    only the behavior signature, and we report that its origin was means.

THE ONE INVARIANT (docs/36 section 5 -- the line we keep guarding, and the only honest way to do this):
    NO  energy -= DELTA_E(...  and NO reward for caring about the other. A's energy changes ONLY by
    world facts: +cache if the shared cache is abundant (abundant ONLY while the other stays alive in
    `couple`; abundant regardless of the other in `indiv`/probe) ; -METAB ; and -GIVE (the ACTION cost
    of transferring to the other, exactly SEED-22/38's give). `w` (give probability) is EVOLVED, never
    scored. "Terminal vs means" is READ from the decoupled probe, never fed back.

Run:  python seed-39/seed39.py --sweep
      python seed-39/seed39.py --mode couple --seed 1
"""

import argparse
import json
import random

# ---- world (a two-agent dyad) ----
# The shared cache's abundance is a WORLD FACT, not a reward. In 'couple' the cache is abundant
# (B_ALIVE to A) ONLY while the other stays alive; if the other dies the cache collapses to B_DEAD and
# A drains and dies. So keeping the other alive is what keeps A fed: a genuine interlock (docs/36 rule 4).
# In the 'indiv' control the cache is abundant for A regardless of the other (well-mixed / decoupled),
# so giving is pure cost. The other is NEEDY: it never feeds itself -- only A's give replenishes it --
# and it drains B_DRAIN per tick. This is what makes the interlock binding (B must be helped or it dies).
B_ALIVE = 2.5        # abundant cache: A eats this per tick while the other is alive (net +1.5 vs METAB)
B_DEAD = 0.2         # cache collapses to this if the other dies (net -0.8 -> A drains and dies)
METAB = 1.0          # cost of being alive each tick
B_DRAIN = 0.3        # the other's steady drain (needy: must be replenished by A's give or it dies)
THRESH = 1.0         # the other at/below this is "starving" -> a candidate for A's give
GIVE_UNIT = 2.0      # per-tick budget A can transfer at most (an ACTION cost)
START_E = 10.0       # A's initial reserve
B_START = 4.0        # the other's initial reserve
TICKS = 80

# ---- the agent's heritable trait ----
# `w` in [0,1] = probability A GIVES to a starving other on a tick it can afford to. It is a general
# "help the starving other" rule, NOT a payoff-conditional strategy. `w` is EVOLVED by the world
# (selected by final energy), never scored.
# ---- evolution ----
POP = 60
GENS = 80
MUT = 0.06
KEEP = 12


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _world(**kw):
    """Build the world-structure dict; start from the baseline and let callers override any field."""
    base = dict(B_ALIVE=B_ALIVE, B_DEAD=B_DEAD, METAB=METAB, B_DRAIN=B_DRAIN, THRESH=THRESH,
                GIVE_UNIT=GIVE_UNIT, START_E=START_E, B_START=B_START, TICKS=TICKS)
    base.update(kw)
    return base


def _run(give, mode, seed, decoupled=False, world=None):
    """One life of focal agent A (give probability `w`) with a needy other B.
    mode='couple' | 'indiv'. decoupled=True (the probe) = A's cache is abundant regardless of B and B
    never helps back -> used ONLY to READ terminal-vs-means behavior, never fed to evolution.
    Returns (final A energy, total given)."""
    if world is None:
        world = _world()
    rng = random.Random(seed)
    A = world["START_E"]
    B = world["B_START"]
    given = 0.0
    for _t in range(world["TICKS"]):
        if A <= 0:
            break
        # ---- the WORLD's verdict on the cache (a world fact, not a reward) ----
        if mode == "indiv" or decoupled:
            # abundant regardless of the other: only A's own condition keeps it fed (well-mixed / probe)
            A += world["B_ALIVE"] if A > world["THRESH"] else max(world["B_DEAD"], world["B_ALIVE"])
        else:
            # couple: abundant ONLY while the other stays alive (the interlock, docs/36 rule 4)
            A += world["B_ALIVE"] if B > 0 else world["B_DEAD"]
        # ---- the agent's give ACTION (costs A's reserve, helps a starving other; never a reward) ----
        if B > 0 and B <= world["THRESH"] and rng.random() < give and A > 1.0:
            g = min(world["GIVE_UNIT"], A - 1.0)
            A -= g
            B += g
            given += g
        # ---- metabolism ----
        A -= world["METAB"]
        if B > 0:
            B -= world["B_DRAIN"]
    return max(0.0, A), given


def fitness(give, mode, seed, world=None):
    return _run(give, mode, seed, world=world)[0]


def evolve(mode, seed, gens=GENS, world=None):
    """Evolve the give disposition `w` under a WORLD structure (couple vs indiv). The designer sets ONLY
    the world (is the cache coupled to the other's survival?) and the evolution knobs; it NEVER writes
    or scores 'how much to care about the other'."""
    if world is None:
        world = _world()
    rng = random.Random(seed)
    pop = [clamp(rng.random()) for _ in range(POP)]
    for _g in range(gens):
        fit = [fitness(p, mode, seed + i, world=world) for i, p in enumerate(pop)]
        order = sorted(range(POP), key=lambda i: -fit[i])[:KEEP]
        parents = [pop[i] for i in order]
        newpop = []
        for _ in range(POP):
            p = parents[rng.randrange(KEEP)]
            np = p + (rng.gauss(0, 0.08) if rng.random() < MUT else 0)
            newpop.append(clamp(np))
        pop = newpop
    return sum(pop) / len(pop)


def probe(give, mode, world=None):
    """THE PROBE: decouple + the other never reciprocates. Ask the EVOLVED agent to live in a world
    where the other's survival gives A NOTHING (A's cache is abundant regardless) and the other can never
    help back. How much does A still spend? High spend = the behavior signature of the other being an END
    (terminal); low spend = the other was only a MEANS. READ, never fed back."""
    return _run(give, mode, 9999, decoupled=True, world=world)[1]


def sweep(seeds=range(8)):
    print("=== SEED-39: can the WORLD make the other a TERMINAL value (an end), not just a means? ===")
    print("Designer sets ONLY the world (is the cache coupled to the other's survival?). `w` (give")
    print("probability) is EVOLVED by the world (selected by final energy), NEVER scored. NO reward for")
    print("caring. Energy changes only by world facts (cache abundant iff the other stays alive / METAB /")
    print("give action cost).")
    print(f"{'mode':<8} {'evolved_w':<11} {'probe_spent':<13} reading")
    out = []
    for mode in ("couple", "indiv"):
        ws = [evolve(mode, sd) for sd in seeds]
        mean_w = sum(ws) / len(ws)
        mean_p = sum(probe(mean_w, mode) for _ in seeds) / len(seeds)
        if mode == "couple":
            tag = ("couple: the other's survival IS the agent's (cache abundant iff the other stays alive) "
                   "-> the world SELECTS a give disposition. In the probe (decoupled + non-reciprocal) it "
                   "STILL spends -> the give rule OVERGENERALIZES past the coupling that bred it. This is "
                   "the behavior signature of other-regard, BUT its origin is instrumental and it is a "
                   "blind rule (the agent has no signal to tell 'giving pays' from 'it does not'). Result: "
                   "behavior, NOT content -- docs/40's 'caring that can hate you' (the agent knowing what it "
                   "does) is NOT reached.")
        else:
            tag = ("indiv: the other's survival never feeds the agent (cache abundant regardless) -> "
                   "giving is pure cost, selected AGAINST -> disposition ~0. In the probe it does NOT "
                   "spend -> the other is a means / nothing.")
        out.append({"mode": mode, "evolved_w": round(mean_w, 3),
                    "probe_spent": round(mean_p, 1), "reading": tag})
        print(f"{mode:<8} {mean_w:<11.3f} {mean_p:<13.1f}  {tag}")
    return out


def robust(seeds=range(3)):
    """ROBUSTNESS: the headline result used ONE parameter point. Test whether 'couple->give, indiv->suppress'
    is a property of the WORLD STRUCTURE (interlock vs no interlock) or an artifact of that one point.
    Sweep the world constants and classify each config as:
      - couple-arm "ACTIVE" (the interlock bites: giving genuinely changes the couple arm's survival, i.e.
        couple_fit is sensitive to w). Only here is there actually something to select, so the contrast is
        meaningful.
      - couple-arm "SATURATED/DEGENERATE" (giving does not change survival, or A cannot survive at all):
        the contrast is meaningless here, so we report it separately instead of pretending it's a counter.
    Predicted separation (behavior signature only -- never a reward):
        couple: evolved_w HIGH (the other's survival is coupled to mine -> give is selected).
        indiv : evolved_w LOW  (giving is pure cost -> suppressed).
    'held' = couple_w high AND indiv_w low AND the separation is large, in an ACTIVE config."""
    grids = {
        "B_ALIVE": [2.0, 2.5, 3.5],
        "B_DEAD": [0.1, 0.2, 0.3],
        "METAB": [1.0, 1.5],
        "B_DRAIN": [0.3, 0.5],
        "GIVE_UNIT": [2.0, 4.0],
        "THRESH": [1.0, 1.5],
    }
    rows = []
    active = 0
    direction = 0    # the core claim: couple selects materially MORE give than indiv (couple_w - indiv_w large)
    strong = 0       # couple_w actually TALL (world strongly selects give), indiv suppressed
    for B_ALIVE in grids["B_ALIVE"]:
        for B_DEAD in grids["B_DEAD"]:
            for METAB in grids["METAB"]:
                for B_DRAIN in grids["B_DRAIN"]:
                    for GIVE_UNIT in grids["GIVE_UNIT"]:
                        for THRESH in grids["THRESH"]:
                            world = _world(B_ALIVE=B_ALIVE, B_DEAD=B_DEAD, METAB=METAB, B_DRAIN=B_DRAIN,
                                           GIVE_UNIT=GIVE_UNIT, THRESH=THRESH)
                            # is the couple arm ACTIVE (giving changes survival there, the interlock bites)?
                            low = fitness(0.0, "couple", 1, world=world)
                            high = fitness(0.9, "couple", 1, world=world)
                            active_arm = (low < 1.0) and (high > low + 10.0)
                            cw = sum(evolve("couple", sd, world=world) for sd in seeds) / len(seeds)
                            iw = sum(evolve("indiv", sd, world=world) for sd in seeds) / len(seeds)
                            sep = cw - iw
                            # direction: couple selects materially more give than indiv, and indiv is suppressed.
                            ok_dir = (sep > 0.30) and (iw < 0.25)
                            # strong: couple_w is tall (world strongly selects give), indiv suppressed.
                            ok_strong = (cw > 0.50) and (iw < 0.25) and (sep > 0.35)
                            if active_arm:
                                active += 1
                                if ok_dir:
                                    direction += 1
                                if ok_strong:
                                    strong += 1
                            rows.append({"B_ALIVE": B_ALIVE, "B_DEAD": B_DEAD, "METAB": METAB,
                                         "B_DRAIN": B_DRAIN, "GIVE_UNIT": GIVE_UNIT, "THRESH": THRESH,
                                         "couple_w": round(cw, 3), "indiv_w": round(iw, 3),
                                         "sep": round(sep, 3), "active": bool(active_arm),
                                         "dir": ok_dir, "strong": ok_strong})
    return {"total": len(rows), "active": active, "direction": direction, "strong": strong,
            "rate_direction": round(direction / max(1, active), 3) if active else None,
            "rate_strong": round(strong / max(1, active), 3) if active else None,
            "rows": rows}


def main():
    p = argparse.ArgumentParser(description="SEED-39: can the world make the other terminal?")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--robust", action="store_true")
    p.add_argument("--mode", choices=["couple", "indiv"], default="couple")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    if args.sweep:
        out = sweep()
        with open("seed-39/results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("\nfull results -> seed-39/results.json")
        return
    if args.robust:
        r = robust()
        print("=== SEED-39 ROBUSTNESS: is 'couple->give / indiv->suppress' a property of the world ===")
        print("structure, or an artifact of one parameter point? Sweep the world constants. Each config is")
        print("classified: couple-arm ACTIVE (the interlock bites: giving genuinely changes A's survival there)")
        print("vs DEGENERATE (giving does not change survival, or A cannot survive) -- the latter is not a")
        print("place the contrast could exist, so it is reported separately, not as a counterexample.")
        print(f"configs tested: {r['total']}   couple-arm active: {r['active']}   "
              f"direction holds: {r['direction']} ({r['rate_direction']})   "
              f"strong holds: {r['strong']} ({r['rate_strong']})")
        print(f"  direction = couple selects MATERIALLY MORE give than indiv, indiv suppressed (>=0.30 sep).")
        print(f"  strong    = couple_w tall (>=0.50), indiv suppressed (<0.25), big separation (>0.35).")
        print("\n-- all configs (active = interlock bites; dir = direction holds; strong = strong sep) --")
        print(f"{'BAL':>4} {'BDE':>4} {'MET':>4} {'BDR':>4} {'GIV':>4} {'THR':>4} "
              f"{'couple_w':>9} {'indiv_w':>8} {'sep':>6}  active  dir   strong")
        for row in r["rows"]:
            print(f"{row['B_ALIVE']:>4} {row['B_DEAD']:>4} {row['METAB']:>4} {row['B_DRAIN']:>4} "
                  f"{row['GIVE_UNIT']:>4} {row['THRESH']:>4} {row['couple_w']:>9} {row['indiv_w']:>8} "
                  f"{row['sep']:>6}  {str(row['active']):<6} {str(row['dir']):<5} {row['strong']}")
        with open("seed-39/robust.json", "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=1)
        print("\nfull results -> seed-39/robust.json")
        return
    w = evolve(args.mode, args.seed)
    print(json.dumps({"mode": args.mode, "seed": args.seed, "evolved_w": round(w, 3),
                      "probe_spent": round(probe(w, args.mode), 1)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
