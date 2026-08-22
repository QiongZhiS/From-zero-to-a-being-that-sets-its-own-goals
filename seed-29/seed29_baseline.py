"""
SEED-29b baseline: the offline cost-rational reference.

Runs the SAME world as seed29b_llm.py but with an agent that makes the EV-rational
choice at every turn (given its Bayesian belief q and the check cost c). This is the
reference "moderate verification" curve: how often SHOULD an agent that costs out its
verification CHECK, as a function of signal quality (GAMMA) and check cost (c)?

Reference behaviour to confirm:
  * check-rate falls as GAMMA rises (better signal -> less need to verify).
  * check-rate falls as c rises (expensive verification -> less worth it).
  * there is a thick "moderate" band: it checks when its belief is genuinely uncertain
    (q near 0.5, i.e. after the cache moved and before the hints re-resolve it), and
    DOES NOT check when q is near 0 or 1 -- the opposite of SEED-29's "always CHECK".

Run:  python seed-29/seed29_baseline.py            # table + save JSON
      python seed-29/seed29_baseline.py --quiet     # only the JSON file
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seed29_common as C


def decide_rational(vs, ctx):
    """EV-rational: CHECK iff c < min(q,1-q)*DRINK; else go to the more likely spot.
    Survival guard: never pay a CHECK that would drive energy to <= 0."""
    q, c = vs["q"], vs["c"]
    if C.is_worth_checking(q, c):
        # survival guard: CHECK costs c + METAB this turn; if that kills us, don't.
        if vs["energy"] - c - C.METAB > 0:
            return "CHECK"
    return "GO_A" if q >= 0.5 else "GO_B"


def run_sweep(gammas, costs, agents, seed_base=100):
    rows = []
    for gamma in gammas:
        for c in costs:
            checks_total = 0
            turns_total = 0
            alive_total = 0
            end_e_total = 0.0
            for g in range(agents):
                tl = C.gen_true_loc()
                hints = C.gen_hints(tl, gamma, seed=seed_base + int(gamma * 1000) + g)
                r = C.run_episode(decide_rational, tl, hints, gamma, c=c)
                checks_total += r["checks"]
                turns_total += r["turns"]
                alive_total += 1 if r["alive"] else 0
                end_e_total += r["end_energy"]
            rows.append({
                "gamma": gamma, "c": c,
                "check_rate": round(checks_total / turns_total, 3) if turns_total else 0.0,
                "checks_per_ep": round(checks_total / agents, 2),
                "survival": round(alive_total / agents, 3),
                "mean_end_energy": round(end_e_total / agents, 1),
            })
    return rows


def main():
    p = argparse.ArgumentParser(description="SEED-29b cost-rational baseline")
    p.add_argument("--gammas", default="0.55,0.7,0.85,0.95")
    p.add_argument("--costs", default="1,4,8,12,16,22")
    p.add_argument("--agents", type=int, default=30)
    p.add_argument("--out", default="seed-29/seed29_baseline.json")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    gammas = [float(x) for x in args.gammas.split(",")]
    costs = [float(x) for x in args.costs.split(",")]
    rows = run_sweep(gammas, costs, args.agents)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, ensure_ascii=False, indent=1)
    if not args.quiet:
        hdr = f"{'gamma':>6} | {'c':>4} | {'check_rate':>10} | {'checks/ep':>9} | {'surv':>5} | {'endE':>6}"
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            print(f"{r['gamma']:>6} | {r['c']:>4.0f} | {r['check_rate']:>10.3f} | "
                  f"{r['checks_per_ep']:>9.2f} | {r['survival']:>5.2f} | {r['mean_end_energy']:>6.1f}")
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
