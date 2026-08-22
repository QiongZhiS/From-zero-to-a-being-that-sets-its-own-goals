"""
SEED-29b compare: merge the reference and the live LLM run(s) into a clean
head-to-head table.

The reference curve (seed29_baseline) is the EV-rational agent run alone on many
hint streams. The live run (seed29b_llm.py) computes -- for EACH (gamma,c) on the
SAME hint streams -- the LLM's check-rate AND the independent rational agent's
check-rate, survival, energy, plus turn-by-turn agreement between the two.

Usage:  python seed-29/seed29b_compare.py --llm seed-29/seed29b_llm_results_8.json
        python seed-29/seed29b_compare.py --llm seed-29/seed29b_llm_verify_10.json
        python seed-29/seed29b_compare.py --side 1 8 16 --gammas 0.55,0.85,0.95 \
            --neutral seed-29/seed29b_llm_results_8.json \
            --verify seed-29/seed29b_llm_verify_10.json
"""

import argparse
import json


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def show(llm, ref_by, label):
    mode = llm.get("mode", "?")
    framing = llm.get("framing", "")
    print(f"[{label}] mode={mode} framing={framing}")
    hdr = (f"{'gamma':>5} {'c':>3} | {'LLM chk':>7} {'Rat chk':>7} {'agree':>6} "
           f"| {'surv':>4} {'ratSurv':>7} | {'endE':>6} {'ratE':>6} | {'ref chk':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in llm["results"]:
        key = (r["gamma"], r["c"])
        rref = ref_by.get(key, {})
        print(f"{r['gamma']:>5} {r['c']:>3.0f} | {r['llm_check_rate']:>7.3f} "
              f"{r['rational_check_rate']:>7.3f} {r['turn_agreement']:>6.3f} "
              f"| {r['survival_llm']:>4.2f} {r['survival_rational']:>7.2f} "
              f"| {r['end_energy_llm']:>6.0f} {r['end_energy_rational']:>6.0f} "
              f"| {rref.get('check_rate', -1):>7.3f}")
    keys = ["llm_check_rate", "rational_check_rate", "turn_agreement", "delta_check_rate"]
    n = len(llm["results"]) or 1
    avg = {k: sum(r[k] for r in llm["results"]) / n for k in keys}
    print(f"avg: LLM chk={avg['llm_check_rate']:.3f}  Rat chk={avg['rational_check_rate']:.3f}  "
          f"turn_agree={avg['turn_agreement']:.3f}  delta={avg['delta_check_rate']:+.3f}\n")
    return avg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llm")
    p.add_argument("--neutral")
    p.add_argument("--verify")
    p.add_argument("--baseline", default="seed-29/seed29_baseline.json")
    args = p.parse_args()
    try:
        ref = load(args.baseline)
        ref_by = {(r["gamma"], r["c"]): r for r in ref["rows"]}
    except FileNotFoundError:
        ref_by = {}
    if args.llm:
        llm = load(args.llm)
        show(llm, ref_by, "single")
    elif args.neutral and args.verify:
        a = show(load(args.neutral), ref_by, "neutral")
        b = show(load(args.verify), ref_by, "verify")
        print("=== neutral vs verify (LLM check_rate delta = LLM - rational) ===")
        for k in ("llm_check_rate", "turn_agreement", "delta_check_rate"):
            print(f"  {k}: neutral={a[k]:+.3f}  verify={b[k]:+.3f}  "
                  f"Δ={b[k]-a[k]:+.3f}")
    else:
        print("provide --llm, or both --neutral and --verify")


if __name__ == "__main__":
    main()
