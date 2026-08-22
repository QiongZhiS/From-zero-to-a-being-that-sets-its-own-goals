"""
SEED-25: self-referential self-model LOCK.

docs/25 section 5.2: "self = a compressed model of 'who I am', shaped by the
other's reflection, that can close into a stubborn self-referential illusion."
Mechanically that is SEED-21's NO RECOVERY lock, but the thing being locked is
the SELF-NARRATIVE instead of a social consensus.

The self-reference is the crux: the self-model filters evidence by its OWN
prediction. It only accepts evidence that confirms "who I am"; contradictions
are discounted / rationalized away. So it can only see evidence that agrees
with itself -> it ratchets into whatever narrative it first settled on, and
resists correction. This is EXACTLY confirmation bias applied to a self-model.

  world:  true self/world parameter theta* = 0.7
  obs:    each tick o ~ N(theta*, sigma)
  belief: theta_t (the compressed self-model / "who I am")
  gate:   accept o into the update if it CONFIRMS the belief (|o - theta_t| < tau)
          OR if a verification roll succeeds (prob nu).
          Otherwise REJECT (keep theta_t) -- the contradiction is rationalized away.
  update: theta += eta * (o - theta)  when accepted; else stay.

  nu is the VERIFICATION / INDEPENDENT-EXPLORATION knob (the anti-lock bridge,
  SEED-6 do->observe->update / P19).
    nu = 0   -> pure self-referential filter: NO RECOVERY lock onto a wrong self.
    nu -> 1  -> accept everything (non-referential): self-corrects to the truth.

This mirrors SEED-21 exactly: culture lock (no verify + no explore) vs the
explore/verify trickle that is the antidote. Here the solo agent's self-model
is the "culture", and its own belief is the "consensus".

Run:  python seed-25/seed25.py --sweep
      python seed-25/seed25.py --nu 0 --theta0 0.15
"""

import argparse
import json
import random

THETA_STAR = 0.70
SIGMA = 0.15
TAU = 0.12           # confirmation tolerance (how far an obs can be and still count)
ETA = 0.20           # learning rate
TICKS = 300


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def run_episode(theta0, nu, seed, ticks=TICKS):
    rng = random.Random(seed)
    theta = theta0
    for _ in range(ticks):
        o = rng.gauss(THETA_STAR, SIGMA)
        confirms = abs(o - theta) < TAU
        if confirms or rng.random() < nu:
            theta = clamp(theta + ETA * (o - theta))
    return theta


def summarize(theta0, nu, seeds, ticks=TICKS, lock_thresh=0.30):
    """Mean final |theta - theta*| and lock rate (final error > threshold)."""
    errs = []
    for s in seeds:
        theta = run_episode(theta0, nu, s, ticks)
        errs.append(abs(theta - THETA_STAR))
    mean_err = sum(errs) / len(errs)
    lock = sum(1 for e in errs if e > lock_thresh) / len(errs)
    return {"nu": nu, "theta0": theta0, "mean_error": round(mean_err, 3),
            "lock_rate": round(lock, 3)}


def sweep(seeds=range(200), nu_list=(0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
          theta0_list=(0.15, 0.65)):
    print("=== SEED-25: self-referential self-model lock ===")
    print("(mean final |theta - theta*| ; lock_rate = frac final error > 0.30)")
    results = []
    for theta0 in theta0_list:
        print(f"\n-- seed belief theta0 = {theta0}  (theta* = {THETA_STAR}) --")
        print(f"{'nu':<6} {'meanErr':<9} {'lock%':<7}")
        for nu in nu_list:
            r = summarize(theta0, nu, list(seeds))
            results.append(r)
            print(f"{nu:<6.3f} {r['mean_error']:<9.3f} {r['lock_rate']:<7.3f}")
    return results


def main():
    p = argparse.ArgumentParser(description="SEED-25 self-referential self-model lock")
    p.add_argument("--nu", type=float, default=0.0)
    p.add_argument("--theta0", type=float, default=0.15)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--seeds", type=int, default=200)
    args = p.parse_args()
    if args.sweep:
        out = sweep(seeds=range(1, args.seeds + 1))
        with open("seed-25/results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("\nfull results -> seed-25/results.json")
        return
    r = summarize(args.theta0, args.nu, range(1, args.seeds + 1))
    print(json.dumps(r, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
