"""
SEED-21 v2: LLM-agent society -- pollution lock without a verification layer

Third phase extension (docs/14, docs/16). P16 says culture without a
verification layer locks onto a wrong consensus (NO RECOVERY, while genes
are immune); P19 says a verification bridge prevents fatal contamination
BUT is itself limited by signal quality.

Here "culture" is a society of agents that copy the strategy of whoever
LOOKS most successful (reputation / displayed score), like LLM agents
copying "the SOTA" from leaderboards, viral claims or benchmark hype:

  strategy A: true mean 10.0 (optimal)
  strategy B: true mean  8.0 (suboptimal)
  noise sigma: the world's signal quality (luck component in success)

Signals:
  actual    -- real payoff an agent earns (mu_k + noise)
  displayed -- smoothed recent score the society sees. During the FRAUD
               window, EVERY B-user's displayed score is inflated by +fake
               (a hype campaign / fake leaderboard that makes the whole
               wrong camp look successful). After the window, honest.

Regimes:
  genetic     no imitation; each agent explores alone (immune control)
  cultural-nv imitate whoever displayed the highest score (gullible)
  cultural-v  imitate only after personally testing the candidate
              (verification bridge: do -> observe -> update, SEED-6)

Questions:
  Q1 (P16): does fraud capture the society onto B and stay locked after
             exposure (NO RECOVERY) in cultural-nv? Explore=0 society:
             permanent lock. With exploration trickle: slow recovery.
  Q2 (P19): does cultural-v reject the fraud? Does protection degrade
             as noise grows?
  Q3 (P15): can pure LUCK (no fraud) lock the society onto B at high
             noise, by making lucky B-users the sampled leaders?

Run:  python seed21.py --sweep
      python seed21.py --mode cultural-nv --noise 3 --fraud 1 --ticks 4000
"""

import argparse
import json
import random
from dataclasses import dataclass, field

MU = {"A": 10.0, "B": 8.0}


@dataclass
class Agent:
    strategy: str = "A"
    actual: float = 0.0              # lifetime REAL payoff (welfare)
    window: list = field(default_factory=list)   # recent actual payoffs
    testing: int = 0                 # ticks left in a verification test
    test_strategy: str = ""
    test_sum: float = 0.0
    tests: int = 0                   # completed verification tests
    adopted_by_test: int = 0         # adoptions that came from a test


def make_society(n, rng):
    return [Agent(strategy=rng.choice(["A", "B"])) for _ in range(n)]


def payoff(rng, strategy, noise):
    return MU[strategy] + rng.gauss(0.0, noise)


def displayed(agent, window_len, fraud_active, fake):
    """Society sees a smoothed recent score. During fraud, EVERY B-user's
    score is inflated (the false claim travels with the strategy)."""
    if not agent.window:
        return 0.0
    bonus = fake if (fraud_active and agent.strategy == "B") else 0.0
    return sum(agent.window) / len(agent.window) + bonus


def run(ticks, mode, noise, fraud, seed, n=200, window=15,
        p_imit=0.05, p_explore=0.02, model_sample=5,
        t_inj=1000, t_rem=2500, invader_frac=0.15, invader_n=None,
        fake=5.0, report_every=500, sample_mode="global"):
    rng = random.Random(seed)
    agents = make_society(n, rng)
    a_share = []
    welfare_ticks = []

    for t in range(ticks):
        fraud_active = fraud and (t_inj <= t < t_rem)
        if fraud and t == t_inj:
            # seed the wrong consensus: actually convert agents to B (SEED-18
            # invasion); during the window every B-user's score is inflated
            k = invader_n if invader_n else int(n * invader_frac)
            for a in rng.sample(agents, min(k, len(agents))):
                a.strategy = "B"

        tick_welfare = 0.0
        for a in agents:
            if a.testing > 0:
                # verification: test the candidate on REAL payoffs
                a.test_sum += payoff(rng, a.test_strategy, noise)
                a.testing -= 1
                if a.testing == 0:
                    cand_avg = a.test_sum / window
                    own_avg = (sum(a.window) / len(a.window)
                               if a.window else cand_avg)
                    a.tests += 1
                    if cand_avg > own_avg:
                        a.strategy = a.test_strategy
                        a.adopted_by_test += 1
                    a.test_sum = 0.0
                continue

            # exploitation: earn
            p = payoff(rng, a.strategy, noise)
            a.actual += p
            a.window.append(p)
            if len(a.window) > window:
                a.window.pop(0)
            tick_welfare += p

            # exploration: curiosity trickle (keeps the alternative alive)
            if rng.random() < p_explore:
                a.strategy = "B" if a.strategy == "A" else "A"
                continue

            # imitation: copy whoever LOOKS most successful
            if mode != "genetic" and rng.random() < p_imit:
                if sample_mode == "local":
                    # gossip: sample a FIXED number of OTHER agents --
                    # a fraud is visible with prob ~ sample_n/N, which
                    # DILUTES with society size (scale-sensitive!)
                    others = [x for x in agents if x is not a]
                    sample = rng.sample(others, min(model_sample,
                                                    len(others)))
                else:
                    sample = rng.sample(agents, min(model_sample,
                                                    len(agents)))
                leader = max(sample,
                             key=lambda x: displayed(x, window,
                                                     fraud_active, fake))
                if leader.strategy != a.strategy:
                    if mode == "cultural-nv":
                        a.strategy = leader.strategy          # gullible copy
                    else:  # cultural-v: verify before adopting
                        a.testing = window
                        a.test_strategy = leader.strategy
                        a.test_sum = 0.0

        welfare_ticks.append(tick_welfare / n)

        if (t + 1) % report_every == 0:
            sh = sum(1 for a in agents if a.strategy == "A") / n
            a_share.append((t + 1, sh))

    # metrics
    end_a = sum(1 for a in agents if a.strategy == "A") / n
    during = [s for (tt, s) in a_share if t_inj <= tt < t_rem]
    post = [s for (tt, s) in a_share if tt >= t_rem]
    during_a = sum(during) / len(during) if during else end_a
    post_a = sum(post) / len(post) if post else end_a
    # recovery: first report tick AFTER exposure with A-share >= 0.5
    recovery = next((tt for (tt, s) in a_share
                     if tt >= t_rem and s >= 0.5), None)
    mean_w = sum(welfare_ticks) / len(welfare_ticks)
    tests = sum(a.tests for a in agents)
    adopted = sum(a.adopted_by_test for a in agents)

    return {
        "mode": mode, "noise": noise, "fraud": bool(fraud), "seed": seed,
        "explore": p_explore,
        "end_a": round(end_a, 3), "during_a": round(during_a, 3),
        "post_a": round(post_a, 3), "recovery": recovery,
        "mean_welfare": round(mean_w, 3),
        "tests": tests, "adopted_by_test": adopted,
        "traj": [(t, round(s, 3)) for t, s in a_share],
    }


def sweep(ticks=4000, seeds=(1, 2, 3, 4, 5, 6), noises=(0.5, 3.0, 10.0, 25.0),
          frauds=(0, 1), modes=("genetic", "cultural-nv", "cultural-v"),
          explore_rows=True):
    results = []
    for mode in modes:
        for noise in noises:
            for fraud in frauds:
                for seed in seeds:
                    results.append(run(ticks=ticks, mode=mode, noise=noise,
                                       fraud=fraud, seed=seed))
    # P16 lock demo: pure cultural society (explore=0) under fraud
    if explore_rows:
        for mode in ("cultural-nv", "cultural-v"):
            for seed in seeds:
                results.append(run(ticks=ticks, mode=mode, noise=3.0,
                                   fraud=1, seed=seed, p_explore=0.0))

    # summary table
    print("=== SEED-21 v2 sweep summary (mean over seeds) ===")
    print(f"{'mode':<12} {'noise':<6} {'fraud':<6} {'expl':<6} "
          f"{'end_A':<7} {'dur_A':<7} {'post_A':<7} {'welf':<7} "
          f"{'lock%':<6} {'rec%':<6}")
    for mode in modes:
        for noise in noises:
            for fraud in frauds:
                rs = [r for r in results
                      if r["mode"] == mode and r["noise"] == noise
                      and r["fraud"] == bool(fraud) and r["explore"] == 0.02]
                if not rs:
                    continue
                end = sum(r["end_a"] for r in rs) / len(rs)
                dur = sum(r["during_a"] for r in rs) / len(rs)
                post = sum(r["post_a"] for r in rs) / len(rs)
                welf = sum(r["mean_welfare"] for r in rs) / len(rs)
                lock = sum(1 for r in rs if r["end_a"] < 0.25) / len(rs)
                rec = sum(1 for r in rs if r["recovery"] is not None) / len(rs)
                print(f"{mode:<12} {noise:<6.1f} {int(fraud):<6} {0.02:<6} "
                      f"{end:<7.3f} {dur:<7.3f} {post:<7.3f} {welf:<7.3f} "
                      f"{lock:<6.2f} {rec:<6.2f}")
        if explore_rows and mode in ("cultural-nv", "cultural-v"):
            rs = [r for r in results
                  if r["mode"] == mode and r["noise"] == 3.0
                  and r["fraud"] and r["explore"] == 0.0]
            end = sum(r["end_a"] for r in rs) / len(rs)
            dur = sum(r["during_a"] for r in rs) / len(rs)
            post = sum(r["post_a"] for r in rs) / len(rs)
            welf = sum(r["mean_welfare"] for r in rs) / len(rs)
            lock = sum(1 for r in rs if r["end_a"] < 0.25) / len(rs)
            rec = sum(1 for r in rs if r["recovery"] is not None) / len(rs)
            print(f"{mode:<12} {3.0:<6.1f} {1:<6} {0.0:<6} "
                  f"{end:<7.3f} {dur:<7.3f} {post:<7.3f} {welf:<7.3f} "
                  f"{lock:<6.2f} {rec:<6.2f}   <- explore=0 (pure culture)")
    with open("seed-21/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("\nfull results -> seed-21/results.json")
    return results


def scale_sweep(ticks=4000, seeds=range(60), ns=(50, 200, 1000),
                mode="cultural-nv", noise=3.0, fraud=1, p_explore=0.02,
                invader_n=None):
    """Route B probe: does pollution capture/recovery depend on society
    size? Scenario A (proportional invaders, 15%): scale-invariant.
    Scenario B (single planted fraud, like the LLM version): visibility
    ~sample_n/N dilutes with size -- the scale-sensitive candidate."""
    tag = f"invader_n={invader_n}" if invader_n else "invader_frac=15%"
    print(f"=== SEED-21 scale sweep ({tag}, {mode}, noise={noise}, "
          f"fraud={fraud}, explore={p_explore}) ===")
    print(f"{'sample':<8} {'N':<6} {'dur_A':<7} {'post_A':<7} {'end_A':<7} "
          f"{'lock%':<6} {'rec%':<6} {'recT':<6}")
    for sm in ("global", "local"):
        for n in ns:
            ends, durs, posts, locks, recs, rects = [], [], [], 0, 0, []
            for s in seeds:
                r = run(ticks=ticks, mode=mode, noise=noise, fraud=fraud,
                        seed=s, n=n, p_explore=p_explore, sample_mode=sm,
                        invader_n=invader_n)
                ends.append(r["end_a"]); durs.append(r["during_a"])
                posts.append(r["post_a"])
                if r["end_a"] < 0.25:
                    locks += 1
                if r["recovery"] is not None:
                    recs += 1
                    rects.append(r["recovery"])
            cnt = len(seeds)
            rect = (sum(rects) / len(rects)) if rects else float("nan")
            print(f"{sm:<8} {n:<6} {sum(durs)/cnt:<7.3f} {sum(posts)/cnt:<7.3f} "
                  f"{sum(ends)/cnt:<7.3f} {locks/cnt:<6.2f} {recs/cnt:<6.2f} "
                  f"{rect:<6.0f}")
    return None


def main():
    p = argparse.ArgumentParser(description="SEED-21 pollution lock society")
    p.add_argument("--mode", choices=["genetic", "cultural-nv", "cultural-v"],
                   default="cultural-nv")
    p.add_argument("--noise", type=float, default=3.0)
    p.add_argument("--fraud", type=int, default=1)
    p.add_argument("--ticks", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window", type=int, default=15)
    p.add_argument("--explore", type=float, default=0.02)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--scale-sweep", action="store_true")
    p.add_argument("--scale-n", type=int, default=60,
                   help="seeds per scale point")
    p.add_argument("--invader-n", type=int, default=None,
                   help="fixed invader count (None = 15% proportional)")
    args = p.parse_args()
    if args.scale_sweep:
        scale_sweep(ticks=args.ticks, seeds=range(args.scale_n),
                    invader_n=args.invader_n)
        return
    if args.sweep:
        sweep(ticks=args.ticks)
        return
    r = run(ticks=args.ticks, mode=args.mode, noise=args.noise,
            fraud=args.fraud, seed=args.seed, window=args.window,
            p_explore=args.explore)
    print(json.dumps(r, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
