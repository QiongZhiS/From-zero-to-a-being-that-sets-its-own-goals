"""
SEED-28: two-agent mutualism game -- does real reciprocity emerge and stay stable?

docs/19/22 + SEED-22: relationship as COUPLING (both free, mutually beneficial =
real mutualism) vs PARASITISM/COERCION (one exploits, or a coerced goal). SEED-22
used a SCRIPTED partner B; here BOTH agents are ACTIVE and can DEFECT (parasitize)
or COOPERATE (mutualism) -- an iterated Prisoner's Dilemma with survival pressure.

docs/12 P20: signal quality is the ceiling. The natural probe: an ERROR rate eps
in reading the partner's last move. Reciprocal cooperation (TFT-like) is exactly
the kind of thing P20 predicts is FRAGILE to low signal quality -- high eps should
destabilize cooperation and collapse it into defection (parasitism).

Payoff per round (energy delta, net of metab):
  CC: both +M   (mutualism, win-win)
  CD: cooperator -G, defector +B   (defector exploits; cooperator bleeds)
  DD: both 0    (baseline, both slowly starve at -1/tick)
Energy starts at START_E; death at 0. Fitness = survival & final energy over R
rounds. Agents are paired and play the SAME partner for R rounds (so reciprocity
is possible). Strategy = memory-one: f(my_last, partner_last) -> {C,D}, a 4-bit
code covering TFT / ALL-C / ALL-D / GRIM / WSLS etc. Co-evolved.

Sweep: payoff generosity (M, B) x error eps. Predict:
  - low eps + repeated interaction  -> reciprocal CC emerges and is stable (real
    mutualism, both benefit).
  - high eps                        -> misreads trigger defection spirals, cooperation
    collapses to defection/parasitism (P20 signal-quality ceiling on reciprocity).

Run:  python seed-28/seed28.py --sweep
      python seed-28/seed28.py --eps 0.3
"""

import argparse
import json
import random

M = 6.0            # mutual cooperation payoff (each)
G = 4.0            # loss for a cooperator who is exploited
B = 8.0            # gain for a defector who exploits
METAB = 1.0
START_E = 50.0
ROUNDS = 40
POP = 80
GENS = 60
MUT = 0.15
KEEP = 16


def act(strategy_code, my_last, partner_last):
    """memory-one: bit index = (my_last, partner_last); 1 = cooperate."""
    idx = (1 if my_last == "C" else 0) * 2 + (1 if partner_last == "C" else 0)
    return "C" if (strategy_code >> idx) & 1 else "D"


def play(sonly, code_a, code_b, eps, seed, rounds=ROUNDS):
    """Play one pairing (2 agents) for R rounds; return both agents' final energy."""
    rng = random.Random(seed)
    ea = START_E
    eb = START_E
    la = "C"
    lb = "C"
    for _ in range(rounds):
        # observe partner's last action, with noise (signal quality / P20)
        ob_a = lb
        ob_b = la
        if rng.random() < eps:
            ob_a = "D" if lb == "C" else "C"
        if rng.random() < eps:
            ob_b = "D" if la == "C" else "C"
        a = act(code_a, la, ob_a)
        b = act(code_b, lb, ob_b)
        # payoffs
        if a == "C" and b == "C":
            ea += M - METAB
            eb += M - METAB
        elif a == "C" and b == "D":
            ea += -G - METAB
            eb += B - METAB
        elif a == "D" and b == "C":
            ea += B - METAB
            eb += -G - METAB
        else:  # DD
            ea += -METAB
            eb += -METAB
        la, lb = a, b
        if ea <= 0:
            ea = 0.0
        if eb <= 0:
            eb = 0.0
    return ea, eb


def evolve(eps, seed, gens=GENS, rounds=ROUNDS):
    rng = random.Random(seed)
    pop = [rng.randrange(16) for _ in range(POP)]   # memory-one codes (0..15)
    for _g in range(gens):
        # random pairing, half the population plays one game
        fit = [0.0] * POP
        order = list(range(POP))
        rng.shuffle(order)
        for i in range(0, POP - 1, 2):
            a, b = order[i], order[i + 1]
            ea, eb = play(rounds, pop[a], pop[b], eps, seed + i)
            fit[a] = ea
            fit[b] = eb
        # select top KEEP, reproduce with bit mutation
        best = sorted(range(POP), key=lambda i: -fit[i])[:KEEP]
        parents = [pop[i] for i in best]
        newpop = []
        for _ in range(POP):
            s = parents[rng.randrange(KEEP)]
            if rng.random() < MUT:
                s = s ^ (1 << rng.randrange(4))
            newpop.append(s)
        pop = newpop
    return pop


def cooperation_rate(eps, seed, samples=40, rounds=ROUNDS):
    """Evolve, then measure fraction of CC rounds among top strategies."""
    pop = evolve(eps, seed)
    rng = random.Random(seed + 999)
    top = pop[:KEEP]
    cc = 0
    n = 0
    for i in range(samples):
        a = top[rng.randrange(KEEP)]
        b = top[rng.randrange(KEEP)]
        la = lb = "C"
        for _ in range(rounds):
            ob_a = lb
            ob_b = la
            if rng.random() < eps:
                ob_a = "D" if lb == "C" else "C"
            if rng.random() < eps:
                ob_b = "D" if la == "C" else "C"
            aa = act(a, la, ob_a)
            bb = act(b, lb, ob_b)
            if aa == "C" and bb == "C":
                cc += 1
            n += 1
            la, lb = aa, bb
    return cc / n if n else 0.0


def sweep(seeds=range(5), eps_list=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5), m_b=(6.0, 8.0)):
    global M, B
    print("=== SEED-28: two-agent mutualism -- does reciprocity emerge, and does signal quality (eps) break it? ===")
    print(f"(payoff M={M} CC, B={B} exploit; eps = error reading partner's move = signal-quality knob P20)")
    print(f"{'eps':<6} {'coop_rate':<10}")
    results = []
    for eps in eps_list:
        rates = [cooperation_rate(eps, s) for s in seeds]
        avg = sum(rates) / len(rates)
        results.append({"eps": eps, "coop_rate": round(avg, 3)})
        print(f"{eps:<6.2f} {avg:<10.3f}")
    return results


def main():
    global M, B
    p = argparse.ArgumentParser(description="SEED-28 two-agent mutualism")
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--seeds", type=int, default=5)
    args = p.parse_args()
    if args.sweep:
        out = sweep(seeds=range(1, args.seeds + 1))
        with open("seed-28/results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("\nfull results -> seed-28/results.json")
        return
    r = cooperation_rate(args.eps, 1)
    print(json.dumps({"eps": args.eps, "coop_rate": round(r, 3)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
