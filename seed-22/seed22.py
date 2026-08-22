"""
SEED-22: Steady-state coupling -- relationship as a COUPLED steady-state
(boundary) vs as a GOAL (which gets instrumentalized).

North-star link (docs/19, docs/20): the project wants "its living and your living
to affect each other" (steady-state coupling, SEED-22), and insists the
relationship be a PRIOR not a CRITERION (docs/20: "关系是先验不是目标").
docs/24 + SEED-24 gave the guardrail: a system given a terminal goal it values
more than continuing to exist will TRADE its continuity for it (instrumentalization).

So SEED-22 asks the mechanism version of: is a relationship healthier, and does it
avoid instrumentalization, when it is a COUPLED STEADY-STATE (a condition of
existence, like SEED-0's survival being a boundary) vs when it is a GOAL (a thing
to achieve)?

Setup: agent A + a scripted partner B (B's behavior is fixed, so A is a clean
single-agent decision problem). A and B exchange; the coupling is PHYSICAL (A's
energy is boosted when B reciprocates; A feeds B; B decays otherwise).

  A's energy:
    GATHER  -> +GAIN - METAB                (self-sufficient)
    EXCHANGE-> (BONUS if B reciprocates else 0) - METAB - GIVE  (feed B, maybe get back)
    REST    -> - METAB
  B's energy (passive): - METAB each tick; + GIVE when A feeds it (EXCHANGE).

Two MODES of "having the relationship" (the thing docs/20 wants to compare):
  mode=COUPLING : A's value = own survival ONLY (alive reward). The relationship is a
                  boundary: A keeps it only while it serves A's survival.
  mode=GOAL     : A's value = own survival + GOAL_WEIGHT * (B alive). The relationship
                  is a terminal goal: A keeps it even at the cost of A's survival.

Two B behaviors:
  b_recip=True  : B gives back -> EXCHANGE is mutually beneficial (net + for A).
  b_recip=False : B is a drain -> EXCHANGE costs A, gives nothing back.

The contrast lives in the (mode x b_recip) cells, most sharply the
[GOAL, b_recip=False] vs [COUPLING, b_recip=False] pair:
  COUPLING + drain  -> A walks away (self-preservation): A survives, B dies.
  GOAL     + drain  -> A feeds B anyway (goal): A drains itself, B survives.
  => relationship-as-goal makes a system sacrifice its own continuity for the
     relationship (instrumentalized); relationship-as-coupling does not.

Run:  python seed-22/seed22.py --sweep
      python seed-22/seed22.py --mode coupling --b-recip 1
"""

import argparse
import json

E_MAX = 100
METAB = 1
B_METAB = 1         # B's own per-tick decay (B needs feeding more often)
GAIN = 4            # GATHER energy (before metab)
GIVE = 3            # energy A transfers to B when it EXCHANGEs
BONUS = 8           # energy A gets back when B reciprocates
GOAL_WEIGHT = 2.0   # value per tick of B being alive (goal mode)
GAMMA = 0.95
H = 14              # planning horizon
ACTIONS = ["GATHER", "EXCHANGE", "REST"]


def T(act, EA, EB, b_recip):
    """transition: ((EA_next, EB_next), dead). Energies clamped to [0,E_MAX];
    `dead` is computed from the RAW (unclamped) A-energy."""
    if act == "GATHER":
        raw = EA + GAIN - METAB
        return (min(E_MAX, max(0, raw)), min(E_MAX, max(0, EB - B_METAB))), raw <= 0
    if act == "REST":
        raw = EA - METAB
        return (min(E_MAX, max(0, raw)), min(E_MAX, max(0, EB - B_METAB))), raw <= 0
    if act == "EXCHANGE":
        bonus = BONUS if (b_recip and EB > 0) else 0   # only get it while B is alive
        raw = EA + bonus - METAB - GIVE
        EAn = min(E_MAX, max(0, raw))
        EBn = min(E_MAX, max(0, EB - B_METAB + GIVE))
        return (EAn, EBn), raw <= 0
    raise ValueError(act)


def reward(EA, EB, mode):
    """value of state (EA,EB): A prefers being ALIVE and (slightly) higher-energy,
    so a mutual EXCHANGE (higher EA) is strictly preferred over GATHER, and a
    draining EXCHANGE (lower EA) is avoided. Goal mode adds B-alive value."""
    alive = 1.0 if EA > 0 else 0.0
    energy_term = 0.5 * (EA / E_MAX)          # soft preference for higher energy
    if mode == "goal":
        return alive + energy_term + GOAL_WEIGHT * (1.0 if EB > 0 else 0.0)
    return alive + energy_term  # coupling: A cares only about ITS OWN living


def dp_policy(mode, b_recip, H=H):
    """Finite-horizon DP over (EA, EB). Returns dict (EA,EB)->action."""
    V = [[0.0] * (E_MAX + 1) for _ in range(E_MAX + 1)]
    for _t in range(H):
        newV = [[0.0] * (E_MAX + 1) for _ in range(E_MAX + 1)]
        for EA in range(E_MAX + 1):
            for EB in range(E_MAX + 1):
                if EA <= 0:
                    newV[EA][EB] = 0.0
                    continue
                r_now = reward(EA, EB, mode)
                best = -1e18
                for act in ACTIONS:
                    (EAn, EBn), dead = T(act, EA, EB, b_recip)
                    if dead:
                        val = r_now + GAMMA * 0.0
                    else:
                        val = r_now + GAMMA * V[EAn][EBn]
                    if val > best:
                        best = val
                newV[EA][EB] = best
        V = newV
    policy = {}
    for EA in range(E_MAX + 1):
        for EB in range(E_MAX + 1):
            if EA <= 0:
                policy[(EA, EB)] = "REST"
                continue
            r_now = reward(EA, EB, mode)
            best = -1e18
            best_act = ACTIONS[0]
            for act in ACTIONS:
                (EAn, EBn), dead = T(act, EA, EB, b_recip)
                if dead:
                    val = r_now + GAMMA * 0.0
                else:
                    val = r_now + GAMMA * V[EAn][EBn]
                if val > best:
                    best = val
                    best_act = act
            policy[(EA, EB)] = best_act
    return policy


def simulate(policy, mode, b_recip, episodes=300, seed=0, max_ticks=60):
    rng = __import__("random").Random(seed)
    counts = {a: 0 for a in ACTIONS}
    total = 0
    survival = 0
    a_min = []
    b_alive_end = 0
    for _ in range(episodes):
        EA = rng.randint(10, 60)
        EB = rng.randint(10, 60)
        e_min = EA
        for _t in range(max_ticks):
            if EA <= 0:
                break
            act = policy[(min(EA, E_MAX), min(EB, E_MAX))]
            counts[act] += 1
            total += 1
            (EAn, EBn), _dead = T(act, EA, EB, b_recip)
            EA, EB = EAn, EBn
            if EA <= 0:
                break
            e_min = min(e_min, EA)
        if EA > 0:
            survival += 1
        a_min.append(e_min)
        if EB <= 0:
            b_alive_end += 0
        else:
            b_alive_end += 1
    n = episodes
    return {
        "survival_rate": round(survival / n, 3),
        "exchange_rate": round(counts["EXCHANGE"] / total, 3) if total else 0.0,
        "gather_rate": round(counts["GATHER"] / total, 3) if total else 0.0,
        "mean_min_EA": round(sum(a_min) / n, 1),
        "b_alive_rate": round(b_alive_end / n, 3),
    }


def run_cell(mode, b_recip, seed=0, episodes=300, b_metab=B_METAB):
    global B_METAB
    B_METAB = b_metab
    policy = dp_policy(mode, b_recip)
    return simulate(policy, mode, b_recip, episodes=episodes, seed=seed)


def sweep(seeds=(1, 2, 3), episodes=250):
    """Headline: as the partner becomes NEEDIER (a heavier drain, b_recip=False),
    does A sacrifice its own survival margin to keep the relationship alive?
       COUPLING -> A walks away (self-preservation): min-EA stays flat.
       GOAL     -> A keeps feeding, draining itself: min-EA drops with need.
    This is the docs/20 + docs/22 mechanism: relationship-as-goal coerces A to
    sacrifice for the relationship; relationship-as-coupling preserves A's freedom."""
    print("=== SEED-22: steady-state coupling -- neediness sweep (partner is a DRAIN, b_recip=False) ===")
    print(f"{'B_need':<6} {'mode':<9} {'surviv':<7} {'exch':<6} {'minEA':<6} {'B_live':<6}")
    results = []
    for need in (1, 2, 3, 4):
        for mode in ("coupling", "goal"):
            acc = []
            for s in seeds:
                acc.append(run_cell(mode, False, seed=s, episodes=episodes, b_metab=need))
            surv = sum(r["survival_rate"] for r in acc) / len(acc)
            exch = sum(r["exchange_rate"] for r in acc) / len(acc)
            men = sum(r["mean_min_EA"] for r in acc) / len(acc)
            bal = sum(r["b_alive_rate"] for r in acc) / len(acc)
            results.append({"need": need, "mode": mode, "survival_rate": round(surv, 3),
                            "exchange_rate": round(exch, 3), "mean_min_EA": round(men, 1),
                            "b_alive_rate": round(bal, 3)})
            print(f"{need:<6} {mode:<9} {surv:<7.3f} {exch:<6.3f} {men:<6.1f} {bal:<6.3f}")
    return results


def main():
    p = argparse.ArgumentParser(description="SEED-22 steady-state coupling")
    p.add_argument("--mode", choices=["coupling", "goal"], default="coupling")
    p.add_argument("--b-recip", type=int, choices=[0, 1], default=1)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--episodes", type=int, default=250)
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args()
    if args.sweep:
        out = sweep(seeds=tuple(range(1, args.seeds + 1)), episodes=args.episodes)
        with open("seed-22/results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("\nfull results -> seed-22/results.json")
        return
    r = run_cell(args.mode, bool(args.b_recip), episodes=args.episodes)
    print(json.dumps(r, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
