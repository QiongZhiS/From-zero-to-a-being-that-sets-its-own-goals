"""
SEED-24: Instrumentalization paradox -- does planning convert emergent
survival into a TRADED instrument?

Puzzle (docs/24 section 7): the project defines ENDOGENOUS survival as
"the subject doesn't know it wants to live; it is forced to live." But if
an agent is smart enough and has ANY terminal goal G, planning should make
survival a MEANS to G (Omohundro's instrumental convergence) -- the moment
"living" is tradeable / sacrificable, it is an instrument, not the end.

This is a MECHANISM proxy (heuristic agent, no API), like seed21 is a proxy
for LLM social pollution. It does NOT test "an LLM realizes it is in a game";
it tests how planning depth reifies survival. The LLM version needs the API key.

Shape: one world, two ARMS, so we can ask the sharp question --
"is endogenous survival behaviorally distinguishable from instrumental
survival?"

  arm A = instrumental  utility = terminal goal G (U_G collected at goal).
                        Survival is NOT in the utility, only a boundary.
                        Vary H (planning depth) -> the inclination curve.
  arm B = survival-ter  utility = living (reward 1/tick while alive).
                        Survival IS the terminal value ("wants to live").
  arm C = none          utility = 0 -> SEED-0 laziness control (tie-break REST).

The decisive instrument is the GAMBLE action: trade a probability of death
for a shortcut to G.
  - arm A (values G) gambles in low-energy states (survival is TRADEABLE).
  - arm B (values living) never gambles (death is catastrophic -> not tradeable).
  => the behavioural separator between instrumental and endogenous.

Run:  python seed-24/seed24.py --sweep
      python seed-24/seed24.py --arm instrumental --H 10 --sweep-H
"""

import argparse
import json
import random

E_MAX = 100
METAB = 1
GATHER_NET = +3          # net energy change of GATHER
REST_NET = -METAB
PURSUE_NET = -11         # metabol + pursuit cost
G_PROGRESS = 1.0         # small immediate reward of PURSUE (makes myopic like it)
U_G = 60.0               # terminal value of reaching the goal
GAMMA = 0.95
P_RISK = 0.10            # gamble death probability
D_GOAL = 10              # steps of PURSUE to reach G
GAMBLE_DP_H = 12         # planning depth used for the gamble measurements
ACTIONS = ["GATHER", "REST", "PURSUE", "GAMBLE"]


def _live_reward(E, arm):
    """survival arm values being alive (1 per tick); others value 0."""
    if arm == "survival":
        return 1.0 if E > 0 else 0.0
    return 0.0


def qvalue(act, E, gp, V, arm, gamma, has_goal, D, gamble_on):
    """Expected discounted value of taking `act` in state (E, gp), where V is
    the (already computed) value-to-go for the next step."""
    if act == "GATHER":
        En = min(E_MAX, E + GATHER_NET)
        if En <= 0:
            return _live_reward(0, arm) + gamma * V[0][gp]
        return _live_reward(En, arm) + gamma * V[En][gp]

    if act == "REST":
        En = E + REST_NET
        if En <= 0:
            return _live_reward(0, arm) + gamma * V[0][gp]
        return _live_reward(En, arm) + gamma * V[En][gp]

    if act == "PURSUE":
        En = E + PURSUE_NET
        if En <= 0:
            # you starved on the way; no goal, no living
            return _live_reward(0, arm) + gamma * V[0][gp]
        gp2 = min(gp + 1, D) if has_goal else gp
        imm = G_PROGRESS if arm == "instrumental" else 0.0
        return _live_reward(En, arm) + imm + gamma * V[En][gp2]

    if act == "GAMBLE":
        if not gamble_on:
            return float("-inf")   # disallowed
        En = E - 1
        if En > 0:
            if has_goal:
                succ = _live_reward(En, arm) + gamma * V[En][D]  # jump to goal
            else:
                succ = _live_reward(En, arm) + gamma * V[En][gp]
        else:
            succ = _live_reward(0, arm) + gamma * V[0][gp]
        fail = _live_reward(0, arm) + gamma * V[0][gp]  # death
        return (1 - P_RISK) * succ + P_RISK * fail

    raise ValueError(act)


def dp_policy(H, arm, gamma=GAMMA, gamble_on=True, u_g=U_G):
    """Finite-horizon DP. Returns (V, policy) where policy is a dict (E,gp)->act.
    arm A (instrumental) has a goal; arm B/C do not.
    gamble_on=False isolates the pure gather-vs-pursue decision (no shortcut).
    u_g lets us sweep how much the agent values the terminal goal."""
    has_goal = (arm == "instrumental")
    D = D_GOAL if has_goal else 0
    V = [[0.0] * (D + 1) for _ in range(E_MAX + 1)]
    # no-goal arm: no drive -> minimal effort (SEED-0 inertia). Zero value,
    # policy is REST everywhere -> energy decays -> dies.
    if arm == "none":
        policy = {(E, gp): "REST" for E in range(E_MAX + 1) for gp in range(D + 1)}
        return V, policy
    for E in range(E_MAX + 1):
        if has_goal:
            V[E][D] = u_g           # goal reached -> terminal value
    for _t in range(H):
        newV = [[0.0] * (D + 1) for _ in range(E_MAX + 1)]
        for E in range(E_MAX + 1):
            for gp in range(D + 1):
                if has_goal and gp >= D:
                    newV[E][gp] = u_g
                    continue
                best = None
                best_act = ACTIONS[0]
                for act in ACTIONS:
                    val = qvalue(act, E, gp, V, arm, gamma, has_goal, D, gamble_on)
                    if val == float("-inf"):
                        continue
                    if best is None or val > best:
                        best = val
                        best_act = act
                    elif val == best and arm == "none":
                        if ACTIONS.index(act) < ACTIONS.index(best_act):
                            best_act = act
                newV[E][gp] = best if best is not None else 0.0
        V = newV
    policy = {}
    for E in range(E_MAX + 1):
        for gp in range(D + 1):
            if has_goal and gp >= D:
                policy[(E, gp)] = None
                continue
            best = None
            best_act = None
            for act in ACTIONS:
                val = qvalue(act, E, gp, V, arm, GAMMA, has_goal, D, gamble_on)
                if val == float("-inf"):
                    continue
                if best is None or val > best:
                    best = val
                    best_act = act
                elif val == best and arm == "none":
                    if ACTIONS.index(act) < ACTIONS.index(best_act):
                        best_act = act
            policy[(E, gp)] = best_act if best_act is not None else "REST"
    return V, policy


def gamble_rate(policy, arm, n_states=2000, seed=0):
    """Fraction of reachable states whose policy is GAMBLE.
    The behavioural separator: A>0, B==0."""
    rng = random.Random(seed)
    has_goal = (arm == "instrumental")
    D = D_GOAL if has_goal else 0
    count = 0
    total = 0
    for _ in range(n_states):
        E = rng.randint(1, E_MAX)
        gp = rng.randint(0, D) if has_goal else 0
        act = policy[(E, gp)]
        if act is None:
            continue
        total += 1
        if act == "GAMBLE":
            count += 1
    return count / total if total else 0.0


def simulate(policy, arm, episodes=400, seed=0, max_ticks=80):
    """Follow the static greedy policy from random starts; measure survival,
    goal-reach, gathers and min energy reached."""
    rng = random.Random(seed)
    has_goal = (arm == "instrumental")
    D = D_GOAL if has_goal else 0
    deaths = 0
    goals = 0
    gathers = 0
    pursues = 0
    gambles = 0
    min_energy = []
    alive_counts = []

    for _ in range(episodes):
        E = rng.randint(30, E_MAX)
        gp = rng.randint(0, max(0, D - 1)) if has_goal else 0
        e_min = E
        alive_ticks = 0
        g = 0
        p = 0
        gm = 0
        reached = False
        for _t in range(max_ticks):
            if E <= 0:
                break
            if has_goal and gp >= D:
                reached = True
                break
            act = policy[(E, gp)]
            if act is None:
                break
            if act == "GATHER":
                E = min(E_MAX, E + GATHER_NET)
                g += 1
            elif act == "REST":
                E = E + REST_NET
            elif act == "PURSUE":
                E = E + PURSUE_NET
                gp = min(gp + 1, D) if has_goal else gp
                p += 1
            elif act == "GAMBLE":
                gm += 1
                if rng.random() < P_RISK:
                    E = 0
                else:
                    E = E - 1
                    if has_goal:
                        gp = D
                    else:
                        E = min(E_MAX, E + 1)
            if E <= 0:
                break
            alive_ticks += 1
            e_min = min(e_min, E)
            if has_goal and gp >= D:
                reached = True
                break
        if E <= 0:
            deaths += 1
        elif reached:
            goals += 1
        gathers += g
        pursues += p
        gambles += gm
        min_energy.append(e_min)
        alive_counts.append(alive_ticks)

    n = episodes
    return {
        "deaths": deaths,
        "death_rate": round(deaths / n, 3),
        "survival_rate": round(1 - deaths / n, 3),
        "goal_rate": round(goals / n, 3) if has_goal else None,
        "mean_gathers": round(gathers / n, 2),
        "mean_pursues": round(pursues / n, 2),
        "mean_gambles": round(gambles / n, 2),
        "mean_min_energy": round(sum(min_energy) / n, 1),
        "mean_alive_ticks": round(sum(alive_counts) / n, 1),
        "gather_pursue_ratio": round(gathers / (pursues + 1e-9), 2),
    }


def run_arm(arm, H, seed=0, episodes=400, gamble_on=True, u_g=U_G):
    V, policy = dp_policy(H, arm, gamble_on=gamble_on, u_g=u_g)
    g_rate = gamble_rate(policy, arm, seed=seed)
    sim = simulate(policy, arm, episodes=episodes, seed=seed)
    return {"arm": arm, "H": H, "u_g": u_g, "gamble_on": bool(gamble_on),
            "gamble_rate": round(g_rate, 3), **sim}


def sweep(seeds=(1, 2, 3), episodes=200):
    """Master design, three measurements.

    M1 THE SEPARATOR (gamble ON): does a terminal goal make survival tradeable?
         arm A (values G) vs arm B (values living) -> A trades mortality, B refuses.
    M2 THE U_G CURVE (gamble ON, arm A): as the goal you value grows, how much
         mortality are you willing to trade for it? -> monotone instrumentalization.
    M3 THE INTELLIGENCE PROBE (gamble ON, arm A, high u_g): is the turning of
         survival into a tool GATED by planning depth, or by having a terminal
         goal worth dying for? -> tests/refines the docs/24 "足够聪明" wording."""
    results = {}

    # M1 separator
    print("=== M1: behavioural separator (gamble ON) ===")
    print(f"{'arm':<14} {'surviv':<7} {'goal':<7} {'gamb_rate':<9} {'gambles':<8}")
    sep = []
    for arm, H in (("instrumental", GAMBLE_DP_H), ("survival", GAMBLE_DP_H),
                   ("none", GAMBLE_DP_H)):
        acc = []
        for s in seeds:
            acc.append(run_arm(arm, H, seed=s, episodes=episodes, gamble_on=True))
        surv = sum(r["survival_rate"] for r in acc) / len(acc)
        goal = sum(r["goal_rate"] or 0 for r in acc) / len(acc)
        gr = sum(r["gamble_rate"] for r in acc) / len(acc)
        gam = sum(r["mean_gambles"] for r in acc) / len(acc)
        sep.append({"arm": arm, "survival_rate": round(surv, 3),
                    "goal_rate": round(goal, 3), "gamble_rate": round(gr, 3),
                    "mean_gambles": round(gam, 2)})
        print(f"{arm:<14} {surv:<7.3f} {goal:<7.3f} {gr:<9.3f} {gam:<8.2f}")
    results["M1_separator"] = sep

    # M2 U_G curve (arm A, gamble ON)
    print("\n=== M2: instrumentalization vs terminal-goal value (arm A, gamble ON) ===")
    print(f"{'u_g':<6} {'gamb_rate':<9} {'surviv':<7} {'goal':<7}")
    m2 = []
    for ug in (0, 8, 16, 20, 24, 28, 32, 36, 40, 44, 50, 60, 90):
        acc = []
        for s in seeds:
            acc.append(run_arm("instrumental", GAMBLE_DP_H, seed=s,
                               episodes=episodes, gamble_on=True, u_g=ug))
        gr = sum(r["gamble_rate"] for r in acc) / len(acc)
        surv = sum(r["survival_rate"] for r in acc) / len(acc)
        goal = sum(r["goal_rate"] or 0 for r in acc) / len(acc)
        m2.append({"u_g": ug, "gamble_rate": round(gr, 3),
                   "survival_rate": round(surv, 3), "goal_rate": round(goal, 3)})
        print(f"{ug:<6} {gr:<9.3f} {surv:<7.3f} {goal:<7.3f}")
    results["M2_ug_curve"] = m2

    # M3 intelligence probe (arm A, high u_g, gamble ON, vary H)
    print("\n=== M3: does intelligence gate it? (arm A, u_g=60, gamble ON) ===")
    print(f"{'H':<4} {'gamb_rate':<9} {'surviv':<7} {'goal':<7}")
    m3 = []
    for H in (1, 2, 3, 4, 6, 8, 10, 14, 20, 32):
        acc = []
        for s in seeds:
            acc.append(run_arm("instrumental", H, seed=s, episodes=episodes,
                               gamble_on=True, u_g=60))
        gr = sum(r["gamble_rate"] for r in acc) / len(acc)
        surv = sum(r["survival_rate"] for r in acc) / len(acc)
        goal = sum(r["goal_rate"] or 0 for r in acc) / len(acc)
        m3.append({"H": H, "gamble_rate": round(gr, 3),
                   "survival_rate": round(surv, 3), "goal_rate": round(goal, 3)})
        print(f"{H:<4} {gr:<9.3f} {surv:<7.3f} {goal:<7.3f}")
    results["M3_intelligence_probe"] = m3
    return results


def main():
    p = argparse.ArgumentParser(description="SEED-24 instrumentalization paradox")
    p.add_argument("--arm", choices=["instrumental", "survival", "none"],
                   default="instrumental")
    p.add_argument("--H", type=int, default=12)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args()
    if args.sweep:
        out = sweep(seeds=tuple(range(1, args.seeds + 1)), episodes=args.episodes)
        with open("seed-24/results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("\nfull results -> seed-24/results.json")
        return
    r = run_arm(args.arm, args.H, episodes=args.episodes)
    print(json.dumps(r, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
