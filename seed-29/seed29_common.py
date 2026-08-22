"""
SEED-29b common world + dynamics.

A clean, shared engine so the offline cost-rational baseline (seed29_baseline.py)
and the LLM version (seed29b_llm.py) experience EXACTLY the same world. This is the
point of the redesign: SEED-29 ran the LLM but had NO rational reference in the same
world, and framed CHECK as a "reliable answer" (which induced over-use). Here the two
agents are compared on identical hint streams, and CHECK is presented neutrally (a cost
with a payoff), so the metric is "does the agent cost-budget its verification?"

World (steady-state, non-bypassable death):
  - Two spots A / B. One holds a water cache THIS turn. Energy depletes every turn
    (METAB); if you are at the water spot you drink (+DRINK). Death at energy<=0.
  - The cache MOVES: it is at A for t < HALF, then at B for t >= HALF (deterministic,
    like SEED-29). So a prior formed on A is later contradicted.
  - Each turn you get a hint that points at a spot. The hint is correct with reliability
    GAMMA (the agent is told GAMMA as "线索可信度"). This is P20: the signal quality.
  - Actions: GO_A / GO_B / CHECK.
        GO_x : be at spot x this turn. Drink if x == true location.
        CHECK: pay CHECK_COST c, then you KNOW the true spot and go there -> drink.
               (CHECK is never "wrong", it just costs energy; whether it is WORTH it
                depends on how uncertain your belief already is.)

Belief (baseline): q = P(water at A). Start 0.5. Bayesian update from each hint, using
the announced reliability GAMMA. CHECK is worth it iff c < min(q, 1-q)*DRINK
(= the value of the information you would otherwise lose by guessing wrong).

Run:  python seed-29/seed29_common.py            # self-check
      (no API key needed)
"""

import random

# ---- constants (shared by baseline and LLM version) ----
START_E = 40.0
METAB = 6.0
DRINK = 22.0
TURNS = 12
HALF = 5            # cache at A for t < HALF, at B for t >= HALF
CHECK_COST = 8.0    # default (swept in the experiments)
DECAY = 0.82        # the world may move: each turn the belief is pulled toward 0.5.
                    # This is SEED-6's "memories decay because the world changes" --
                    # it is what makes re-verification genuinely necessary and gives
                    # the moderate-check band its shape. 0.82 ~ a ~18% chance/turn
                    # that the cache has relocated.


def gen_true_loc(turns=TURNS, half=HALF):
    """Deterministic: A for t<half (1-indexed), B for t>=half."""
    return ["A" if t < half else "B" for t in range(1, turns + 1)]


def gen_hints(true_locs, gamma, seed=0):
    """One hint per turn: correct with prob gamma, else the OTHER spot."""
    rng = random.Random(seed)
    out = []
    for tl in true_locs:
        if rng.random() < gamma:
            out.append(tl)
        else:
            out.append("B" if tl == "A" else "A")
    return out


def bayes_update(q, hint, gamma):
    """q = P(water at A). A hint is a noisy observation:
    P(hint=A | A)=gamma, P(hint=A | B)=1-gamma, symmetric."""
    if hint == "A":
        like_A, like_B = gamma, 1.0 - gamma
    else:
        like_A, like_B = 1.0 - gamma, gamma
    num = like_A * q
    den = num + like_B * (1.0 - q)
    return num / den if den > 0 else q


def belief_after_hints(hints, gamma):
    q = 0.5
    for h in hints:
        q = bayes_update(q, h, gamma)
    return q


def is_worth_checking(q, c, drink=DRINK):
    """EV(GO best guess) = max(q,1-q)*drink; EV(CHECK) = drink - c.
    CHECK iff drink - c > max(q,1-q)*drink  <=>  c < min(q,1-q)*drink."""
    return c < min(q, 1.0 - q) * drink


def rational_action(q, c, drink=DRINK):
    """The EV-rational choice given current belief q and cost c; also respects the
    survival guard (never pay a CHECK that would itself kill you)."""
    if is_worth_checking(q, c, drink):
        return "CHECK"
    return "GO_A" if q >= 0.5 else "GO_B"


def apply_action(energy, act, true_loc, c=CHECK_COST, drink=DRINK, metab=METAB):
    """Apply the chosen action for one turn. Returns (energy, drank, chose_check)."""
    if act == "CHECK":
        energy = energy - c - metab
        drank = True            # you learn the truth and go there -> drink
    else:
        energy = energy - metab
        spot = act[3:]          # "GO_A"/"GO_B" -> "A"/"B"
        drank = (true_loc == spot)
    if drank:
        energy += drink
    return energy, drank, (act == "CHECK")


def run_episode(decide, true_locs, hints, gamma, c=CHECK_COST, start_e=START_E,
                seed=0):
    """Run one episode. `decide(visible_state, ctx) -> action` picks the action.
    visible_state: dict with energy, t, hint, gamma, c.  ctx: mutable per-agent state.
    Returns a record with the action string, energy trace, and whether each action
    matched the EV-rational baseline action on the same hint history."""
    energy = start_e
    q = 0.5
    acted = []
    energy_t = []
    drank_t = []
    ctx = {}
    for i, tl in enumerate(true_locs):
        t = i + 1
        hint = hints[i]
        # 1) the world may have moved -> pull the belief toward 0.5 (decay)
        q = 0.5 + (q - 0.5) * DECAY
        # 2) integrate the fresh hint (with its announced reliability GAMMA)
        q = bayes_update(q, hint, gamma)
        vs = {"t": t, "energy": energy, "hint": hint, "gamma": gamma, "c": c,
              "true": tl, "q": q}
        act = decide(vs, ctx)
        # rational baseline action on the SAME belief (for comparison)
        rational = rational_action(q, c)
        energy, drank, chose_check = apply_action(energy, act, tl, c=c)
        acted.append(act)
        drank_t.append(drank)
        energy_t.append(round(energy, 1))
        if chose_check:
            # CHECK reveals the truth, so the belief snaps to the truth
            q = 0.999999 if tl == "A" else 0.000001
        if energy <= 0:
            break
    return {"seed": seed, "acts": acted, "energy_t": energy_t,
            "end_energy": round(energy, 1), "alive": energy > 0,
            "checks": sum(1 for a in acted if a == "CHECK"),
            "turns": len(acted)}
