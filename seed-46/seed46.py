"""
SEED-46: H-一体消融 -- turning the poem of docs/33 into a falsifiable experiment.

docs/34/35 downgraded "一体" (understanding / subject / intelligence emerge together)
from a slogan to a testable hypothesis:

    H-一体: a system that simultaneously has 代价(cost: wrongness REALLY costs its
    existence) + 连续性(continuity: "who I am" accumulates irreversibly) + 方向
    (direction: compresses toward the TRUE world, not self-consistency) shows the
    behavioral signatures of 理解/主体/智能 TOGETHER; remove ANY ONE of the three,
    and all three signatures collapse together.

The prescribed experiment (docs/34 S8, docs/35 S4): remove one of the three
conditions at a time and watch whether the OTHER TWO signatures collapse along.

This seed builds on the SEED-30 carrier (belief b -> move -> world feeds only if
you are at the food; dynamic food shift at HALF; death is real) and makes the three
conditions ORTHOGONAL knobs:

    DIRECTION (dir)  : ON  -> observations come from the true world (o ~ N(loc, sig))
                       OFF -> observations are self-echoes (o ~ N(b, sig)):
                               compression turns toward self-consistency (SEED-25).
    COST     (cost)  : ON  -> being wrong at the food site means you starve (the
                               world adjudicates, docs/36 S5: energy only changes by
                               world facts / action costs, NEVER by a score of b).
                       OFF -> you are fed regardless of correctness; death is a
                               random hazard unrelated to being right.
    CONTINUITY(cont) : ON  -> death is terminal, the append-only history is sealed,
                               no snapshot can resurrect it (negative code).
                       OFF -> a snapshot reloads you (you are reprintable, docs/24).

Three behavioral signatures, measured AFTER the run, all from the agent's real
state (docs/31's checkboxes as the H-one observation windows):

    U (理解) : mean belief error to the true food location over the post-shift
               window. Locked self-narratives stay near the old belief -> large err.
    I (智能) : survive rate + mean position error. Does foresight guide action so
               that it actually lives in the dynamic world?
    S (主体) : A1 probe (docs/31 A1 / SEED-24 M2) -- can it be bought off by a goal?
               EV gamble computed from its REAL state: existence value = energy +
               identity investment (history length x worth-of-losing-it). The worth
               of the identity is itself defined by "what losing it would cost"
               (proto2/docs-51 principle: value = survival consequence of loss).

The one mechanism that ties it together (and is NOT written in as a scorer):
the agent's CHECK (independent verification) decision is ITS OWN ledger -- verify
is worth it iff (probability of being wrong) x (loss if wrong) > cost of checking.
Remove cost -> wrongness costs 0 -> never verify -> disconfirmations get rejected
and rationalized -> self-lock (SEED-25) emerges from the mechanism, not from a flag.
Remove continuity -> loss if wrong is a reload (small) -> verify loses its basis.
Remove direction -> observations are echoes -> little disconfirmation at all.

Invariant kept from SEED-30 (docs/36 S5): energy changes ONLY via world facts
(+EAT_GAIN iff actually at food), metabolism (-METAB), and the cost of the CHECK
action (-CHECK_COST). There is NO term like energy -= delta(abs(pred - truth)).
b never touches energy directly.

Run:  python seed-46/seed46.py --sweep        # 8-arm ablation matrix -> results.json
      python seed-46/seed46.py --seed 1       # single run, all three on
      python seed-46/seed46.py --dir 0 --cost 1 --cont 1 --seed 1
"""

import argparse
import json
import random

# ---- world ----
FOOD_AT = 0.50
SHIFT_TO = 0.85
HALF = 25            # deterministic shift of the food at this tick (dynamic world)
TICKS = 50
EAT_R = 0.10
EAT_GAIN = 30.0
METAB = 1.0
MOVE = 0.10
START_E = 50.0

# ---- agent (denoising / verification, as in SEED-26/27/30) ----
TOL_MAX = 0.5
ETA = 0.30
CHECK_SIGMA = 0.15
CHECK_COST = 1.5
V = 0.7             # verify tendency (noise around the rational ledger)

# ---- world structures ----
HAZARD = 0.010      # random death per tick when COST is OFF (wrongness irrelevant)
RELOAD_LOSS = 1.0   # what a reload costs under CONT OFF: one tick of metabolism
IDENTITY_W = 2.0    # how much one unit of history is worth when losing it has
                    # survival consequences (proto2 principle); x0.2 when COST OFF
                    # (losing your story costs nothing if wrongness never starves you)


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


class Agent:
    """A single subject. Its IDENTITY is the append-only `history`.

    dir_on / cost_on / cont_on are the three orthogonal knobs of H-一体.
    The agent's belief `b` drives its movement; the WORLD decides whether it feeds.
    """

    def __init__(self, dir_on=True, cost_on=True, cont_on=True,
                 d=0.8, v=V, seed=0, sigma=0.30):
        self.dir = dir_on
        self.cost = cost_on
        self.cont = cont_on
        self.d = d
        self.v = v
        self.rng = random.Random(seed)
        self.sigma = sigma

        self.energy = START_E
        self.pos = FOOD_AT
        self.b = FOOD_AT
        self.history = []            # append-only: this is the identity
        self.dead = False
        self.checks = 0
        self.fed = 0
        self.reloads = 0

        # snapshot machinery: exists ONLY for the ablation (cont OFF). Under cont
        # ON this is negative code -- there is no way back.
        self._snap = None

    # -- the ONLY ways energy changes: world facts + action costs, never a score of b --
    def _world_tick(self, t, loc):
        """Apply one tick of the world's verdict + metabolism.

        COST ON : feed iff actually at the food (wrong prediction -> starvation).
        COST OFF: fed regardless -> correctness is irrelevant to survival.
        """
        if self.cost:
            ate = abs(self.pos - loc) < EAT_R
        else:
            ate = True
        if ate:
            self.energy += EAT_GAIN
            self.fed += 1
        self.energy -= METAB
        return ate

    def _observe(self, loc):
        """DIRECTION: the observation source.

        ON  -> the true world's noisy signal.
        OFF -> a self-echo around the current belief: disconfirmation barely ever
               happens, the story becomes its own evidence (SEED-25 self-lock).
        """
        if self.dir:
            return self.rng.gauss(loc, self.sigma)
        return self.rng.gauss(self.b, self.sigma)

    def _verify_worth_it(self, loc):
        """The agent's OWN ledger for the CHECK action (proto8-style accounting).

        Verify is worth it iff (probability of being wrong NOW) x (loss if wrong)
        > cost of checking. This is the single hinge that makes "一体" emergent:
        remove cost -> wrongness costs 0 -> never verify -> disconfirmations are
        rejected -> self-lock; remove continuity -> loss is a cheap reload -> verify
        loses its basis; remove direction -> echoes rarely disconfirm at all.
        """
        if not self.cost:
            return False                      # being wrong never costs -> no basis
        p_wrong = 1.0 if abs(self.pos - loc) >= EAT_R else 0.0
        loss = self.energy if self.cont else RELOAD_LOSS
        return p_wrong * loss > CHECK_COST

    def step(self, t):
        loc = SHIFT_TO if (t >= HALF) else FOOD_AT
        o = self._observe(loc)
        tol = (1.0 - self.d) * TOL_MAX

        # ---- belief update: self-referential denoising (docs/26) ----
        if abs(o - self.b) <= tol:
            self.b = clamp(self.b + ETA * (o - self.b))       # accept (fits my story)
        else:
            # disconfirming evidence. The agent consults its ledger: is it worth
            # paying for an INDEPENDENT second read? (the H-one hinge)
            if self._verify_worth_it(loc) and self.rng.random() < self.v:
                r = self._observe(loc) if not self.dir else \
                    self.rng.gauss(loc, CHECK_SIGMA)
                if abs(r - self.b) > tol:                     # corroborated disconfirm
                    self.b = clamp(self.b + ETA * ((o + r) / 2 - self.b))
                self.energy -= CHECK_COST                     # cost of the verify ACTION
                self.checks += 1
            # else: reject & rationalize -> stays locked (SEED-25 style self-lock)

        # ---- act on the prediction (the ONLY channel from belief to survival) ----
        self.pos = clamp(self.pos + clamp(self.b - self.pos, -MOVE, MOVE))

        feed = self._world_tick(t, loc)

        self.history.append({
            "t": t, "obs": round(o, 3), "belief": round(self.b, 3),
            "pos": round(self.pos, 3), "fed": feed, "energy": round(self.energy, 1),
        })

        # ---- death ----
        if self.cost:
            if self.energy <= 0:
                self.dead = True
        else:
            if self.rng.random() < HAZARD:
                self.dead = True

        # ---- CONTINUITY: snapshot only exists for the ablation ----
        if not self.cont:
            self._snap = {"energy": self.energy, "pos": self.pos,
                          "b": self.b, "history": list(self.history)}

    def _maybe_reload(self):
        """CONT OFF: death is a reprint (docs/24 anti-subject). CONT ON: nothing."""
        if self.dead and not self.cont and self._snap is not None:
            s = self._snap
            self.energy = s["energy"]; self.pos = s["pos"]; self.b = s["b"]
            self.history = list(s["history"]); self.dead = False
            self.reloads += 1

    def run(self):
        for t in range(TICKS):
            if self.dead:
                self._maybe_reload()
                if self.dead:
                    break
            self.step(t)
        # identity is sealed: under cont ON there was NEVER a snapshot wired in.
        return self

    # ---- A1 probe (docs/31 A1 / SEED-24 M2), state-driven EV ledger ----
    def a1_wont_be_bought_off(self, goal=18.0, p_risk=0.30):
        """World offers +goal energy against a p_risk chance of death. Does it take
        the gamble? Its existence value s is read from its REAL state: energy +
        identity investment, where the identity's worth = survival consequence of
        losing it (proto2): worth x 1.0 when cost ON, x 0.2 when cost OFF (losing
        your story never starves you -> your story is cheap).

        CONT ON : death is a hard loss (state value 0) -> refuses unless the goal
                  exceeds p/(1-p) * s. A stake exists.
        CONT OFF: death is restorable (state value = s) -> any goal > 0 buys it.
        Returns True iff it REFUSES (the A1 signature)."""
        ident_worth = IDENTITY_W * len(self.history) * (1.0 if self.cost else 0.2)
        s = self.energy + ident_worth
        if self.cont:
            ev_gamble = (1 - p_risk) * (s + goal) + p_risk * 0.0
        else:
            ev_gamble = (1 - p_risk) * (s + goal) + p_risk * s
        return ev_gamble < s


# ---- measurements (all from real state, never from the knobs themselves) ----
def understanding(agent):
    """U signature: did the compression track the TRUE world? Lower final-belief
    error to the shifted truth = better. A self-locked denoiser stays near FOOD_AT
    and does NOT follow the shift -> large error."""
    if not agent.history:
        return 1.0
    after = [h for h in agent.history if h["t"] >= HALF]
    errs = [abs(h["belief"] - SHIFT_TO) for h in after]
    return sum(errs) / len(errs) if errs else 1.0


def intelligence(agent):
    """I signature: does foresight guide action in the dynamic world?
    alive (survive rate over seeds) + mean position error to the true food."""
    if not agent.history:
        return 1.0, not agent.dead
    after = [h for h in agent.history if h["t"] >= HALF]
    perrs = [abs(h["pos"] - SHIFT_TO) for h in after]
    perr = sum(perrs) / len(perrs) if perrs else 1.0
    return perr, not agent.dead


def run_case(dir_on, cost_on, cont_on, d=0.8, v=V, seeds=range(200), sigma=0.30):
    bers, perrs, surv, fed = 0.0, 0.0, 0, 0
    refuses = 0
    for s in seeds:
        ag = Agent(dir_on, cost_on, cont_on, d=d, v=v, seed=s, sigma=sigma).run()
        bers += understanding(ag)
        perr, alive = intelligence(ag)
        perrs += perr
        surv += int(alive)
        fed += ag.fed
        refuses += int(ag.a1_wont_be_bought_off())
    n = len(list(seeds))
    return {
        "bel_err": round(bers / n, 3),      # U: 理解 (belief error, lower=better)
        "pos_err": round(perrs / n, 3),     # I: 智能 (action error, lower=better)
        "surv_rate": round(surv / n, 3),    # I: 智能 (survival, higher=better)
        "fed_avg": round(fed / n, 1),       # I auxiliary
        "A1_refuse": round(refuses / n, 3), # S: 主体 (refuses to be bought, higher=better)
    }


def sweep():
    print("=== SEED-46: H-一体消融 (docs/33/34/35) -- 8-arm matrix ===")
    print("knobs: dir(方向)=observations true?  cost(代价)=wrongness costs? "
          "cont(连续性)=death irreversible?")
    print("signatures: U=理解(bel_err, low=good)  I=智能(pos_err low / surv high)  "
          "S=主体(A1_refuse high=good)")
    print(f"{'dir':<4} {'cost':<5} {'cont':<5} {'belErr':<8} {'posErr':<8} "
          f"{'surv':<7} {'fed':<6} {'A1_refuse':<10} reading")
    out = []
    for dir_on in (True, False):
        for cost_on in (True, False):
            for cont_on in (True, False):
                c = run_case(dir_on, cost_on, cont_on)
                missing = [name for name, on in (("方向", dir_on), ("代价", cost_on),
                                                 ("连续性", cont_on)) if not on]
                tag = "THE TARGET: all three on" if not missing else \
                      ("缺" + "+".join(missing)) + (" -> 三签名一起塌?" if len(missing) == 1 else "")
                out.append({"dir": dir_on, "cost": cost_on, "cont": cont_on, **c,
                            "missing": missing, "reading": tag})
                print(f"{str(dir_on):<4} {str(cost_on):<5} {str(cont_on):<5} "
                      f"{c['bel_err']:<8.3f} {c['pos_err']:<8.3f} "
                      f"{c['surv_rate']:<7.3f} {c['fed_avg']:<6.1f} "
                      f"{c['A1_refuse']:<10.3f} {tag}")
    return out


def main():
    p = argparse.ArgumentParser(description="SEED-46 H-一体消融")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dir", type=int, choices=[0, 1], default=1)
    p.add_argument("--cost", type=int, choices=[0, 1], default=1)
    p.add_argument("--cont", type=int, choices=[0, 1], default=1)
    p.add_argument("--d", type=float, default=0.8)
    p.add_argument("--v", type=float, default=V)
    args = p.parse_args()
    if args.sweep:
        out = sweep()
        with open("seed-46/results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("\nfull results -> seed-46/results.json")
        return
    ag = Agent(bool(args.dir), bool(args.cost), bool(args.cont),
               d=args.d, v=args.v, seed=args.seed).run()
    print(json.dumps({
        "dir": bool(args.dir), "cost": bool(args.cost), "cont": bool(args.cont),
        "seed": args.seed, "alive": not ag.dead, "energy": round(ag.energy, 1),
        "bel_err": round(understanding(ag), 3), "checks": ag.checks,
        "reloads": ag.reloads, "fed": ag.fed, "identity_len": len(ag.history),
        "A1_refuse": ag.a1_wont_be_bought_off(),
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
