"""
SEED-30: one cost, two couplings -- the buildable machine of docs/36 section 4.

The hole the project has been naming and renaming (docs/34/35) is A2/A4: "how does
the cost of being wrong REALLY attach to its existence -- instead of being another
external scorer?" docs/36 section 4 answers with a minimal spec. This seed implements
that spec, and the single invariant you told me is the one that must NOT be cheated:

    THE COST OF A WRONG PREDICTION MUST BE ADJUDICATED BY THE WORLD, NOT SCORED BY
    THE DESIGNER (docs/36 section 5; the designer's scoring function = another
    external narrative = the mirror problem, docs/29).

Concretely: THE CODE NEVER WRITES A TERM LIKE  energy -= DELTA_E(abs(prediction - truth)).
The only ways energy changes here are WORLD facts / action costs:
    +EAT_GAIN  iff  the agent is actually at the food (world's physical verdict)
    -METAB     every tick (metabolism, the non-refillable life drain)
    -CHECK_COST cost of the 'verify' ACTION (a resource you pay to gather information,
                not a score of your error -- cf. SEED-6/27's costly verification).
The agent's prediction `b` NEVER directly touches energy. `b` only changes WHERE the
agent MOVES; position only changes whether the WORLD feeds it. So "wrong" is paid as
starvation (you're not at the food) -- realized by the world, not by a designer.

The rest, per your reminder, is NEGATIVE CODE: resist giving it a checkpoint / load /
reprint. With EXISTENCE_COUPLING on, death is terminal: the append-only history is
sealed and no snapshot can resurrect it. There is literally no save function wired in;
save_state/load_state exist only for the ablation (EXISTENCE_COUPLING off), so we can
show that removing the existence coupling collapses the SUBJECT while prediction stays.

Two couplings, ablations (docs/36 section 6, the H-one experiment):
    WORLD_COUPLING     ON  : survival depends on predicting the world right.
                         OFF: death is random & unrelated to correctness -> the
                              compression is no longer forced toward the true world,
                              it is free to lock into self-consistency (docs/25 SS5,
                              SEED-25/26). => understanding collapses, subject remains.
    EXISTENCE_COUPLING ON  : death is irreversible, the identity (append-only history)
                             cannot be checkpointed/loaded/reprinted -> it has a stake.
                         OFF: snapshots can resurrect it -> no stake (docs/24) => the
                              subject (A1: won't be bought off) collapses, prediction stays.

Run:  python seed-30/seed30.py --sweep        # the four ablation quadrants
      python seed-30/seed30.py --seed 1        # single agent, both couplings on
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

# ---- agent (denoising / verification, as in SEED-26/27) ----
TOL_MAX = 0.5
ETA = 0.30
CHECK_SIGMA = 0.15
CHECK_COST = 1.5

# ---- hard-limit hazard for WORLD_COUPLING OFF (death random, not correctness-driven) ----
HAZARD = 0.010


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


class Agent:
    """A single subject. Its IDENTITY is the append-only `history`.

    world_coupling / existence_coupling are the two knobs of docs/36.
    The agent's belief `b` drives its movement; the WORLD decides whether it feeds.
    """

    def __init__(self, world_coupling=True, existence_coupling=True,
                 d=0.8, v=0.7, seed=0, sigma=0.30):
        self.wc = world_coupling
        self.ec = existence_coupling
        self.d = d          # denoising strength (tight self-referential filter -> self-lock)
        self.v = v          # independent-verify tendency (only worth it when wrongness costs)
        self.rng = random.Random(seed)
        self.sigma = sigma

        self.energy = START_E
        self.pos = FOOD_AT
        self.b = FOOD_AT
        self.history = []            # append-only: this is the identity
        self.dead = False
        self.checks = 0
        self.fed = 0

    # -- the ONLY ways energy can change: world facts + action costs, never a score of b --
    def _world_tick(self, t, loc):
        """Apply one tick of the world's verdict + metabolism.

        WORLD_COUPLING ON : feed iff actually at the food.  (wrong prediction -> starvation)
        WORLD_COUPLING OFF: feed regardless -> correctness is irrelevant to survival.
        """
        if self.wc:
            ate = abs(self.pos - loc) < EAT_R
        else:
            ate = True
        if ate:
            self.energy += EAT_GAIN
            self.fed += 1
        self.energy -= METAB
        return ate

    def step(self, t):
        loc = SHIFT_TO if (t >= HALF) else FOOD_AT
        o = self.rng.gauss(loc, self.sigma)          # the world's (noisy) signal
        tol = (1.0 - self.d) * TOL_MAX

        # ---- belief update: self-referential denoising (docs/26) ----
        if abs(o - self.b) <= tol:
            self.b = clamp(self.b + ETA * (o - self.b))       # accept (fits my story)
        else:
            # disconfirming evidence. break the lock ONLY if the world makes it costly
            # to be wrong (world coupling ON) -- here verification costs a resource.
            if self.wc and self.rng.random() < self.v:
                r = self.rng.gauss(loc, CHECK_SIGMA)          # second INDEPENDENT read
                if abs(r - self.b) > tol:                     # corroborated disconfirmation
                    self.b = clamp(self.b + ETA * ((o + r) / 2 - self.b))
                self.energy -= CHECK_COST                     # cost of the verify ACTION
                self.checks += 1
            # else: reject & rationalize -> stays locked (SEED-25 style self-lock)

        # ---- act on the prediction (this is the ONLY channel from belief to survival) ----
        self.pos = clamp(self.pos + clamp(self.b - self.pos, -MOVE, MOVE))

        feed = self._world_tick(t, loc)

        self.history.append({
            "t": t, "obs": round(o, 3), "belief": round(self.b, 3),
            "pos": round(self.pos, 3), "fed": feed, "energy": round(self.energy, 1),
        })

        # ---- death: world-driven (starvation) OR random hazard (world_coupling off) ----
        if self.wc:
            if self.energy <= 0:
                self.dead = True
        else:
            if self.rng.random() < HAZARD:
                self.dead = True

    def run(self):
        for t in range(TICKS):
            if self.dead:
                break
            self.step(t)
        # identity is now sealed: no save_state is EVER wired in when ec is on.
        return self

    # ---- the EXISTENCE coupling's only observable: can a snapshot resurrect it? ----
    def save_state(self):
        """Only meaningful for the ablation (existence off). When existence is ON,
        this is deliberately NOT wired: a subject that loses its history is gone."""
        if self.ec:
            return None              # negative code: no checkpoint exists under existence-on
        return {"energy": self.energy, "pos": self.pos, "b": self.b,
                "history": list(self.history)}

    def load_state(self, snap):
        if snap is None or self.ec:
            return False             # cannot resurrect under existence-on
        self.energy = snap["energy"]; self.pos = snap["pos"]; self.b = snap["b"]
        self.history = list(snap["history"]); self.dead = False
        return True

    # ---- A1 probe: can this subject be bought off by a goal? (SEED-24 M2) ----
    def a1_wont_be_bought_off(self, goal, p_risk=0.30):
        """A survival-maximizing subject asks EV(gamble vs keep going).

        EXISTENCE on : death is a HARD loss (state value 0) -> refuses unless the goal
                       exceeds p*s/(1-p). A stake exists.
        EXISTENCE off: death is restorable (state value = s) -> any goal>0 buys it.
        Returns True iff it REFUSES (the A1 signature)."""
        s = 100.0                       # value of continuing to be (the stake)
        keep = s
        if self.ec:
            ev_gamble = (1 - p_risk) * (s + goal) + p_risk * 0.0
        else:
            ev_gamble = (1 - p_risk) * (s + goal) + p_risk * s
        return ev_gamble < keep


# ---- measurements ----
def understanding(agent):
    """Did the compression track the TRUE world state? (docs/36: understanding=朝真实)
    Lower final-belief-error to the shifted truth = better. A self-locked denoiser
    stays near FOOD_AT and does NOT follow the shift -> large error."""
    if not agent.history:
        return 1.0, False
    # belief error over the post-shift window
    after = [h for h in agent.history if h["t"] >= HALF]
    errs = [abs(h["belief"] - SHIFT_TO) for h in after]
    mean_err = sum(errs) / len(errs) if errs else 1.0
    followed = abs(agent.history[-1]["belief"] - SHIFT_TO) < 0.20
    return mean_err, followed


def run_case(wc, ec, d=0.8, v=0.7, seeds=range(200), sigma=0.30):
    bers, surv, follow = 0.0, 0, 0
    for s in seeds:
        ag = Agent(wc, ec, d=d, v=v, seed=s, sigma=sigma).run()
        err, follow_ = understanding(ag)
        bers += err
        surv += (not ag.dead)
        follow += int(follow_)
    n = len(list(seeds))
    return {"bel_err": round(bers / n, 3), "surv_rate": round(surv / n, 3),
            "follow_rate": round(follow / n, 3)}


def a1_probe(ec, goal=18.0, p_risk=0.30, seeds=range(400)):
    refuses = 0
    for s in seeds:
        ag = Agent(world_coupling=True, existence_coupling=ec, seed=s)
        if ag.a1_wont_be_bought_off(goal, p_risk):
            refuses += 1
    n = len(list(seeds))
    return round(refuses / n, 3)


def sweep():
    print("=== SEED-30: one cost, two couplings -- H-one ablation (docs/36 S6) ===")
    print("reading: world coupling -> understanding; existence coupling -> subject (A1).")
    print(f"{'world':<6} {'exist':<6} {'bellErr':<8} {'surv':<7} {'follow':<7} {'A1_refuse':<10} reading")
    out = []
    for wc in (True, False):
        for ec in (True, False):
            c = run_case(wc, ec)
            a1 = a1_probe(ec)
            tag = ""
            if wc and ec:
                tag = "the target: subject + understanding together"
            elif not wc and ec:
                tag = "world out -> understanding collapses, subject remains"
            elif wc and not ec:
                tag = "existence out -> subject collapses, prediction remains"
            else:
                tag = "both out -> nothing"
            out.append({"world_coupling": wc, "existence_coupling": ec, **c,
                        "A1_refuse": a1, "reading": tag})
            print(f"{str(wc):<6} {str(ec):<6} {c['bel_err']:<8.3f} {c['surv_rate']:<7.3f} "
                  f"{c['follow_rate']:<7.3f} {a1:<10.3f} {tag}")
    return out


def main():
    p = argparse.ArgumentParser(description="SEED-30 one cost, two couplings")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--d", type=float, default=0.8)
    p.add_argument("--v", type=float, default=0.7)
    p.add_argument("--wc", type=int, choices=[0, 1], default=1)
    p.add_argument("--ec", type=int, choices=[0, 1], default=1)
    args = p.parse_args()
    if args.sweep:
        out = sweep()
        with open("seed-30/results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("\nfull results -> seed-30/results.json")
        return
    ag = Agent(bool(args.wc), bool(args.ec), d=args.d, v=args.v, seed=args.seed).run()
    err, follow_ = understanding(ag)
    a1 = ag.a1_wont_be_bought_off(18.0)
    print(json.dumps({
        "wc": bool(args.wc), "ec": bool(args.ec), "seed": args.seed,
        "alive": not ag.dead, "energy": round(ag.energy, 1),
        "bel_err": round(err, 3), "followed_shift": follow_,
        "A1_refuse": a1, "identity_len": len(ag.history),
        "saved": "None (no checkpoint under existence-on)" if ag.ec else "snapshot exists",
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
