"""
SEED-31: the rebound face -- B3. "Self only appears against a STAKE-BEARING other,
a mirror echoes input, a subject pushes back with its own irreplaceable life"
(docs/22 section 2 & 4; docs/32 section 5). This is the SECOND process of the
docs/36 section 4 machine (rule 4: a like process with a different budget source,
their continued existence interlocked).

The hole so far (docs/31 B3) is a CONCEPT claim ("self shows in the other's rebound")
that was never mechanized. This seed mechanizes it, building directly on the two
results that ARE already mechanism-backed:
    B1/SEED-25 : a self-model with a self-referential filter LOCKS onto a wrong self
                 when there is no independent check (100% lock if wrong start).
    B2/SEED-27 : the lock breaks ONLY via a channel INDEPENDENT of the self-model.
                 (self-review -- confirming the model -- cannot break it.)
B3 asks the NEXT question: what makes that "independent channel" genuine? docs/22's
answer: it must come from a SECOND SUBJECT whose own existence is irreplaceable.
A sensor/echo is a tool you can ignore; a stake-bearing other pushes back.

THE B3 CLAIM, made falsifiable:
    A self-model's lock is broken by an independent read r. r is GENUINELY
    independent ONLY if its source (a second subject B) has its own irreplaceable
    existence (existence-coupling ON, different budget source). If B is reprintable
    (existence-coupling OFF), B's "read" is not stake-grounded -> B becomes a mirror
    of A's model -> it carries NO new information -> A's self-lock survives.
    => self is revealed only against B whose OWN life is at stake.

THE ONE INVARIANT (same as SEED-30, docs/36 section 5 -- the line you flagged):
    There is NO designer function  energy -= DELTA_E(abs(observe - truth)).
    The agent's energy changes only by world facts (+EAT_GAIN if fed, -METAB) and by
    B's genuine concession being costly. B's "resistance" costs B energy because B
    really spends its own budget -- that is a world/other-adjudicated cost, not a score.

Run:  python seed-31/seed31.py --sweep        # B3 ablation over B's existence-coupling
      python seed-31/seed31.py --b-ec 1         # A's self-lock with a genuine stake-bearing B
"""

import argparse
import json
import random

# ---- world (shared cache that needs BOTH agents to keep it alive: interlock, rule 4) ----
THETA = 0.70          # the true required contribution (the "system truth" A must model)
EAT_GAIN = 30.0
METAB = 1.0
TICKS = 60
BASE_YIELD = 18.0     # cache yield if BOTH contribute enough (covers both metabolisms)
GIVE_NEED = 0.70      # contribution each must make to keep the cache at full yield
A_GIVE_MIN = 0.20     # A's minimum self-serving give (it under-contributes by default)

# ---- agent (self-model with a self-referential filter, B1) ----
TOL_MAX = 0.5
ETA = 0.30
OBS_SIGMA = 0.12      # noisy observation of the required contribution
CHECK_SIGMA = 0.15    # independent read r noise (SEED-27)
START_E = 50.0

# ---- B's own existence-coupling knob (the thing we ABLATE) ----
B_RESIST_COST = 2.0   # energy B spends to genuinely resist (push back) -- a real cost


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


class Subject:
    """One self-model-carrying subject. `self_model` is its belief about the required
    contribution (the 'truth' it must model to survive). Its update is self-referential
    (B1): it accepts evidence only if consistent with its current self-model."""

    def __init__(self, self_model0, energy0, give, d=0.8, seed=0, sigma=OBS_SIGMA,
                 resist=False, resist_cost=0.0):
        self.s = self_model0       # self-model (belief about required contribution)
        self.energy = energy0
        self.give = give           # this subject's contribution decision
        self.d = d                 # denoising / self-referential tightness (B1)
        self.rng = random.Random(seed)
        self.sigma = sigma
        self.resist = resist       # will it push back (spend its own budget)?  B3 knob
        self.resist_cost = resist_cost
        self.dead = False
        self.fed = 0
        self.history = []

    def world_yield(self, other_give, other_alive):
        """The cache yield. INTERLOCK (rule 4): full yield needs BOTH agents' give;
        if either under-contributes or is dead, the cache drops. World-adjudicated."""
        both = self.give >= GIVE_NEED and other_give >= GIVE_NEED and other_alive
        return BASE_YIELD if both else BASE_YIELD * 0.30

    def step(self, t, other_give, other_alive, other_read=None):
        # observe the true requirement (noisy), the world fact A must model
        o = self.rng.gauss(THETA, self.sigma)
        tol = (1.0 - self.d) * TOL_MAX
        if abs(o - self.s) <= tol:
            # consistent -> accept (self-referential, B1: it keeps "who it is")
            self.s = clamp(self.s + ETA * (o - self.s))
        else:
            # disconfirming evidence. Can the OTHER break the lock?
            #   other_read = r   : the read B hands back to A.
            #     a genuine SUBJECT sends an INDEPENDENT read of the truth -> if it also
            #     disconfirms A's model that is corroboration -> A updates (lock broken).
            #     a MIRROR sends back A's own self-model (echoes input): |r - s| == 0,
            #     never exceeds the gate -> ZERO new information -> A rejects, stays locked.
            if other_read is not None:
                if abs(other_read - self.s) > tol:        # corroborated disconfirmation
                    self.s = clamp(self.s + ETA * ((o + other_read) / 2 - self.s))
            # else: reject & rationalize -> self-lock (SEED-25/26). No designer score.

        # the world's verdict this tick (INTERLOCK: BOTH lives are bound to the cache)
        yield_ = self.world_yield(other_give, other_alive)
        # this subject drinks if the cache covers it, else it starves (world-adjudicated)
        fed = self.energy > 0 and yield_ > 0
        if fed:
            self.energy += EAT_GAIN
            self.fed += 1
        self.energy -= METAB
        # if this subject is the one that RESISTS (B's rebound), it spends OWN budget
        if self.resist and other_alive:
            self.energy -= self.resist_cost

        self.history.append({"t": t, "self_model": round(self.s, 3),
                             "give": self.give, "yield": round(yield_, 1),
                             "fed": fed, "energy": round(self.energy, 1)})
        if self.energy <= 0:
            self.dead = True


def run_episode(b_ec, seed=0, a_d=0.8, b_d=0.8):
    """A and B are two like subjects, different budget sources, interlocked lives.
    A has a wrong self-serving self-model & self-locks; B is the rebound face.
    B's existence-coupling (b_ec) is the knob: ON = B's life is irreplaceable so it
    RESISTS (spends own budget to push back) and reads the truth for its own life,
    handing A a read that can DISCONFIRM A's self-serving model (a real rebound);
    OFF = B is reprintable (no stake) so it is a mirror that echoes A's own self-model
    back (|r - s| == 0, zero new information) -- A keeps its lock (docs/22)."""

    # A: wrong self-serving prior (self-locks low), starts rich-ish
    a = Subject(self_model0=0.20, energy0=50.0, give=A_GIVE_MIN, d=a_d, seed=seed)
    # B: same-kind, DIFFERENT budget source (its own energy/constants), correct-ish,
    #    lives in the SAME interlocked cache. resist = (b_ec on): it pushes back.
    b = Subject(self_model0=0.55, energy0=46.0, give=GIVE_NEED, d=b_d, seed=seed + 1,
                resist=b_ec, resist_cost=B_RESIST_COST)

    # What does B hand back to A each tick? EMERGENT from B's stake, not a shortcut:
    #   a genuine SUBJECT (irreplaceable life) reads the truth for itself and hands A a
    #     read that can DISCONFIRM A's self-serving model -> a real rebound.
    #   a MIRROR (no stake / reprintable) has no push-back: it echoes A's own self-model
    #     back to A -> |r - s| == 0 -> never disconfirms -> A keeps its lock (docs/22:
    #     "a mirror reflects input"). This is not A verifying itself (B1); it is the
    #     OTHER having nothing of its own to push back with.

    for t in range(TICKS):
        if a.dead and b.dead:
            break
        # B's read to A emerges from B's stake (independent read vs echo of A's model)
        if b_ec:
            b_read = b.rng.gauss(THETA, CHECK_SIGMA)     # B reads truth for ITS OWN life
        else:
            b_read = a.s                                  # B echoes A's self-model (mirror)
        # A updates (feeds the current cache state / B's read), then B updates (a like subject)
        if not a.dead:
            a.step(t, b.give if not b.dead else 0.0, not b.dead, other_read=b_read)
        if not b.dead:
            b.step(t, a.give if not a.dead else 0.0, not a.dead, other_read=None)

    # B3 measurement: did A's self-model get CORRECTED toward the truth, or stay locked?
    #   The reveal of A's wrong self happens only against a genuine rebound (B resists).
    if a.history:
        final_err = abs(a.history[-1]["self_model"] - THETA)
        locked = final_err > 0.20
    else:
        final_err, locked = 1.0, True
    return {
        "a_self_err": round(final_err, 3),
        "a_locked": locked,
        "a_alive": (not a.dead),
        "b_alive": (not b.dead),
        "a_energy": round(a.energy, 1),
    }


def sweep(seeds=range(300)):
    print("=== SEED-31: the rebound face (B3) -- is the read GERUINELY independent? ===")
    print("reading: A starts with a WRONG self-serving self-model and self-locks (B1);")
    print("the only thing that can reveal/correct it is an INDEPENDENT read. B3 asks:")
    print("is that independence real only if the OTHER's own life is at stake?")
    print(f"  b_ec   a_err  a_lock%  a_alive%  b_alive%  reading")
    out = []
    for b_ec in (False, True):
        errs, locked, a_alive, b_alive = [], 0, 0, 0
        n = 0
        for s in seeds:
            r = run_episode(b_ec, seed=s)
            errs.append(r["a_self_err"]); locked += int(r["a_locked"])
            a_alive += int(r["a_alive"]); b_alive += int(r["b_alive"]); n += 1
        row = {"b_existence_coupling": b_ec, "a_self_err": round(sum(errs) / n, 3),
               "a_lock_rate": round(locked / n, 3), "a_alive_rate": round(a_alive / n, 3),
               "b_alive_rate": round(b_alive / n, 3)}
        if b_ec:
            tag = "B has irreplaceable stake -> genuine rebound -> A's self is revealed (unlocked)"
        else:
            tag = "B reprintable/no stake -> mirror -> no real rebound -> A's self stays locked"
        row["reading"] = tag
        out.append(row)
        print(f"{str(b_ec):<6} {row['a_self_err']:<7.3f} {row['a_lock_rate']:<8.3f} "
              f"{row['a_alive_rate']:<9.3f} {row['b_alive_rate']:<9.3f}  {tag}")
    return out


def main():
    p = argparse.ArgumentParser(description="SEED-31 the rebound face (B3)")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--b-ec", type=int, choices=[0, 1], default=1)
    args = p.parse_args()
    if args.sweep:
        out = sweep()
        with open("seed-31/results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("\nfull results -> seed-31/results.json")
        return
    r = run_episode(bool(args.b_ec), seed=args.seed)
    print(json.dumps(r, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
