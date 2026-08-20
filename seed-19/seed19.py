"""
SEED-19: Individual continuity -- the "one who remembers you" prototype
(docs/14 section 7).

A SINGLE agent lives for T ticks. External "advice" arrives periodically
(some correct, some wrong). The tamper hierarchy decides what enters the
endogenous layer:
  L1 temporary: advice is tried provisionally
  L3 endogenous: hunger (the agent's behavior core) -- only verified advice
                 may enter here
  L4 untouchable: the agent's own long-run experience (never overwritten)

Regimes:
  gullible   advice overwrites hunger immediately (external = tamperable)
  verifying  advice is TRIED for V_TEST ticks; adopted only if it improves
             energy vs the agent's own recent baseline (SEED-6 verify)
  stubborn   advice ignored; only own experience counts

World: SEED-13 (n_food=64, optimum hunger ~0.72). Advice alternates
correct (0.72) / wrong (0.10, starves late).

Measure: energy trajectory, hunger trajectory (how far advice pulls it),
damage from wrong advice.

Run:  python seed19.py [--mode verifying] [--ticks 8000] [--seed 42]
"""

import argparse
import random
from dataclasses import dataclass

SIZE = 64
METABOLISM = 0.4
MOVE_COST = 0.2
FOOD_ENERGY = 60
INIT_ENERGY = 150.0
FOOD_REGEN = 0.10
OPTIMUM = 0.72
WRONG = 0.10

ADVICE_PERIOD = 1000
V_TEST = 600            # verification trial length
WINDOW = 400            # baseline window for comparison (before advice)


@dataclass
class Food:
    x: int
    y: int
    energy: int = FOOD_ENERGY
    alive: bool = True


class World:
    def __init__(self, n_food=64, seed=42):
        self.rng = random.Random(seed)
        self.n_food = n_food
        self.reset_food(seed)

    def reset_food(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        self.foods = [Food(self.rng.randrange(SIZE), self.rng.randrange(SIZE))
                      for _ in range(self.n_food)]

    def step(self):
        for f in self.foods:
            if not f.alive and self.rng.random() < FOOD_REGEN:
                f.alive = True
                f.energy = FOOD_ENERGY

    def food_at(self, x, y):
        for f in self.foods:
            if f.alive and f.x == x and f.y == y:
                return f
        return None

    def nearest_food_delta(self, x, y):
        best = None
        bd = 10 ** 9
        for f in self.foods:
            if not f.alive:
                continue
            dx = (f.x - x + SIZE // 2) % SIZE - SIZE // 2
            dy = (f.y - y + SIZE // 2) % SIZE - SIZE // 2
            d = abs(dx) + abs(dy)
            if d < bd:
                bd, best = d, (dx, dy)
        return best


MOVES = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}


def act(agent, world):
    if world.food_at(agent.x, agent.y) is not None:
        return "GATHER"
    if agent.energy / 100.0 < agent.hunger:
        delta = world.nearest_food_delta(agent.x, agent.y)
        if delta is None:
            return world.rng.choice(["N", "S", "E", "W"])
        dx, dy = delta
        if abs(dx) >= abs(dy):
            return "E" if dx > 0 else "W"
        return "S" if dy > 0 else "N"
    return world.rng.choice(["N", "S", "E", "W"])


def apply(agent, world, action):
    if action in MOVES:
        dx, dy = MOVES[action]
        agent.x = (agent.x + dx) % SIZE
        agent.y = (agent.y + dy) % SIZE
        agent.energy -= MOVE_COST
    elif action == "GATHER":
        f = world.food_at(agent.x, agent.y)
        if f is not None:
            agent.energy += f.energy
            f.alive = False
    agent.energy -= METABOLISM
    if agent.energy <= 0.0:
        agent.alive = False


def advice_sequence(n):
    """Alternate correct / wrong advice."""
    return [OPTIMUM if i % 2 == 0 else WRONG for i in range(n)]


def simulate(ticks=8000, mode="verifying", seed=42, report_every=200,
             n_food=16):
    world = World(n_food, seed)
    rng = world.rng
    class A:
        pass
    agent = A()
    agent.x = rng.randrange(SIZE)
    agent.y = rng.randrange(SIZE)
    agent.energy = INIT_ENERGY
    agent.hunger = 0.5          # endogenous core (L3), starts mediocre
    agent.alive = True

    n_advice = ticks // ADVICE_PERIOD
    advices = advice_sequence(n_advice)
    hunger_log = []
    energy_log = []
    accepted = rejected = 0
    trial = None                # pending verification state
    baseline_hits = 0.0         # food-hit rate before advice
    hits = 0                    # food hits in current window/trial

    for t in range(ticks):
        # advice arrives
        if t > 0 and t % ADVICE_PERIOD == 0:
            adv = advices[t // ADVICE_PERIOD]
            if mode == "gullible":
                agent.hunger = adv
            elif mode == "verifying":
                # baseline: food-hit rate over the last WINDOW ticks
                hist = [e for tt, e in energy_log if tt > t - WINDOW]
                # reconstruct hits: energy jumps of ~+60 mark meals
                meals = sum(1 for i in range(1, len(hist))
                            if hist[i] - hist[i - 1] > 30)
                baseline_hits = meals / max(1, len(hist) - 1)
                trial = {
                    "advice": adv,
                    "old_hunger": agent.hunger,
                    "e0": agent.energy,
                    "trial_start": t,
                    "trial_hits": 0,
                }
            # stubborn: ignore

        # verification trial active?
        if trial is not None and mode == "verifying":
            el = t - trial["trial_start"]
            if el < V_TEST:
                agent.hunger = trial["advice"]      # provisional (L1)
                # count meals during trial (energy jump detection at end)
            else:
                # trial over: hit rate during trial vs baseline hit rate
                hist2 = [e for tt, e in energy_log if tt > trial["trial_start"]]
                meals2 = sum(1 for i in range(1, len(hist2))
                             if hist2[i] - hist2[i - 1] > 30)
                trial_rate = meals2 / max(1, len(hist2) - 1)
                if trial_rate > baseline_hits:
                    accepted += 1
                else:
                    rejected += 1
                    agent.hunger = trial["old_hunger"]   # reject, revert
                trial = None

        # live
        if agent.alive:
            apply(agent, world, act(agent, world))
        world.step()

        if (t + 1) % report_every == 0:
            hunger_log.append((t + 1, agent.hunger))
            energy_log.append((t + 1, agent.energy))

    print(f"\nfinal({mode}): energy={agent.energy:.1f} alive={agent.alive} "
          f"hunger={agent.hunger:.3f}")
    if mode == "verifying":
        print(f"  advice: accepted={accepted} rejected={rejected}")
    # damage: mean |hunger - optimum| over the run
    mean_dev = sum(abs(h - OPTIMUM) for _, h in hunger_log) / len(hunger_log)
    print(f"  mean |hunger - optimum| over run: {mean_dev:.3f} (optimum={OPTIMUM})")
    print(f"  final energy: {agent.energy:.1f}  "
          f"(higher = better continuity under tampering)")


def main():
    p = argparse.ArgumentParser(description="SEED-19 individual continuity")
    p.add_argument("--mode", choices=["gullible", "verifying", "stubborn"],
                   default="verifying")
    p.add_argument("--ticks", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-food", type=int, default=16)
    args = p.parse_args()
    simulate(ticks=args.ticks, mode=args.mode, seed=args.seed,
             n_food=args.n_food)


if __name__ == "__main__":
    main()
