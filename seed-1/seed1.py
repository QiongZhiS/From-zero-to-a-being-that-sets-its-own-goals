"""
SEED-1: Reproduction & Evolution
Population of homeostatic agents with HERITABLE behavior parameters.
Natural selection does the design: limited food, split when energy is high,
die when energy runs out. Behavior (hunger threshold) evolves by mutation.

Run:  python seed1.py [--ticks 8000] [--seed 42] [--n-food 128]
"""

import argparse
import math
import random
from dataclasses import dataclass, field

SIZE = 64
METABOLISM = 0.4        # per tick, unavoidable
MOVE_COST = 0.2
FOOD_ENERGY = 60
SPLIT_ENERGY = 150.0    # split threshold (energy halves on split)
INIT_ENERGY = 100.0
MAX_POP = 400           # hard cap, prevents explosion
MUTATION = 0.08         # std of hunger mutation
FOOD_REGEN = 0.10


@dataclass
class Food:
    x: int
    y: int
    energy: int = FOOD_ENERGY
    alive: bool = True


@dataclass
class Agent:
    x: int
    y: int
    energy: float
    hunger: float        # HERITABLE: fraction of energy below which we seek food
    generation: int = 0
    alive: bool = True
    births: int = 0


class World:
    def __init__(self, n_food=128, seed=42):
        self.n_food = n_food
        self.rng = random.Random(seed)
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


def act(agent, world):
    """Policy parameterized ONLY by the heritable 'hunger' threshold.
    hunger=1.0 -> always seek food; hunger=0.0 -> never seek (wander)."""
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
    # healthy: wander (exploration)
    return world.rng.choice(["N", "S", "E", "W"])


MOVES = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}


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
            agent.births += 0  # placeholder
            f.alive = False
    agent.energy -= METABOLISM
    if agent.energy <= 0.0:
        agent.alive = False


def simulate(ticks=8000, n_food=128, seed=42, report_every=1000):
    world = World(n_food, seed)
    rng = world.rng
    # founding population: one agent, random hunger
    agents = [Agent(rng.randrange(SIZE), rng.randrange(SIZE),
                    INIT_ENERGY, rng.random())]
    stats = {"births": 0, "deaths": 0, "splits": 0}

    for t in range(ticks):
        # act
        for a in agents:
            if a.alive:
                apply(a, world, act(a, world))
        world.step()

        # split (reproduction) -- iterate on a copy, append children
        for a in list(agents):
            if a.alive and a.energy >= SPLIT_ENERGY and len(agents) < MAX_POP:
                child = Agent(a.x + rng.choice([-1, 0, 1]),
                              a.y + rng.choice([-1, 0, 1]),
                              a.energy / 2.0,
                              min(1.0, max(0.0, a.hunger + rng.gauss(0, MUTATION))),
                              a.generation + 1)
                a.energy /= 2.0
                agents.append(child)
                stats["splits"] += 1
                stats["births"] += 1

        # remove dead
        alive = [a for a in agents if a.alive]
        stats["deaths"] += len(agents) - len(alive)
        agents = alive

        if (t + 1) % report_every == 0:
            if agents:
                hs = [a.hunger for a in agents]
                gs = [a.generation for a in agents]
                es = [a.energy for a in agents]
                print(f"t={t+1:6d}  pop={len(agents):4d}  "
                      f"hunger={sum(hs)/len(hs):.3f}±{stdev(hs):.3f}  "
                      f"gen={sum(gs)/len(gs):.1f}  births={stats['births']:5d}  "
                      f"food_left={sum(1 for f in world.foods if f.alive):3d}")

    print(f"\nfinal: pop={len(agents)}  births={stats['births']}  "
          f"deaths={stats['deaths']}  splits={stats['splits']}")
    if agents:
        hs = sorted(a.hunger for a in agents)
        print(f"hunger distribution: min={hs[0]:.3f} med={hs[len(hs)//2]:.3f} "
              f"max={hs[-1]:.3f}  mean={sum(hs)/len(hs):.3f}")
    else:
        print("population went extinct")


def stdev(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def main():
    p = argparse.ArgumentParser(description="SEED-1 reproduction & evolution")
    p.add_argument("--ticks", type=int, default=8000)
    p.add_argument("--n-food", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report", type=int, default=1000)
    args = p.parse_args()
    simulate(ticks=args.ticks, n_food=args.n_food, seed=args.seed,
             report_every=args.report)


if __name__ == "__main__":
    main()
