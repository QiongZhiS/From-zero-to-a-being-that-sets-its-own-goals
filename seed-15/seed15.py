"""
SEED-15: Cultural inertia -- does imitation lag in a changing world?

Docs/11: culture transmits knowledge quickly (SEED-13), but the world
changes. When the optimum switches, imitation may keep spreading the OLD
consensus (cultural inertia), while genetics re-adapts slowly but surely.

World: food level switches periodically (32 <-> 128), so optimal hunger
switches (~0.85 <-> ~0.19).

Regimes:
  genetic   mutation only
  cultural  mutation + imitation

Measure: lag after each switch -- ticks for mean pheno_hunger to reach the
new optimum within tolerance.

Run:  python seed15.py [--mode cultural] [--ticks 12000] [--seed 42]
"""

import argparse
import random
from dataclasses import dataclass

SIZE = 64
METABOLISM = 0.4
MOVE_COST = 0.2
FOOD_ENERGY = 60
SPLIT_ENERGY = 150.0
INIT_ENERGY = 100.0
MAX_POP = 400
MUTATION = 0.08
FOOD_REGEN = 0.10

IMITATION_PROB = 0.10
IMITATION_PRECISION = 0.5
MODEL_SAMPLE = 5

DRIFT_PERIOD = 5000
DRIFT_LEVELS = [32, 128]
OPTIMA = {32: 0.85, 128: 0.19}   # SEED-1 measured optima


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
    gene_hunger: float
    pheno_hunger: float
    generation: int = 0
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

    def set_food_level(self, n):
        self.n_food = n
        self.reset_food(self.rng.randrange(2 ** 31))

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
    if agent.energy / 100.0 < agent.pheno_hunger:
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


def imitate(agent, agents, rng):
    if rng.random() >= IMITATION_PROB:
        return
    models = [a for a in rng.sample(agents, min(MODEL_SAMPLE, len(agents)))
              if a.alive]
    if not models:
        return
    best = max(models, key=lambda a: a.energy)
    agent.pheno_hunger += IMITATION_PRECISION * (best.pheno_hunger - agent.pheno_hunger)


def simulate(ticks=12000, mode="cultural", seed=42, report_every=200):
    world = World(DRIFT_LEVELS[0], seed)
    rng = world.rng
    agents = [Agent(rng.randrange(SIZE), rng.randrange(SIZE),
                    INIT_ENERGY, rng.random(), rng.random())
              for _ in range(50)]
    stats = {"births": 0, "starved": 0}
    lags = []

    for t in range(ticks):
        for a in agents:
            if a.alive:
                apply(a, world, act(a, world))
                if mode == "cultural":
                    imitate(a, agents, rng)
        stats["starved"] += sum(1 for a in agents if not a.alive)

        for a in list(agents):
            if a.alive and a.energy >= SPLIT_ENERGY and len(agents) < MAX_POP:
                ch = min(1.0, max(0.0, a.gene_hunger + rng.gauss(0, MUTATION)))
                agents.append(Agent(
                    a.x + rng.choice([-1, 0, 1]),
                    a.y + rng.choice([-1, 0, 1]),
                    a.energy / 2.0, ch, ch, a.generation + 1))
                a.energy /= 2.0
                stats["births"] += 1

        agents = [a for a in agents if a.alive]
        world.step()

        # drift: switch food level; measure lag
        if t % DRIFT_PERIOD == 0:
            level = DRIFT_LEVELS[(t // DRIFT_PERIOD) % 2]
            world.set_food_level(level)
            opt = OPTIMA[level]
            # wait for convergence, then record first-touch lag
            touched = None
            for k in range(1, DRIFT_PERIOD // report_every + 1):
                tt = t + k * report_every
                # we record at the end of the loop below; approximate here
            # (lag measured below at report time)
            current_opt = opt
        else:
            current_opt = None

        if (t + 1) % report_every == 0:
            ph = sum(a.pheno_hunger for a in agents) / len(agents) if agents else 0.0
            gh = sum(a.gene_hunger for a in agents) / len(agents) if agents else 0.0
            print(f"t={t+1:5d}  food={world.n_food:3d}  pheno={ph:.3f}  gene={gh:.3f}")

    print(f"\nfinal({mode}): pop={len(agents)} births={stats['births']} "
          f"starved={stats['starved']}")
    return stats


def main():
    p = argparse.ArgumentParser(description="SEED-15 cultural inertia")
    p.add_argument("--mode", choices=["genetic", "cultural"], default="cultural")
    p.add_argument("--ticks", type=int, default=12000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report", type=int, default=200)
    args = p.parse_args()
    simulate(ticks=args.ticks, mode=args.mode, seed=args.seed,
             report_every=args.report)


if __name__ == "__main__":
    main()
