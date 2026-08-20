"""
SEED-2: Predator-Prey Co-evolution

Prey: heritable 'hunger' (SEED-1 rule). Prey has NO knowledge of predators --
its only behavior parameter is hunger. Avoidance must be discovered by
evolution alone, because the predator is a physical fact of the world.

Predators: heritable 'aggression' (how eagerly to hunt). Predators also have
an energy homeostat -- they starve if they do not eat. Two species co-evolve.

Run:  python seed2.py [--ticks 4000] [--n-food 96] [--n-pred 1] [--seed 42]
      --n-pred 0  ->  SEED-1 control
"""

import argparse
import math
import random
from dataclasses import dataclass

SIZE = 64
METABOLISM = 0.4
MOVE_COST = 0.2
FOOD_ENERGY = 60
SPLIT_ENERGY = 100.0        # prey split threshold (2 food)
PRED_SPLIT_ENERGY = 180.0   # predator split threshold
PRED_MAX_ENERGY = 200.0     # satiation cap: extra food is NOT stored
PRED_CHILD_ENERGY = 70.0    # each child after split (net split cost 60)
PRED_METABOLISM = 0.6
PRED_MOVE_COST = 0.3
KILL_REWARD = 50.0
PRED_VISION = 5           # predators only perceive prey within this radius
MAX_PREY = 400
MAX_PRED = 40
MUTATION = 0.08
FOOD_REGEN = 0.10


@dataclass
class Food:
    x: int
    y: int
    energy: int = FOOD_ENERGY
    alive: bool = True


@dataclass
class Agent:
    """Prey. Only heritable parameter: hunger."""
    x: int
    y: int
    energy: float
    hunger: float
    generation: int = 0
    alive: bool = True


@dataclass
class Predator:
    """Heritable parameter: aggression (0..1) = how eagerly to hunt."""
    x: int
    y: int
    energy: float
    aggression: float
    generation: int = 0
    alive: bool = True
    kills: int = 0


class World:
    def __init__(self, n_food=96, seed=42):
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


MOVES = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}


def apply_prey(agent, world):
    """SEED-1 policy: hunger-driven seeking, otherwise wander."""
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


def do_move(a, world, action, move_cost):
    dx, dy = MOVES[action]
    a.x = (a.x + dx) % SIZE
    a.y = (a.y + dy) % SIZE
    a.energy -= move_cost


def step_prey(agent, world):
    action = apply_prey(agent, world)
    if action == "GATHER":
        f = world.food_at(agent.x, agent.y)
        if f is not None:
            agent.energy += f.energy
            f.alive = False
    else:
        do_move(agent, world, action, MOVE_COST)
    agent.energy -= METABOLISM
    if agent.energy <= 0.0:
        agent.alive = False


def step_pred(pred, world, preys):
    """Hunt: move toward nearest VISIBLE prey (vision-limited) if aggression
    says so, else wander. Predators cannot perceive the whole map -- a prey
    outside PRED_VISION is simply not there. This is a physical limit of the
    predator, exactly as prey's ignorance of predators is a physical limit."""
    visible = [a for a in preys
               if a.alive and torus_dist(pred, a, SIZE) <= PRED_VISION]
    if pred.aggression > world.rng.random() and visible:
        target = min(visible, key=lambda p: torus_dist(pred, p, SIZE))
        dx = (target.x - pred.x + SIZE // 2) % SIZE - SIZE // 2
        dy = (target.y - pred.y + SIZE // 2) % SIZE - SIZE // 2
        if abs(dx) >= abs(dy):
            action = "E" if dx > 0 else "W"
        else:
            action = "S" if dy > 0 else "N"
        do_move(pred, world, action, PRED_MOVE_COST)
    else:
        do_move(pred, world, world.rng.choice(["N", "S", "E", "W"]),
                PRED_MOVE_COST)
    pred.energy -= PRED_METABOLISM
    if pred.energy <= 0.0:
        pred.alive = False


def torus_dist(a, b, size):
    dx = (a.x - b.x + size // 2) % size - size // 2
    dy = (a.y - b.y + size // 2) % size - size // 2
    return abs(dx) + abs(dy)


def simulate(ticks=6000, n_food=96, n_pred=2, n_prey=50, seed=42,
             report_every=1200):
    world = World(n_food, seed)
    rng = world.rng
    preys = [Agent(rng.randrange(SIZE), rng.randrange(SIZE),
                   100.0, rng.random()) for _ in range(n_prey)]
    preds = [Predator(rng.randrange(SIZE), rng.randrange(SIZE),
                      100.0, rng.random()) for _ in range(n_pred)]
    stats = {"prey_births": 0, "prey_starved": 0, "prey_eaten": 0,
             "pred_births": 0, "pred_starved": 0}

    for t in range(ticks):
        # prey act (starvation deaths happen here)
        for a in preys:
            if a.alive:
                step_prey(a, world)
        n_starved = sum(1 for a in preys if not a.alive)
        stats["prey_starved"] += n_starved
        # predators hunt
        for p in preds:
            if p.alive:
                step_pred(p, world, preys)
        # predation: predator on prey cell eats it
        for p in preds:
            if not p.alive:
                continue
            for a in preys:
                if a.alive and a.x == p.x and a.y == p.y:
                    a.alive = False
                    stats["prey_eaten"] += 1
                    p.energy = min(p.energy + KILL_REWARD, PRED_MAX_ENERGY)
                    p.kills += 1
                    break  # one meal per tick
        world.step()

        # prey split
        for a in list(preys):
            if a.alive and a.energy >= SPLIT_ENERGY and len(preys) < MAX_PREY:
                preys.append(Agent(a.x + rng.choice([-1, 0, 1]),
                                   a.y + rng.choice([-1, 0, 1]),
                                   a.energy / 2.0,
                                   min(1.0, max(0.0, a.hunger + rng.gauss(0, MUTATION))),
                                   a.generation + 1))
                a.energy /= 2.0
                stats["prey_births"] += 1
        # predator split (with net reproduction cost)
        for p in list(preds):
            if p.alive and p.energy >= PRED_SPLIT_ENERGY and len(preds) < MAX_PRED:
                preds.append(Predator(p.x + rng.choice([-1, 0, 1]),
                                      p.y + rng.choice([-1, 0, 1]),
                                      PRED_CHILD_ENERGY,
                                      min(1.0, max(0.0, p.aggression + rng.gauss(0, MUTATION))),
                                      p.generation + 1))
                p.energy = PRED_CHILD_ENERGY
                stats["pred_births"] += 1

        # filter dead
        preys = [a for a in preys if a.alive]
        n_alive_pred = sum(1 for p in preds if p.alive)
        stats["pred_starved"] += len(preds) - n_alive_pred
        preds = [p for p in preds if p.alive]

        if (t + 1) % report_every == 0:
            ph = sum(a.hunger for a in preys) / len(preys) if preys else 0.0
            pa = sum(p.aggression for p in preds) / len(preds) if preds else 0.0
            print(f"t={t+1:5d}  prey={len(preys):4d}  pred={len(preds):3d}  "
                  f"hunger={ph:.3f}  aggr={pa:.3f}  eaten={stats['prey_eaten']:5d}")

    print(f"\nfinal: prey={len(preys)} pred={len(preds)}")
    print(f"  prey_births={stats['prey_births']} prey_starved={stats['prey_starved']} "
          f"prey_eaten={stats['prey_eaten']}")
    print(f"  pred_births={stats['pred_births']} pred_starved={stats['pred_starved']}")
    if preys:
        hs = sorted(a.hunger for a in preys)
        print(f"  prey hunger: min={hs[0]:.3f} med={hs[len(hs)//2]:.3f} "
              f"max={hs[-1]:.3f} mean={sum(hs)/len(hs):.3f}")
    if preds:
        as_ = sorted(p.aggression for p in preds)
        print(f"  pred aggression: min={as_[0]:.3f} med={as_[len(as_)//2]:.3f} "
              f"max={as_[-1]:.3f} mean={sum(as_)/len(as_):.3f}")


def main():
    p = argparse.ArgumentParser(description="SEED-2 predator-prey co-evolution")
    p.add_argument("--ticks", type=int, default=6000)
    p.add_argument("--n-food", type=int, default=96)
    p.add_argument("--n-pred", type=int, default=2)
    p.add_argument("--n-prey", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report", type=int, default=1200)
    args = p.parse_args()
    simulate(ticks=args.ticks, n_food=args.n_food, n_pred=args.n_pred,
             n_prey=args.n_prey, seed=args.seed, report_every=args.report)


if __name__ == "__main__":
    main()
