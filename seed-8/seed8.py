"""
SEED-8: Prediction-driven strategy switch -- the USE of event boundaries

SEED-7 lesson: segmenting (clearing memory) was never selected, because it
throws information away. The real use of an event boundary is a BEHAVIOR
switch, not a memory operation.

Here: the agent predicts that its current region is drying up -- the interval
between food hits is stretching well beyond its own history. When predicted,
it switches from "keep foraging here" to "migrate to a new region" BEFORE
starvation forces it. The switch is the event; the memory is kept intact.

Heritable: prediction_sensitivity -- how much interval stretching counts as
"this region is dying". 1.0 = leave when interval exceeds its own average;
0.3 = paranoid (leave early); 3.0 = stoic (leave late).

Regimes (--ps):
  fixed0   sensitivity huge   (never predict, SEED-6 style passive)
  evolve   sensitivity evolves
  low      sensitivity 0.3    (always fleeing, misreads noise)

World: clustered with relocation + slow regen (regions genuinely dry up).

Run:  python seed8.py [--ps evolve] [--ticks 6000] [--shift 1500]
"""

import argparse
import random
from collections import deque
from dataclasses import dataclass, field

SIZE = 64
METABOLISM = 0.4
MOVE_COST = 0.2
FOOD_ENERGY = 60
SPLIT_ENERGY = 100.0
MAX_POP = 400
MUTATION = 0.08
FOOD_REGEN = 0.10
VISION = 4
MEMORY_CAP = 8
MEMORY_MAX_AGE = 400
CONF_DECAY = 0.997
N_CLUSTERS = 8
CLUSTER_RADIUS = 6


@dataclass
class Food:
    x: int
    y: int
    energy: int = FOOD_ENERGY
    alive: bool = True


@dataclass
class Mem:
    x: int
    y: int
    age: int = 0
    confirmed: float = 1.0


@dataclass
class Agent:
    x: int
    y: int
    energy: float
    hunger: float
    prediction_sensitivity: float = 1.0
    generation: int = 0
    alive: bool = True
    memory: list = field(default_factory=list)
    hit_intervals: deque = field(default_factory=lambda: deque(maxlen=4))
    since_hit: int = 999
    migrating: bool = False
    migrations: int = 0
    visited: dict = field(default_factory=dict)


class World:
    def __init__(self, n_food=64, seed=42, shift_every=1500, regen=0.10):
        self.n_food = n_food
        self.shift_every = shift_every
        self.regen = regen
        self.rng = random.Random(seed)
        self.reset_food(seed)

    def reset_food(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        self.foods = []
        self._place_clusters()

    def _place_clusters(self):
        per = max(1, self.n_food // N_CLUSTERS)
        for _ in range(N_CLUSTERS):
            cx = self.rng.randrange(SIZE)
            cy = self.rng.randrange(SIZE)
            for _ in range(per):
                x = (cx + int(self.rng.gauss(0, CLUSTER_RADIUS))) % SIZE
                y = (cy + int(self.rng.gauss(0, CLUSTER_RADIUS))) % SIZE
                self.foods.append(Food(x, y))

    def maybe_shift(self, t):
        if self.shift_every and t % self.shift_every == 0:
            self.foods = []
            self._place_clusters()

    def step(self):
        for f in self.foods:
            if not f.alive and self.rng.random() < self.regen:
                f.alive = True
                f.energy = FOOD_ENERGY

    def food_at(self, x, y):
        for f in self.foods:
            if f.alive and f.x == x and f.y == y:
                return f
        return None

    def visible_foods(self, x, y):
        out = []
        for f in self.foods:
            if f.alive and max(abs(f.x - x), abs(f.y - y)) <= VISION:
                out.append(f)
        return out


MOVES = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}


def move_toward(a, tx, ty):
    dx = (tx - a.x + SIZE // 2) % SIZE - SIZE // 2
    dy = (ty - a.y + SIZE // 2) % SIZE - SIZE // 2
    if abs(dx) >= abs(dy):
        return "E" if dx > 0 else "W"
    return "S" if dy > 0 else "N"


def torus_d2(a, tx, ty):
    dx = (tx - a.x + SIZE // 2) % SIZE - SIZE // 2
    dy = (ty - a.y + SIZE // 2) % SIZE - SIZE // 2
    return dx * dx + dy * dy


def remember(agent, food):
    for m in agent.memory:
        if m.x == food.x and m.y == food.y:
            m.age = 0
            m.confirmed = max(m.confirmed, 1.0)
            return
    agent.memory.append(Mem(food.x, food.y))
    if len(agent.memory) > MEMORY_CAP:
        agent.memory.sort(key=lambda m: -m.confirmed)
        agent.memory = agent.memory[:MEMORY_CAP]


def decay_memory(agent):
    for m in list(agent.memory):
        m.age += 1
        m.confirmed *= CONF_DECAY
        if m.age > MEMORY_MAX_AGE or m.confirmed < 0.05:
            agent.memory.remove(m)


def predict_drought(agent):
    """Event detector: is the region drying up? Interval since last food
    stretched beyond own history x sensitivity."""
    if not agent.hit_intervals:
        return False
    avg = sum(agent.hit_intervals) / len(agent.hit_intervals)
    return agent.since_hit > agent.prediction_sensitivity * avg


def least_known_direction(agent, world):
    scored = []
    for d, (dx, dy) in MOVES.items():
        nx = (agent.x + dx) % SIZE
        ny = (agent.y + dy) % SIZE
        scored.append((agent.visited.get((nx // 4, ny // 4), 0), d))
    scored.sort()
    ties = [d for c, d in scored if c == scored[0][0]]
    return world.rng.choice(ties)


def step_prey(agent, world):
    decay_memory(agent)
    agent.since_hit += 1

    # ---- event detection: switch strategy, do NOT erase memory ----
    if not agent.migrating and predict_drought(agent):
        agent.migrating = True
        agent.migrations += 1

    # ---- act ----
    f = world.food_at(agent.x, agent.y)
    if f is not None:
        agent.energy += f.energy
        f.alive = False
        agent.hit_intervals.append(agent.since_hit)
        agent.since_hit = 0
        agent.migrating = False          # region paid off again
        remember(agent, f)
        action = None
    elif agent.migrating:
        # migrate: head to least-known area (explore new regions), keep memory
        vis = world.visible_foods(agent.x, agent.y)
        if vis:
            for v in vis:
                remember(agent, v)
            target = min(vis, key=lambda v: (v.x - agent.x) ** 2 + (v.y - agent.y) ** 2)
            action = move_toward(agent, target.x, target.y)
        else:
            action = least_known_direction(agent, world)
    elif agent.energy / 100.0 < agent.hunger:
        # passive foraging (SEED-6 logic)
        vis = world.visible_foods(agent.x, agent.y)
        if vis:
            for v in vis:
                remember(agent, v)
            target = min(vis, key=lambda v: (v.x - agent.x) ** 2 + (v.y - agent.y) ** 2)
            action = move_toward(agent, target.x, target.y)
        else:
            trusted = [m for m in agent.memory if m.confirmed >= 1.0]
            if trusted:
                target = max(trusted, key=lambda m: m.confirmed - m.age / 400.0)
                action = move_toward(agent, target.x, target.y)
                if torus_d2(agent, target.x, target.y) <= 1:
                    if world.food_at(target.x, target.y) is not None:
                        target.confirmed += 1.0
                    else:
                        agent.memory.remove(target)
            elif agent.memory:
                if world.rng.random() < 0.65:
                    target = min(agent.memory, key=lambda m: torus_d2(agent, m.x, m.y))
                    action = move_toward(agent, target.x, target.y)
                    if torus_d2(agent, target.x, target.y) <= 1:
                        if world.food_at(target.x, target.y) is not None:
                            target.confirmed += 1.0
                        else:
                            agent.memory.remove(target)
                else:
                    action = world.rng.choice(["N", "S", "E", "W"])
            else:
                action = world.rng.choice(["N", "S", "E", "W"])
    else:
        # healthy, not migrating: idle wander
        action = world.rng.choice(["N", "S", "E", "W"])

    if action is not None:
        dx, dy = MOVES[action]
        agent.x = (agent.x + dx) % SIZE
        agent.y = (agent.y + dy) % SIZE
        agent.energy -= MOVE_COST
        c = agent.visited.get((agent.x // 4, agent.y // 4), 0)
        agent.visited[(agent.x // 4, agent.y // 4)] = c + 1
    agent.energy -= METABOLISM
    if agent.energy <= 0.0:
        agent.alive = False


def simulate(ticks=6000, n_food=64, ps_mode="evolve", seed=42,
             report_every=1200, shift_every=1500, regen=0.10):
    world = World(n_food, seed, shift_every, regen)
    rng = world.rng
    if ps_mode == "fixed0":
        ps_init = lambda: 999.0
    elif ps_mode == "low":
        ps_init = lambda: 0.3
    else:
        ps_init = lambda: rng.uniform(0.2, 3.0)
    preys = [Agent(rng.randrange(SIZE), rng.randrange(SIZE),
                   100.0, rng.random(), ps_init())
             for _ in range(50)]
    stats = {"births": 0, "starved": 0}

    for t in range(ticks):
        for a in preys:
            if a.alive:
                step_prey(a, world)
        stats["starved"] += sum(1 for a in preys if not a.alive)

        for a in list(preys):
            if a.alive and a.energy >= SPLIT_ENERGY and len(preys) < MAX_POP:
                preys.append(Agent(
                    a.x + rng.choice([-1, 0, 1]),
                    a.y + rng.choice([-1, 0, 1]),
                    a.energy / 2.0,
                    min(1.0, max(0.0, a.hunger + rng.gauss(0, MUTATION))),
                    min(3.0, max(0.2, a.prediction_sensitivity + rng.gauss(0, 0.15))),
                    a.generation + 1))
                a.energy /= 2.0
                stats["births"] += 1

        preys = [a for a in preys if a.alive]
        world.step()
        world.maybe_shift(t)

        if (t + 1) % report_every == 0:
            ps = sum(a.prediction_sensitivity for a in preys) / len(preys) if preys else 0.0
            print(f"t={t+1:5d}  pop={len(preys):4d}  ps={ps:.3f}")

    print(f"\nfinal(ps={ps_mode}): pop={len(preys)} births={stats['births']} "
          f"starved={stats['starved']}")
    if preys:
        vs = sorted(a.prediction_sensitivity for a in preys)
        print(f"  sensitivity: min={vs[0]:.3f} med={vs[len(vs)//2]:.3f} "
              f"mean={sum(vs)/len(vs):.3f}")
        print(f"  migrations total={sum(a.migrations for a in preys)}")


def main():
    p = argparse.ArgumentParser(description="SEED-8 prediction-driven switch")
    p.add_argument("--ticks", type=int, default=6000)
    p.add_argument("--n-food", type=int, default=64)
    p.add_argument("--ps", choices=["fixed0", "evolve", "low"], default="evolve")
    p.add_argument("--shift", type=int, default=1500)
    p.add_argument("--regen", type=float, default=0.10,
                   help="food regen probability (0 = irreversible depletion)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report", type=int, default=1200)
    args = p.parse_args()
    simulate(ticks=args.ticks, n_food=args.n_food, ps_mode=args.ps,
             seed=args.seed, report_every=args.report, shift_every=args.shift,
             regen=args.regen)


if __name__ == "__main__":
    main()
