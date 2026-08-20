"""
SEED-9: Boldness -- acting under uncertainty (the courage layer)

SEED-8 lesson was NOT only perception: even with a signal, acting on it is a
separate decision. In reality signals are always fuzzy. Courage = the
willingness to act on a fuzzy signal. Heritable boldness (0..1) is the
probability of leaving when the region LOOKS like it is drying up.

Two independent layers:
  1. perception: smoother signal -- recent 50-tick food hit rate vs lifetime
     average (less noisy than single intervals, SEED-8's flaw)
  2. courage: boldness -- probability of acting on the fuzzy signal

Courage pays off only when destinations differ in quality (clustered world
with slow regen: clusters genuinely dry up, others still hold food).

Regimes (--b):
  timid    boldness 0     (only move when starving, SEED-6 style)
  evolve   boldness evolves
  bold     boldness 1     (always flee on any decline signal)

Run:  python seed9.py [--b evolve] [--ticks 8000] [--regen 0.02]
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
FOOD_REGEN = 0.05      # moderate regen: clusters dry up, others still hold
VISION = 4
MEMORY_CAP = 8
MEMORY_MAX_AGE = 400
CONF_DECAY = 0.997
N_CLUSTERS = 8
CLUSTER_RADIUS = 6
WIN = 50                # perception window


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
    boldness: float = 0.0
    generation: int = 0
    alive: bool = True
    memory: list = field(default_factory=list)
    hit_win: deque = field(default_factory=lambda: deque(maxlen=WIN))
    total_hits: int = 0
    age: int = 0
    migrating: bool = False
    migrations: int = 0
    visited: dict = field(default_factory=dict)


class World:
    def __init__(self, n_food=64, seed=42, regen=FOOD_REGEN):
        self.n_food = n_food
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


def recent_rate(agent):
    if len(agent.hit_win) < 20:
        return None
    return sum(agent.hit_win) / len(agent.hit_win)


def lifetime_rate(agent):
    return agent.total_hits / max(1, agent.age)


def dry_signal(agent):
    """Fuzzy perception: recent hit rate clearly below lifetime average."""
    r = recent_rate(agent)
    if r is None:
        return False
    lr = lifetime_rate(agent)
    return lr > 0.005 and r < 0.4 * lr


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
    agent.age += 1

    f = world.food_at(agent.x, agent.y)
    if f is not None:
        agent.energy += f.energy
        f.alive = False
        agent.total_hits += 1
        agent.hit_win.append(1)
        agent.migrating = False
        remember(agent, f)
        action = None
    else:
        agent.hit_win.append(0)
        # COURAGE: act on the fuzzy dry signal (only when healthy)
        if (not agent.migrating and not agent.energy / 100.0 < agent.hunger
                and dry_signal(agent) and world.rng.random() < agent.boldness):
            agent.migrating = True
            agent.migrations += 1
        # act
        if agent.migrating:
            vis = world.visible_foods(agent.x, agent.y)
            if vis:
                for v in vis:
                    remember(agent, v)
                target = min(vis, key=lambda v: (v.x - agent.x) ** 2 + (v.y - agent.y) ** 2)
                action = move_toward(agent, target.x, target.y)
            else:
                action = least_known_direction(agent, world)
        elif agent.energy / 100.0 < agent.hunger:
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


def simulate(ticks=8000, n_food=64, b_mode="evolve", seed=42,
             report_every=1600, regen=FOOD_REGEN):
    world = World(n_food, seed, regen)
    rng = world.rng
    if b_mode == "timid":
        b_init = lambda: 0.0
    elif b_mode == "bold":
        b_init = lambda: 1.0
    else:
        b_init = rng.random
    preys = [Agent(rng.randrange(SIZE), rng.randrange(SIZE),
                   100.0, rng.random(), b_init())
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
                    min(1.0, max(0.0, a.boldness + rng.gauss(0, MUTATION))),
                    a.generation + 1))
                a.energy /= 2.0
                stats["births"] += 1

        preys = [a for a in preys if a.alive]
        world.step()

        if (t + 1) % report_every == 0:
            b = sum(a.boldness for a in preys) / len(preys) if preys else 0.0
            print(f"t={t+1:5d}  pop={len(preys):4d}  boldness={b:.3f}")

    print(f"\nfinal(b={b_mode}): pop={len(preys)} births={stats['births']} "
          f"starved={stats['starved']}")
    if preys:
        bs = sorted(a.boldness for a in preys)
        print(f"  boldness: min={bs[0]:.3f} med={bs[len(bs)//2]:.3f} "
              f"mean={sum(bs)/len(bs):.3f}")
        print(f"  migrations total={sum(a.migrations for a in preys)}")


def main():
    p = argparse.ArgumentParser(description="SEED-9 boldness under uncertainty")
    p.add_argument("--ticks", type=int, default=8000)
    p.add_argument("--n-food", type=int, default=64)
    p.add_argument("--b", choices=["timid", "evolve", "bold"], default="evolve")
    p.add_argument("--regen", type=float, default=FOOD_REGEN)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report", type=int, default=1600)
    args = p.parse_args()
    simulate(ticks=args.ticks, n_food=args.n_food, b_mode=args.b,
             seed=args.seed, report_every=args.report, regen=args.regen)


if __name__ == "__main__":
    main()
