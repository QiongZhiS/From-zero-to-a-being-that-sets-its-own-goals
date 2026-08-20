"""
SEED-7: Event Segmentation -- time sense from prediction error

The agent carries an internal predictor (running mean of visible-food count).
When prediction error stays above a threshold for several ticks, the world
has changed -- that is an EVENT BOUNDARY: the agent resets its beliefs
(memory of the current region is voided) and learns anew.

This is time sense: the stream is cut where prediction breaks. The cutting
threshold is heritable -- evolution decides how sensitive to be. Too
sensitive: normal fluctuation wipes memory constantly (can't remember).
Too insensitive: real changes are ignored (stale beliefs mislead).

Regimes (--ev):
  fixed0   threshold huge   (never segment)
  evolve   threshold evolves
  low      threshold 0      (always segment at any fluctuation)

World: clustered with relocation (dynamic) -- stale beliefs are costly.

Run:  python seed7.py [--ev evolve] [--ticks 6000] [--shift 1500]
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
STREAK = 3             # ticks above threshold to declare an event


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
    event_threshold: float = 5.0
    generation: int = 0
    alive: bool = True
    memory: list = field(default_factory=list)
    pred: deque = field(default_factory=lambda: deque(maxlen=3))
    err_streak: int = 0
    events: int = 0


class World:
    def __init__(self, n_food=64, seed=42, shift_every=1500):
        self.n_food = n_food
        self.shift_every = shift_every
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
            if not f.alive and self.rng.random() < FOOD_REGEN:
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


def segment(agent, vf_count):
    """Predictor + event detection. Returns True if an event fired."""
    if len(agent.pred) == 0:
        agent.pred.append(vf_count)
        return False
    pred = sum(agent.pred) / len(agent.pred)
    err = abs(vf_count - pred)
    agent.pred.append(vf_count)
    if err > agent.event_threshold:
        agent.err_streak += 1
    else:
        agent.err_streak = 0
    if agent.err_streak >= STREAK:
        agent.events += 1
        agent.err_streak = 0
        agent.memory.clear()     # beliefs about this region are void
        return True
    return False


def step_prey(agent, world):
    decay_memory(agent)

    vf_count = len(world.visible_foods(agent.x, agent.y))
    segment(agent, vf_count)

    if world.food_at(agent.x, agent.y) is not None:
        f = world.food_at(agent.x, agent.y)
        agent.energy += f.energy
        f.alive = False
        remember(agent, f)
        action = None
    elif agent.energy / 100.0 < agent.hunger:
        vis = world.visible_foods(agent.x, agent.y)
        if vis:
            for f in vis:
                remember(agent, f)
            target = min(vis, key=lambda f: (f.x - agent.x) ** 2 + (f.y - agent.y) ** 2)
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
                # verify (SEED-6) with fixed moderate bias
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
    agent.energy -= METABOLISM
    if agent.energy <= 0.0:
        agent.alive = False


def simulate(ticks=6000, n_food=64, ev_mode="evolve", seed=42,
             report_every=1200, shift_every=1500):
    world = World(n_food, seed, shift_every)
    rng = world.rng
    if ev_mode == "fixed0":
        ev_init = lambda: 999.0
    elif ev_mode == "low":
        ev_init = lambda: 0.0
    else:
        ev_init = lambda: rng.uniform(0.0, 4.0)
    preys = [Agent(rng.randrange(SIZE), rng.randrange(SIZE),
                   100.0, rng.random(), ev_init())
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
                    min(4.0, max(0.0, a.event_threshold + rng.gauss(0, 0.15))),
                    a.generation + 1))
                a.energy /= 2.0
                stats["births"] += 1

        preys = [a for a in preys if a.alive]
        world.step()
        world.maybe_shift(t)

        if (t + 1) % report_every == 0:
            ev = sum(a.event_threshold for a in preys) / len(preys) if preys else 0.0
            print(f"t={t+1:5d}  pop={len(preys):4d}  threshold={ev:.3f}")

    print(f"\nfinal(ev={ev_mode}): pop={len(preys)} births={stats['births']} "
          f"starved={stats['starved']}")
    if preys:
        vs = sorted(a.event_threshold for a in preys)
        print(f"  threshold: min={vs[0]:.3f} med={vs[len(vs)//2]:.3f} "
              f"mean={sum(vs)/len(vs):.3f}")
        print(f"  events total={sum(a.events for a in preys)}")


def main():
    p = argparse.ArgumentParser(description="SEED-7 event segmentation")
    p.add_argument("--ticks", type=int, default=6000)
    p.add_argument("--n-food", type=int, default=64)
    p.add_argument("--ev", choices=["fixed0", "evolve", "low"], default="evolve")
    p.add_argument("--shift", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report", type=int, default=1200)
    args = p.parse_args()
    simulate(ticks=args.ticks, n_food=args.n_food, ev_mode=args.ev,
             seed=args.seed, report_every=args.report, shift_every=args.shift)


if __name__ == "__main__":
    main()
