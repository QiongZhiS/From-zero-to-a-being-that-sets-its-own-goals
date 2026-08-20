"""
SEED-3: Cognition -- Local Perception + Memory

Prey see only VISION cells around them (reactive). They can REMEMBER food
locations (goal-directed). Memory has real costs: finite capacity, decay,
and staleness -- food may be gone when you arrive, so trusting memory can
waste energy.

Heritable: hunger (when to seek) + memory_weight (how much to trust memory).

Compare three regimes:
  --mem none     memory_weight fixed 0   (purely reactive baseline)
  --mem evolve   memory_weight evolves   (evolution discovers its value)
  --mem always   memory_weight fixed 1   (blind trust -> stale traps)

Run:  python seed3.py [--ticks 6000] [--mem evolve] [--seed 42]
"""

import argparse
import random
from dataclasses import dataclass, field

SIZE = 64
METABOLISM = 0.4
MOVE_COST = 0.2
FOOD_ENERGY = 60
SPLIT_ENERGY = 100.0
MAX_POP = 400
MUTATION = 0.08
FOOD_REGEN = 0.10
VISION = 4              # local perception radius
MEMORY_CAP = 8          # finite memory slots
MEMORY_MAX_AGE = 300    # forget after this many ticks unseen


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


@dataclass
class Agent:
    x: int
    y: int
    energy: float
    hunger: float
    memory_weight: float = 0.0
    generation: int = 0
    alive: bool = True
    memory: list = field(default_factory=list)
    goals: int = 0       # times we moved toward a remembered location


class World:
    def __init__(self, n_food=64, seed=42):
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

    def visible_foods(self, x, y):
        """Food within VISION (Chebyshev radius). Local perception only."""
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


def forget_old(agent):
    agent.memory = [m for m in agent.memory if m.age < MEMORY_MAX_AGE]
    if len(agent.memory) > MEMORY_CAP:
        # keep the freshest
        agent.memory.sort(key=lambda m: -m.age)
        agent.memory = agent.memory[:MEMORY_CAP]


def remember(agent, food):
    for m in agent.memory:
        if m.x == food.x and m.y == food.y:
            m.age = 0
            return
    agent.memory.append(Mem(food.x, food.y))
    forget_old(agent)


def step_prey(agent, world):
    # age memories
    for m in agent.memory:
        m.age += 1
    forget_old(agent)

    # 1. food under feet
    if world.food_at(agent.x, agent.y) is not None:
        f = world.food_at(agent.x, agent.y)
        agent.energy += f.energy
        f.alive = False
        action = None
    elif agent.energy / 100.0 < agent.hunger:
        # 2a. visible food -> react and remember
        vis = world.visible_foods(agent.x, agent.y)
        if vis:
            for f in vis:
                remember(agent, f)
            target = min(vis, key=lambda f: (f.x - agent.x) ** 2 + (f.y - agent.y) ** 2)
            action = move_toward(agent, target.x, target.y)
        # 2b. memory -> goal-directed (probabilistic trust)
        elif agent.memory and world.rng.random() < agent.memory_weight:
            target = min(agent.memory, key=lambda m: torus_d2(agent, m.x, m.y))
            action = move_toward(agent, target.x, target.y)
            agent.goals += 1
            # verify on arrival: if we reached the remembered cell and it is
            # empty, the memory is stale -> drop it (memory has a cost)
            if torus_d2(agent, target.x, target.y) <= 1:
                if world.food_at(target.x, target.y) is None:
                    agent.memory.remove(target)
        # 2c. explore
        else:
            action = world.rng.choice(["N", "S", "E", "W"])
    else:
        # healthy: explore
        action = world.rng.choice(["N", "S", "E", "W"])

    if action is not None:
        dx, dy = MOVES[action]
        agent.x = (agent.x + dx) % SIZE
        agent.y = (agent.y + dy) % SIZE
        agent.energy -= MOVE_COST
    agent.energy -= METABOLISM
    if agent.energy <= 0.0:
        agent.alive = False


def torus_d2(a, tx, ty):
    dx = (tx - a.x + SIZE // 2) % SIZE - SIZE // 2
    dy = (ty - a.y + SIZE // 2) % SIZE - SIZE // 2
    return dx * dx + dy * dy


def simulate(ticks=6000, n_food=64, mem_mode="evolve", seed=42,
             report_every=1200):
    world = World(n_food, seed)
    rng = world.rng
    if mem_mode == "none":
        mw_init = lambda: 0.0
    elif mem_mode == "always":
        mw_init = lambda: 1.0
    else:
        mw_init = rng.random
    preys = [Agent(rng.randrange(SIZE), rng.randrange(SIZE),
                   100.0, rng.random(), mw_init())
             for _ in range(50)]
    stats = {"births": 0, "starved": 0}

    for t in range(ticks):
        for a in preys:
            if a.alive:
                step_prey(a, world)
        n_starved = sum(1 for a in preys if not a.alive)
        stats["starved"] += n_starved

        for a in list(preys):
            if a.alive and a.energy >= SPLIT_ENERGY and len(preys) < MAX_POP:
                preys.append(Agent(a.x + rng.choice([-1, 0, 1]),
                                   a.y + rng.choice([-1, 0, 1]),
                                   a.energy / 2.0,
                                   min(1.0, max(0.0, a.hunger + rng.gauss(0, MUTATION))),
                                   min(1.0, max(0.0, a.memory_weight + rng.gauss(0, MUTATION))),
                                   a.generation + 1))
                a.energy /= 2.0
                stats["births"] += 1

        preys = [a for a in preys if a.alive]
        world.step()

        if (t + 1) % report_every == 0:
            h = sum(a.hunger for a in preys) / len(preys) if preys else 0.0
            w = sum(a.memory_weight for a in preys) / len(preys) if preys else 0.0
            g = sum(a.goals for a in preys)
            print(f"t={t+1:5d}  pop={len(preys):4d}  hunger={h:.3f}  "
                  f"mem_w={w:.3f}  goals={g:6d}")

    print(f"\nfinal({mem_mode}): pop={len(preys)} births={stats['births']} "
          f"starved={stats['starved']}")
    if preys:
        hs = sorted(a.hunger for a in preys)
        ws = sorted(a.memory_weight for a in preys)
        print(f"  hunger: min={hs[0]:.3f} med={hs[len(hs)//2]:.3f} mean={sum(hs)/len(hs):.3f}")
        print(f"  mem_w : min={ws[0]:.3f} med={ws[len(ws)//2]:.3f} mean={sum(ws)/len(ws):.3f}")
        print(f"  goals total={sum(a.goals for a in preys)} "
              f"avg={sum(a.goals for a in preys)/len(preys):.1f}")
        avg_mem = sum(len(a.memory) for a in preys) / len(preys)
        print(f"  avg memory slots used: {avg_mem:.2f} / {MEMORY_CAP}")


def main():
    p = argparse.ArgumentParser(description="SEED-3 local perception + memory")
    p.add_argument("--ticks", type=int, default=6000)
    p.add_argument("--n-food", type=int, default=64)
    p.add_argument("--mem", choices=["none", "evolve", "always"], default="evolve")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report", type=int, default=1200)
    args = p.parse_args()
    simulate(ticks=args.ticks, n_food=args.n_food, mem_mode=args.mem,
             seed=args.seed, report_every=args.report)


if __name__ == "__main__":
    main()
