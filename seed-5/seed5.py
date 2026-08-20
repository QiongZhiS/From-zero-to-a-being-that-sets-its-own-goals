"""
SEED-5: Intuition -- individual-level integration of experience

Memory -> intuition within one lifetime. Each agent carries a 'fatness map'
(which coarse regions are food-rich), accumulated from its own experience:
eating food adds score to the current cell, scores decay over time. When
hungry and no food is visible, the agent may go directly toward the highest
scoring region -- WITHOUT consulting its explicit memory. That is intuition:
the influence remains, the source (which meal) is gone.

The fatness map is NOT heritable: experience dies with the agent. Only the
LEARNING RATE (how fast experience integrates) is heritable -- capacity is
inherited, knowledge is not (anti-Lamarckian).

Heritable: hunger, memory_weight, curiosity, learning_rate.

Regimes (--lr):
  fixed0   learning_rate = 0   (no individual learning, SEED-4 baseline)
  evolve   learning_rate evolves
  high     learning_rate fixed 0.8 (learns fast, forgets fast)

World: clustered with periodic cluster relocation (dynamic).

Run:  python seed5.py [--lr evolve] [--ticks 6000] [--shift 1500]
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
VISION = 4
MEMORY_CAP = 8
MEMORY_MAX_AGE = 300
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


@dataclass
class Agent:
    x: int
    y: int
    energy: float
    hunger: float
    memory_weight: float = 0.0
    curiosity: float = 0.0
    learning_rate: float = 0.0
    generation: int = 0
    alive: bool = True
    memory: list = field(default_factory=list)
    fatmap: dict = field(default_factory=dict)   # (cx,cy) -> score (NOT heritable)
    intuition_moves: int = 0
    memory_moves: int = 0


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


def forget_old(agent):
    agent.memory = [m for m in agent.memory if m.age < MEMORY_MAX_AGE]
    if len(agent.memory) > MEMORY_CAP:
        agent.memory.sort(key=lambda m: -m.age)
        agent.memory = agent.memory[:MEMORY_CAP]


def remember(agent, food):
    for m in agent.memory:
        if m.x == food.x and m.y == food.y:
            m.age = 0
            return
    agent.memory.append(Mem(food.x, food.y))
    forget_old(agent)


def decay_fatmap(agent):
    """Experience fades. Higher learning_rate = faster integration AND faster
    forgetting -- the map tracks the present, not the past."""
    if not agent.fatmap:
        return
    d = 1.0 - agent.learning_rate * 0.02
    for k in list(agent.fatmap):
        v = agent.fatmap[k] * d
        if v < 0.05:
            del agent.fatmap[k]
        else:
            agent.fatmap[k] = v


def best_region(agent):
    if not agent.fatmap:
        return None, 0.0
    k = max(agent.fatmap, key=agent.fatmap.get)
    return k, agent.fatmap[k]


def step_prey(agent, world):
    for m in agent.memory:
        m.age += 1
    forget_old(agent)
    decay_fatmap(agent)

    if world.food_at(agent.x, agent.y) is not None:
        f = world.food_at(agent.x, agent.y)
        agent.energy += f.energy
        f.alive = False
        # reinforce: this region is fat
        c = (agent.x // 4, agent.y // 4)
        agent.fatmap[c] = agent.fatmap.get(c, 0.0) + 1.0
        action = None
    elif agent.energy / 100.0 < agent.hunger:
        vis = world.visible_foods(agent.x, agent.y)
        if vis:
            for f in vis:
                remember(agent, f)
            target = min(vis, key=lambda f: (f.x - agent.x) ** 2 + (f.y - agent.y) ** 2)
            action = move_toward(agent, target.x, target.y)
        else:
            # INTUITION: go to the fattest known region, no explicit memory.
            # Confidence grows with accumulated score.
            region, score = best_region(agent)
            if region is not None and score > 1.0 \
                    and world.rng.random() < min(1.0, score / 4.0):
                action = move_toward(agent, region[0] * 4, region[1] * 4)
                agent.intuition_moves += 1
            # explicit memory (source traceable)
            elif agent.memory and world.rng.random() < agent.memory_weight:
                target = min(agent.memory, key=lambda m: torus_d2(agent, m.x, m.y))
                action = move_toward(agent, target.x, target.y)
                agent.memory_moves += 1
                if torus_d2(agent, target.x, target.y) <= 1:
                    if world.food_at(target.x, target.y) is None:
                        agent.memory.remove(target)
            # explore
            else:
                action = world.rng.choice(["N", "S", "E", "W"])
    else:
        # healthy: curiosity-driven exploration vs random wander
        if world.rng.random() < agent.curiosity:
            # go toward least-visited coarse cell
            scored = []
            for d, (dx, dy) in MOVES.items():
                nx = (agent.x + dx) % SIZE
                ny = (agent.y + dy) % SIZE
                scored.append((agent.fatmap.get((nx // 4, ny // 4), 0.0), d))
            # NOTE: curiosity uses visited-logic in SEED-4; here we approximate
            # exploration by going to the LEAST reinforced region (unknown).
            scored.sort()
            ties = [d for s, d in scored if s == scored[0][0]]
            action = world.rng.choice(ties)
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


def simulate(ticks=6000, n_food=64, lr_mode="evolve", seed=42,
             report_every=1200, shift_every=1500):
    world = World(n_food, seed, shift_every)
    rng = world.rng
    if lr_mode == "fixed0":
        lr_init = lambda: 0.0
    elif lr_mode == "high":
        lr_init = lambda: 0.8
    else:
        lr_init = rng.random
    preys = [Agent(rng.randrange(SIZE), rng.randrange(SIZE),
                   100.0, rng.random(), rng.random(), rng.random(), lr_init())
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
                    min(1.0, max(0.0, a.memory_weight + rng.gauss(0, MUTATION))),
                    min(1.0, max(0.0, a.curiosity + rng.gauss(0, MUTATION))),
                    min(1.0, max(0.0, a.learning_rate + rng.gauss(0, MUTATION))),
                    a.generation + 1))
                a.energy /= 2.0
                stats["births"] += 1

        preys = [a for a in preys if a.alive]
        world.step()
        world.maybe_shift(t)

        if (t + 1) % report_every == 0:
            lr = sum(a.learning_rate for a in preys) / len(preys) if preys else 0.0
            im = sum(a.intuition_moves for a in preys)
            print(f"t={t+1:5d}  pop={len(preys):4d}  lr={lr:.3f}  "
                  f"intuition={im:6d}")

    print(f"\nfinal(lr={lr_mode}): pop={len(preys)} births={stats['births']} "
          f"starved={stats['starved']}")
    if preys:
        ls = sorted(a.learning_rate for a in preys)
        print(f"  learning_rate: min={ls[0]:.3f} med={ls[len(ls)//2]:.3f} "
              f"mean={sum(ls)/len(ls):.3f}")
        print(f"  intuition moves={sum(a.intuition_moves for a in preys)}  "
              f"memory moves={sum(a.memory_moves for a in preys)}")
        fm = sum(len(a.fatmap) for a in preys) / len(preys)
        print(f"  avg fatmap size: {fm:.1f}")


def main():
    p = argparse.ArgumentParser(description="SEED-5 intuition via experience")
    p.add_argument("--ticks", type=int, default=6000)
    p.add_argument("--n-food", type=int, default=64)
    p.add_argument("--lr", choices=["fixed0", "evolve", "high"], default="evolve")
    p.add_argument("--shift", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report", type=int, default=1200)
    args = p.parse_args()
    simulate(ticks=args.ticks, n_food=args.n_food, lr_mode=args.lr,
             seed=args.seed, report_every=args.report, shift_every=args.shift)


if __name__ == "__main__":
    main()
