"""
SEED-4: Curiosity -- does exploration emerge from survival pressure?

When healthy, agents either wander randomly (reactive) or move toward
least-visited areas (curiosity), controlled by heritable parameter
curiosity (0..1). Exploration costs energy but reveals food-rich regions
-- IF food is clustered. In uniform worlds exploration has no value.

Three regimes:
  --world uniform    --cur evolve     expect curiosity ~ 0
  --world clustered  --cur evolve     expect curiosity > 0
  --world clustered  --cur fixed0     control

Run:  python seed4.py [--world clustered] [--cur evolve] [--ticks 6000]
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
    generation: int = 0
    alive: bool = True
    memory: list = field(default_factory=list)
    visited: dict = field(default_factory=dict)  # (cx,cy) -> visit count
    explores: int = 0     # curiosity-driven moves


class World:
    def __init__(self, n_food=64, mode="uniform", seed=42, shift_every=0):
        self.n_food = n_food
        self.mode = mode
        self.shift_every = shift_every
        self.rng = random.Random(seed)
        self.reset_food(seed)

    def reset_food(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        self.foods = []
        if self.mode == "clustered":
            self._place_clusters()
        else:
            for _ in range(self.n_food):
                self.foods.append(Food(self.rng.randrange(SIZE),
                                       self.rng.randrange(SIZE)))

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
        """Clusters relocate periodically: the world is dynamic, so knowing
        where food WAS is not enough -- exploration must keep finding it.
        Old food is removed (the old rich region is now barren)."""
        if self.shift_every and self.mode == "clustered" \
                and t % self.shift_every == 0:
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


def explore_direction(agent, world):
    """Move toward the least-visited coarse cell (curiosity)."""
    scored = []
    for d, (dx, dy) in MOVES.items():
        nx = (agent.x + dx) % SIZE
        ny = (agent.y + dy) % SIZE
        scored.append((agent.visited.get((nx // 4, ny // 4), 0), d))
    scored.sort()
    ties = [d for c, d in scored if c == scored[0][0]]
    return world.rng.choice(ties)


def step_prey(agent, world):
    for m in agent.memory:
        m.age += 1
    forget_old(agent)

    if world.food_at(agent.x, agent.y) is not None:
        f = world.food_at(agent.x, agent.y)
        agent.energy += f.energy
        f.alive = False
        action = None
    elif agent.energy / 100.0 < agent.hunger:
        vis = world.visible_foods(agent.x, agent.y)
        if vis:
            for f in vis:
                remember(agent, f)
            target = min(vis, key=lambda f: (f.x - agent.x) ** 2 + (f.y - agent.y) ** 2)
            action = move_toward(agent, target.x, target.y)
        elif agent.memory and world.rng.random() < agent.memory_weight:
            target = min(agent.memory, key=lambda m: torus_d2(agent, m.x, m.y))
            action = move_toward(agent, target.x, target.y)
            if torus_d2(agent, target.x, target.y) <= 1:
                if world.food_at(target.x, target.y) is None:
                    agent.memory.remove(target)
        else:
            action = world.rng.choice(["N", "S", "E", "W"])
    else:
        # healthy: curiosity-driven exploration vs random wander
        if world.rng.random() < agent.curiosity:
            action = explore_direction(agent, world)
            agent.explores += 1
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


def simulate(ticks=6000, n_food=64, world_mode="uniform", cur_mode="evolve",
             seed=42, report_every=1200, shift_every=0):
    world = World(n_food, world_mode, seed, shift_every)
    rng = world.rng
    if cur_mode == "fixed0":
        cur_init = lambda: 0.0
    else:
        cur_init = rng.random
    preys = [Agent(rng.randrange(SIZE), rng.randrange(SIZE),
                   100.0, rng.random(), rng.random(), cur_init())
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
                    a.generation + 1))
                a.energy /= 2.0
                stats["births"] += 1

        preys = [a for a in preys if a.alive]
        world.step()
        world.maybe_shift(t)

        if (t + 1) % report_every == 0:
            c = sum(a.curiosity for a in preys) / len(preys) if preys else 0.0
            e = sum(a.explores for a in preys)
            print(f"t={t+1:5d}  pop={len(preys):4d}  curiosity={c:.3f}  "
                  f"explores={e:6d}")

    print(f"\nfinal(world={world_mode}, cur={cur_mode}): pop={len(preys)} "
          f"births={stats['births']} starved={stats['starved']}")
    if preys:
        cs = sorted(a.curiosity for a in preys)
        print(f"  curiosity: min={cs[0]:.3f} med={cs[len(cs)//2]:.3f} "
              f"mean={sum(cs)/len(cs):.3f}")
        print(f"  explores total={sum(a.explores for a in preys)}")
        es = sorted(a.energy for a in preys)
        print(f"  energy: med={es[len(es)//2]:.1f} mean={sum(es)/len(es):.1f}")


def main():
    p = argparse.ArgumentParser(description="SEED-4 curiosity emergence")
    p.add_argument("--ticks", type=int, default=6000)
    p.add_argument("--n-food", type=int, default=64)
    p.add_argument("--world", choices=["uniform", "clustered"], default="uniform")
    p.add_argument("--cur", choices=["evolve", "fixed0"], default="evolve")
    p.add_argument("--shift", type=int, default=0,
                   help="cluster relocation interval (0 = static)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report", type=int, default=1200)
    args = p.parse_args()
    simulate(ticks=args.ticks, n_food=args.n_food, world_mode=args.world,
             cur_mode=args.cur, seed=args.seed, report_every=args.report,
             shift_every=args.shift)


if __name__ == "__main__":
    main()
