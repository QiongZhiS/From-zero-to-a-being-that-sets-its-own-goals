"""
SEED-10: Open capability space -- neuroevolution + post-hoc archaeology

SEED-9 lesson (docs/08): every capability parameter (boldness, curiosity...)
was predefined by US, then evolution tuned its value. That is tuning, not
creation. Here the capability space is a small neural net whose 71 weights
are the genome. We do not know what behaviors it can express. After
evolution we analyze behavior and NAME capabilities post-hoc -- archaeology,
not design.

Raw perception inputs (no semantic capability dimensions):
  energy/100, recent hit rate, visible-food dx, visible-food dy, since_hit/100
Outputs: N/S/E/W/GATHER (argmax).
Genome: 71 real weights, mutated by Gaussian noise on reproduction.

World: clustered, moderate regen (SEED-9's world, where migration can pay).

Run:  python seed10.py [--ticks 8000] [--seed 42]
"""

import argparse
import math
import random
from collections import deque
from dataclasses import dataclass, field

SIZE = 64
METABOLISM = 0.4
MOVE_COST = 0.2
FOOD_ENERGY = 60
SPLIT_ENERGY = 80.0
INIT_ENERGY = 200.0
MAX_POP = 300
MUT = 0.12
FOOD_REGEN = 0.05
VISION = 10             # larger field: initial individuals can see a cluster
N_CLUSTERS = 8
CLUSTER_RADIUS = 6
WIN = 30

HIDDEN = 6
N_IN = 5
N_OUT = 5
W1 = N_IN * HIDDEN          # 30
B1 = HIDDEN                 # 6
W2 = HIDDEN * N_OUT         # 30
B2 = N_OUT                  # 5
GENOME_LEN = W1 + B1 + W2 + B2   # 71

ACTIONS = ["N", "S", "E", "W", "GATHER"]
MOVES = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}
ACT_IDX = {a: i for i, a in enumerate(ACTIONS)}


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
    genome: list
    generation: int = 0
    alive: bool = True
    hit_win: deque = field(default_factory=lambda: deque(maxlen=WIN))
    total_hits: int = 0
    age: int = 0


class World:
    def __init__(self, n_food=128, seed=42, regen=FOOD_REGEN):
        self.n_food = n_food
        self.regen = regen
        self.rng = random.Random(seed)
        self.reset_food(seed)

    def reset_food(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        self.foods = []
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

    def vis_food_delta(self, x, y):
        best = None
        for f in self.foods:
            if not f.alive:
                continue
            if max(abs(f.x - x), abs(f.y - y)) <= VISION:
                dx = (f.x - x) / VISION
                dy = (f.y - y) / VISION
                if best is None or abs(dx) + abs(dy) < abs(best[0]) + abs(best[1]):
                    best = (dx, dy)
        return best or (0.0, 0.0)


def net_forward(genome, x):
    w1 = genome[0:W1]
    b1 = genome[W1:W1 + B1]
    w2 = genome[W1 + B1:W1 + B1 + W2]
    b2 = genome[W1 + B1 + W2:]
    h = []
    for i in range(HIDDEN):
        s = b1[i]
        for j in range(N_IN):
            s += w1[i * N_IN + j] * x[j]
        h.append(math.tanh(s))
    out = []
    for k in range(N_OUT):
        s = b2[k]
        for i in range(HIDDEN):
            s += w2[i * N_OUT + k] * h[i]
        out.append(s)
    return out


def perceive(agent, world):
    r = sum(agent.hit_win) / len(agent.hit_win) if agent.hit_win else 0.0
    dx, dy = world.vis_food_delta(agent.x, agent.y)
    return [agent.energy / 100.0, r, dx, dy, min(agent.age, 100) / 100.0]


def step_prey(agent, world):
    agent.age += 1
    x = perceive(agent, world)
    logits = net_forward(agent.genome, x)
    ai = max(range(N_OUT), key=lambda k: logits[k])
    action = ACTIONS[ai]

    f = world.food_at(agent.x, agent.y)
    if action == "GATHER" and f is not None:
        agent.energy += f.energy
        f.alive = False
        agent.total_hits += 1
        agent.hit_win.append(1)
    elif action in MOVES:
        dx, dy = MOVES[action]
        agent.x = (agent.x + dx) % SIZE
        agent.y = (agent.y + dy) % SIZE
        agent.energy -= MOVE_COST
        agent.hit_win.append(0)
    else:
        agent.hit_win.append(0)
    agent.energy -= METABOLISM
    if agent.energy <= 0.0:
        agent.alive = False


def random_genome(rng):
    g = [rng.gauss(0, 0.3) for _ in range(GENOME_LEN)]
    # minimal STRUCTURE prior (docs/08): a "seek visible food" pathway.
    # Physical worlds carry structure (physics IS structure); pure random
    # genomes starve in a sparse world (SEED-0's gap, reproduced here).
    # Everything beyond this pathway is left free for evolution to discover.
    # inputs: 0=energy 1=hit_rate 2=dx 3=dy 4=since_hit
    # outputs: 0=N 1=S 2=E 3=W 4=GATHER
    hid0 = 0
    g[hid0 * N_IN + 2] += 1.5   # hid0 detects dx (food to the east)
    g[hid0 * N_IN + 3] += 1.5   # hid0 detects dy (food to the south)
    o = W1 + B1 + hid0 * N_OUT
    g[o + 2] += 1.5             # hid0 -> E
    g[o + 3] += -1.5            # hid0 -> W
    g[o + 1] += 1.5             # hid0 -> S
    g[o + 0] += -1.5            # hid0 -> N
    return g


def mutate(genome, rng):
    return [g + rng.gauss(0, MUT) for g in genome]


def simulate(ticks=8000, n_food=192, seed=42, report_every=1600):
    world = World(n_food, seed)
    rng = world.rng
    agents = [Agent(rng.randrange(SIZE), rng.randrange(SIZE),
                    INIT_ENERGY, random_genome(rng))
              for _ in range(50)]
    stats = {"births": 0, "starved": 0}

    for t in range(ticks):
        for a in agents:
            if a.alive:
                step_prey(a, world)
        stats["starved"] += sum(1 for a in agents if not a.alive)

        for a in list(agents):
            if a.alive and a.energy >= SPLIT_ENERGY and len(agents) < MAX_POP:
                agents.append(Agent(
                    a.x + rng.choice([-1, 0, 1]),
                    a.y + rng.choice([-1, 0, 1]),
                    a.energy / 2.0,
                    mutate(a.genome, rng),
                    a.generation + 1))
                a.energy /= 2.0
                stats["births"] += 1

        agents = [a for a in agents if a.alive]
        world.step()

        if (t + 1) % report_every == 0:
            print(f"t={t+1:5d}  pop={len(agents):4d}  gen={sum(a.generation for a in agents)/max(1,len(agents)):.1f}")

    print(f"\nfinal: pop={len(agents)} births={stats['births']} starved={stats['starved']}")
    return agents, world


def archaeology(agents, world, n=60):
    """POST-HOC analysis: probe the evolved networks with controlled
    perception inputs, record the response spectrum, then let the reader
    name the capabilities. We do NOT predefine what to look for."""
    if not agents:
        print("population extinct -- no behavior to analyze")
        return
    sample = agents[:n]

    def probe(label, energy, hit_rate, dx, dy):
        x = [energy, hit_rate, dx, dy, 0.5]
        counts = {"N": 0, "S": 0, "E": 0, "W": 0, "GATHER": 0}
        for a in sample:
            logits = net_forward(a.genome, x)
            ai = max(range(N_OUT), key=lambda k: logits[k])
            counts[ACTIONS[ai]] += 1
        tot = len(sample)
        dist = {k: f"{v / tot:.2f}" for k, v in sorted(counts.items())}
        print(f"  {label:28s} {dist}")

    print("\n=== behavior archaeology: input response spectrum (post-hoc) ===")
    probe("healthy, food to the EAST", 0.9, 0.5, 0.5, 0.0)
    probe("healthy, food to the SOUTH", 0.9, 0.5, 0.0, 0.5)
    probe("healthy, no food visible", 0.9, 0.5, 0.0, 0.0)
    probe("healthy, dry signal (low hits)", 0.9, 0.02, 0.0, 0.0)
    probe("hungry, food to the EAST", 0.2, 0.1, 0.5, 0.0)
    probe("hungry, no food visible", 0.2, 0.1, 0.0, 0.0)
    probe("starving, no food visible", 0.05, 0.02, 0.0, 0.0)


def main():
    p = argparse.ArgumentParser(description="SEED-10 neuroevolution + archaeology")
    p.add_argument("--ticks", type=int, default=8000)
    p.add_argument("--n-food", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report", type=int, default=1600)
    p.add_argument("--smoke", action="store_true", help="short run")
    args = p.parse_args()
    if args.smoke:
        args.ticks = 2000
        args.report = 500
    agents, world = simulate(ticks=args.ticks, n_food=args.n_food,
                             seed=args.seed, report_every=args.report)
    archaeology(agents, world)


if __name__ == "__main__":
    main()
