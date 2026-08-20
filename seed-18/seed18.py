"""
SEED-18: Continuity -- individual AND population (docs/14)

User's point: continuity is needed not only by individuals but by the
POPULATION. A population's "memory" has its own tamper hierarchy:
  genes (most internal) -> cultural tradition (imitated) -> norms (immune)
The core question: can a population's memory resist external tampering?

Design (SEED-13 world, single param hunger, optimal ~0.72):
every INVASION_PERIOD ticks, 20 agents are replaced by "invaders" carrying
a WRONG hunger (too low = starves late, or too high = wastes energy).
Measure how far the population's consensus is pushed off the optimum
(tamper depth) and how fast it recovers (continuity resilience).

Regimes:
  genetic   no imitation (population memory = genes only)
  cultural  imitation (population memory = tradition, with immune response:
            only successful agents are copied)

Run:  python seed18.py [--mode cultural] [--invader 0.1] [--ticks 8000]
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
OPTIMUM = 0.72          # SEED-13 measured optimum for n_food=64

IMITATION_PROB = 0.10
IMITATION_PRECISION = 0.5
MODEL_SAMPLE = 5

INVASION_PERIOD = 2000
INVADER_COUNT = 20
TOL = 0.05              # recovery tolerance
RECOVERY_WINDOW = 3     # report points (600 ticks) inside tolerance


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


def invade(agents, world, invader_hunger):
    """External tampering: replace random agents with invaders carrying a
    wrong behavior value. Simulates injected misinformation."""
    targets = [a for a in agents if a.alive]
    if not targets:
        return
    chosen = world.rng.sample(targets, min(INVADER_COUNT, len(targets)))
    for a in chosen:
        a.gene_hunger = invader_hunger
        a.pheno_hunger = invader_hunger


def simulate(ticks=8000, mode="cultural", invader=0.1, seed=42,
             report_every=200):
    world = World(64, seed)
    rng = world.rng
    agents = [Agent(rng.randrange(SIZE), rng.randrange(SIZE),
                    INIT_ENERGY, rng.random(), rng.random())
              for _ in range(50)]
    stats = {"births": 0, "starved": 0}
    history = []
    invasions = []

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

        if t > 0 and t % INVASION_PERIOD == 0:
            invade(agents, world, invader)
            invasions.append(t)

        if (t + 1) % report_every == 0:
            ph = sum(a.pheno_hunger for a in agents) / len(agents) if agents else 0.0
            history.append((t + 1, ph))

    # tamper depth & recovery for each invasion
    print(f"\nfinal({mode}, invader={invader}): pop={len(agents)} "
          f"births={stats['births']} starved={stats['starved']}")
    print("=== population continuity report ===")
    for inv in invasions:
        pre = next((h for h in history if h[0] >= inv - 200), (inv, None))
        # find max deviation in the 1000 ticks after invasion
        after = [h for h in history if inv <= h[0] <= inv + 1000]
        if not after or pre[1] is None:
            continue
        max_dev = max(abs(h[1] - OPTIMUM) for h in after)
        # recovery: first point inside tolerance sustained for the window
        rec = None
        for i in range(len(after) - RECOVERY_WINDOW):
            if all(abs(after[j][1] - OPTIMUM) <= TOL
                   for j in range(i, i + RECOVERY_WINDOW)):
                rec = after[i][0]
                break
        rec_info = f"recovered_at={rec}" if rec else "NO RECOVERY"
        print(f"  invasion@{inv:5d}: max_dev={max_dev:.3f}  {rec_info}")

    # no-invasion control value
    final_ph = history[-1][1] if history else 0.0
    print(f"  final consensus={final_ph:.3f} (optimum={OPTIMUM})")
    return stats


def main():
    p = argparse.ArgumentParser(description="SEED-18 population continuity")
    p.add_argument("--mode", choices=["genetic", "cultural"], default="cultural")
    p.add_argument("--invader", type=float, default=0.1)
    p.add_argument("--ticks", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    simulate(ticks=args.ticks, mode=args.mode, invader=args.invader,
             seed=args.seed)


if __name__ == "__main__":
    main()
