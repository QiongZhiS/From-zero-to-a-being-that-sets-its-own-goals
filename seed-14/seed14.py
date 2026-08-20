"""
SEED-14: Ratchet effect -- does culture reach what genetics cannot?

Docs/11: culture accumulates (cumulative culture / ratchet). In a
multi-parameter behavior space with INTERACTION effects, pure genetic
mutation searches the combination space blindly and slowly. Imitation
copies the whole successful COMBINATION (vector), preserving interactions
and locking in gains -- the ratchet.

Behavior: SEED-9 world (clustered, slow regen, dry signal). Two heritable
params with interaction: hunger (when to forage) x boldness (when to leave
a drying region). The combo matters: bold without hunger-awareness wastes
energy; hungry without bold dies in dry regions.

Gene/pheno separation (SEED-13): imitation changes pheno only; children
re-learn from gene. Imitation copies the WHOLE vector (ratchet), not params
independently.

Regimes:
  genetic   mutation only (blind combo search)
  cultural  mutation + imitation of successful combo (ratchet)

Measure: adaptation over time (avg energy / births), final fitness, and
param variance (consensus = lock-in).

Run:  python seed14.py [--mode cultural] [--ticks 8000] [--seed 42]
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
INIT_ENERGY = 200.0
MAX_POP = 400
MUTATION = 0.08
FOOD_REGEN = 0.05
VISION = 4
N_CLUSTERS = 8
CLUSTER_RADIUS = 6
WIN = 50

IMITATION_PROB = 0.10
IMITATION_PRECISION = 0.5
MODEL_SAMPLE = 5


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
    gene_h: float
    gene_b: float
    pheno_h: float
    pheno_b: float
    generation: int = 0
    alive: bool = True
    hit_win: deque = field(default_factory=lambda: deque(maxlen=WIN))
    total_hits: int = 0
    age: int = 0
    migrating: bool = False


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

    def nearest_food_delta(self, x, y):
        """LOCAL perception (VISION radius): finding food takes effort,
        so the hunger x boldness combination matters. No global info."""
        best = None
        bd = 10 ** 9
        for f in self.foods:
            if not f.alive:
                continue
            if max(abs(f.x - x), abs(f.y - y)) > VISION:
                continue
            dx = (f.x - x + SIZE // 2) % SIZE - SIZE // 2
            dy = (f.y - y + SIZE // 2) % SIZE - SIZE // 2
            d = abs(dx) + abs(dy)
            if d < bd:
                bd, best = d, (dx, dy)
        return best


MOVES = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}


def lifetime_rate(agent):
    return agent.total_hits / max(1, agent.age)


def recent_rate(agent):
    if len(agent.hit_win) < 20:
        return None
    return sum(agent.hit_win) / len(agent.hit_win)


def dry_signal(agent):
    r = recent_rate(agent)
    if r is None:
        return False
    lr = lifetime_rate(agent)
    return lr > 0.005 and r < 0.4 * lr


def step_prey(agent, world):
    agent.age += 1
    f = world.food_at(agent.x, agent.y)
    if f is not None:
        agent.energy += f.energy
        f.alive = False
        agent.total_hits += 1
        agent.hit_win.append(1)
        agent.migrating = False
        action = None
    else:
        agent.hit_win.append(0)
        # courage: act on dry signal (healthy only)
        if (not agent.migrating and not agent.energy / 100.0 < agent.pheno_h
                and dry_signal(agent) and world.rng.random() < agent.pheno_b):
            agent.migrating = True
        if agent.migrating:
            delta = world.nearest_food_delta(agent.x, agent.y)
            if delta is not None:
                dx, dy = delta
                if abs(dx) >= abs(dy):
                    action = "E" if dx > 0 else "W"
                else:
                    action = "S" if dy > 0 else "N"
            else:
                action = world.rng.choice(["N", "S", "E", "W"])
        elif agent.energy / 100.0 < agent.pheno_h:
            delta = world.nearest_food_delta(agent.x, agent.y)
            if delta is not None:
                dx, dy = delta
                if abs(dx) >= abs(dy):
                    action = "E" if dx > 0 else "W"
                else:
                    action = "S" if dy > 0 else "N"
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


def imitate(agent, agents, rng):
    """Ratchet: copy the WHOLE successful combination (vector), preserving
    interaction effects. Pheno only; gene untouched."""
    if rng.random() >= IMITATION_PROB:
        return
    models = [a for a in rng.sample(agents, min(MODEL_SAMPLE, len(agents)))
              if a.alive]
    if not models:
        return
    best = max(models, key=lambda a: a.energy)
    agent.pheno_h += IMITATION_PRECISION * (best.pheno_h - agent.pheno_h)
    agent.pheno_b += IMITATION_PRECISION * (best.pheno_b - agent.pheno_b)


def simulate(ticks=8000, n_food=128, mode="cultural", seed=42,
             report_every=800):
    world = World(n_food, seed)
    rng = world.rng
    agents = [Agent(rng.randrange(SIZE), rng.randrange(SIZE),
                    INIT_ENERGY, rng.random(), rng.random(),
                    rng.random(), rng.random())
              for _ in range(50)]
    stats = {"births": 0, "starved": 0}

    for t in range(ticks):
        for a in agents:
            if a.alive:
                step_prey(a, world)
                if mode == "cultural":
                    imitate(a, agents, rng)
        stats["starved"] += sum(1 for a in agents if not a.alive)

        for a in list(agents):
            if a.alive and a.energy >= SPLIT_ENERGY and len(agents) < MAX_POP:
                gh = min(1.0, max(0.0, a.gene_h + rng.gauss(0, MUTATION)))
                gb = min(1.0, max(0.0, a.gene_b + rng.gauss(0, MUTATION)))
                agents.append(Agent(
                    a.x + rng.choice([-1, 0, 1]),
                    a.y + rng.choice([-1, 0, 1]),
                    a.energy / 2.0,
                    gh, gb, gh, gb,
                    a.generation + 1))
                a.energy /= 2.0
                stats["births"] += 1

        agents = [a for a in agents if a.alive]
        world.step()

        if (t + 1) % report_every == 0:
            if agents:
                ph = sum(a.pheno_h for a in agents) / len(agents)
                pb = sum(a.pheno_b for a in agents) / len(agents)
                en = sum(a.energy for a in agents) / len(agents)
                print(f"t={t+1:5d}  pop={len(agents):4d}  energy={en:6.1f}  "
                      f"pheno_h={ph:.3f}  pheno_b={pb:.3f}")

    print(f"\nfinal({mode}): pop={len(agents)} births={stats['births']} "
          f"starved={stats['starved']}")
    if agents:
        phs = sorted(a.pheno_h for a in agents)
        pbs = sorted(a.pheno_b for a in agents)
        gh = sorted(a.gene_h for a in agents)
        gb = sorted(a.gene_b for a in agents)
        print(f"  pheno_h: med={phs[len(phs)//2]:.3f} range=[{phs[0]:.3f},{phs[-1]:.3f}]")
        print(f"  pheno_b: med={pbs[len(pbs)//2]:.3f} range=[{pbs[0]:.3f},{pbs[-1]:.3f}]")
        print(f"  gene_h: med={gh[len(gh)//2]:.3f}  gene_b: med={gb[len(gb)//2]:.3f}")
    return stats


def main():
    p = argparse.ArgumentParser(description="SEED-14 ratchet effect")
    p.add_argument("--mode", choices=["genetic", "cultural"], default="cultural")
    p.add_argument("--ticks", type=int, default=8000)
    p.add_argument("--n-food", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report", type=int, default=800)
    args = p.parse_args()
    simulate(ticks=args.ticks, n_food=args.n_food, mode=args.mode,
             seed=args.seed, report_every=args.report)


if __name__ == "__main__":
    main()
