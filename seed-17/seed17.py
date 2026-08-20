"""
SEED-17: Non-imitable skill -- does experience resist imitation?

SEED-16: imitation homogenizes the group (copies the "most successful" and
erases division of labor). Real division of labor (ants, humans) is
maintained by skills that CANNOT be imitated. Here: skill = LIFETIME
experience (foraging efficiency in a zone grows with meals taken there).
Experience is NOT heritable and NOT imitable -- imitation copies the
preference parameter only, never the experience.

So an imitator can copy "specialize in A" (preference=0) but starts with
zero A-experience: the skill must be re-acquired by practice. If division
of labor survives under imitation, non-imitable skill is the mechanism.

Regimes (SEED-16 world: A sparse-rich, B dense-poor):
  genetic    preference evolves by mutation
  cultural   preference evolves + imitation (copies pref, not experience)

Measure: preference distribution (bimodal = division survives), starvation.

Run:  python seed17.py [--mode cultural] [--ticks 8000] [--seed 42]
"""

import argparse
import random
from dataclasses import dataclass

SIZE = 64
METABOLISM = 0.4
MOVE_COST = 0.2
SPLIT_ENERGY = 100.0
INIT_ENERGY = 150.0
MAX_POP = 400
MUTATION = 0.05
FOOD_REGEN = 0.10

A_N = 40
A_ENERGY = 80
B_N = 96
B_ENERGY = 30
A_X_MAX = 16      # A zone far left; B zone far right; middle is desert
B_X_MIN = 48      # crossing the desert is expensive -> specialization pays
VISION_BASE = 3
VISION_EXP = 8        # extra sight per experience (max 8 exp levels)
EXP_CAP = 8

IMITATION_PROB = 0.10
IMITATION_PRECISION = 0.5
MODEL_SAMPLE = 5


@dataclass
class Food:
    x: int
    y: int
    energy: int
    zone: str
    alive: bool = True


@dataclass
class Agent:
    x: int
    y: int
    energy: float
    gene_pref: float
    pheno_pref: float
    streak_a: int = 0       # consecutive meals in A (switch resets to 0)
    streak_b: int = 0
    generation: int = 0
    alive: bool = True


class World:
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.reset_food(seed)

    def reset_food(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        self.foods = []
        for _ in range(A_N):
            self.foods.append(Food(self.rng.randrange(A_X_MAX + 1),
                                   self.rng.randrange(SIZE), A_ENERGY, "A"))
        for _ in range(B_N):
            self.foods.append(Food(self.rng.randrange(B_X_MIN, SIZE),
                                   self.rng.randrange(SIZE), B_ENERGY, "B"))

    def step(self):
        for f in self.foods:
            if not f.alive and self.rng.random() < FOOD_REGEN:
                f.alive = True
                f.energy = A_ENERGY if f.zone == "A" else B_ENERGY

    def food_at(self, x, y):
        for f in self.foods:
            if f.alive and f.x == x and f.y == y:
                return f
        return None

    def nearest_food_in_zone(self, x, y, zone, vision):
        best = None
        bd = 10 ** 9
        for f in self.foods:
            if not f.alive or f.zone != zone:
                continue
            if max(abs(f.x - x), abs(f.y - y)) > vision:
                continue
            dx = (f.x - x + SIZE // 2) % SIZE - SIZE // 2
            dy = (f.y - y + SIZE // 2) % SIZE - SIZE // 2
            d = abs(dx) + abs(dy)
            if d < bd:
                bd, best = d, (dx, dy)
        return best


MOVES = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}


def vision_of(agent, zone):
    """Skill = CONSECUTIVE-meal bonus. Efficiency grows with streak in the
    zone; switching zones resets it. Imitators copy the preference but the
    bonus must be re-earned by practice -- this is the non-imitable part.
    A generalist (mid preference) never holds a streak -> no bonus."""
    streak = agent.streak_a if zone == "A" else agent.streak_b
    return VISION_BASE + VISION_EXP * min(streak, EXP_CAP) / EXP_CAP


def act(agent, world):
    if world.food_at(agent.x, agent.y) is not None:
        return "GATHER"
    zone = "B" if world.rng.random() < agent.pheno_pref else "A"
    delta = world.nearest_food_in_zone(agent.x, agent.y, zone,
                                       vision_of(agent, zone))
    if delta is None:
        for z in (["A", "B"] if zone == "B" else ["B", "A"]):
            delta = world.nearest_food_in_zone(agent.x, agent.y, z,
                                               vision_of(agent, z))
            if delta is not None:
                break
        if delta is None:
            return world.rng.choice(["N", "S", "E", "W"])
        dx, dy = delta
    else:
        dx, dy = delta
    if abs(dx) >= abs(dy):
        return "E" if dx > 0 else "W"
    return "S" if dy > 0 else "N"


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
            if f.zone == "A":
                agent.streak_a += 1
                agent.streak_b = 0    # switching resets the bonus
            else:
                agent.streak_b += 1
                agent.streak_a = 0
    agent.energy -= METABOLISM
    if agent.energy <= 0.0:
        agent.alive = False


def imitate(agent, agents, rng):
    """Copy preference ONLY. Experience is NOT copied (non-imitable skill)."""
    if rng.random() >= IMITATION_PROB:
        return
    models = [a for a in rng.sample(agents, min(MODEL_SAMPLE, len(agents)))
              if a.alive]
    if not models:
        return
    best = max(models, key=lambda a: a.energy)
    agent.pheno_pref += IMITATION_PRECISION * (best.pheno_pref - agent.pheno_pref)


def simulate(ticks=8000, mode="cultural", seed=42, report_every=800):
    world = World(seed)
    rng = world.rng
    agents = [Agent(rng.randrange(SIZE), rng.randrange(SIZE),
                    INIT_ENERGY, rng.random(), rng.random())
              for _ in range(50)]
    stats = {"births": 0, "starved": 0}

    for t in range(ticks):
        for a in agents:
            if a.alive:
                apply(a, world, act(a, world))
                if mode == "cultural":
                    imitate(a, agents, rng)
        stats["starved"] += sum(1 for a in agents if not a.alive)

        for a in list(agents):
            if a.alive and a.energy >= SPLIT_ENERGY and len(agents) < MAX_POP:
                gp = min(1.0, max(0.0, a.gene_pref + rng.gauss(0, MUTATION)))
                # children start with ZERO streak (skill not heritable)
                agents.append(Agent(
                    a.x + rng.choice([-1, 0, 1]),
                    a.y + rng.choice([-1, 0, 1]),
                    a.energy / 2.0, gp, gp, 0, 0, a.generation + 1))
                a.energy /= 2.0
                stats["births"] += 1

        agents = [a for a in agents if a.alive]
        world.step()

        if (t + 1) % report_every == 0:
            pref = sum(a.pheno_pref for a in agents) / len(agents) if agents else 0.0
            lo = sum(1 for a in agents if a.pheno_pref < 0.25) / max(1, len(agents))
            hi = sum(1 for a in agents if a.pheno_pref > 0.75) / max(1, len(agents))
            mid = 1.0 - lo - hi
            exp = sum(a.streak_a + a.streak_b for a in agents) / max(1, len(agents))
            print(f"t={t+1:5d}  pop={len(agents):4d}  mean_pref={pref:.3f}  "
                  f"A={lo:.2f}  mid={mid:.2f}  B={hi:.2f}  avg_streak={exp:.1f}")

    print(f"\nfinal({mode}): pop={len(agents)} births={stats['births']} "
          f"starved={stats['starved']}")
    if agents:
        lo = sum(1 for a in agents if a.pheno_pref < 0.25)
        hi = sum(1 for a in agents if a.pheno_pref > 0.75)
        mid = len(agents) - lo - hi
        print(f"  preference: A-spec={lo}  mid={mid}  B-spec={hi}")
    return stats


def main():
    p = argparse.ArgumentParser(description="SEED-17 non-imitable skill")
    p.add_argument("--mode", choices=["genetic", "cultural"], default="cultural")
    p.add_argument("--ticks", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report", type=int, default=800)
    args = p.parse_args()
    simulate(ticks=args.ticks, mode=args.mode, seed=args.seed,
             report_every=args.report)


if __name__ == "__main__":
    main()
