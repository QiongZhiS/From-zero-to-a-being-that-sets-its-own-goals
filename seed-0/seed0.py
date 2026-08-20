"""
SEED-0: Survival Engine for Emergent Dynamics
A minimal living agent driven by ONE homeostatic variable (energy).
No external reward function: survival is the only implicit objective.

Run:  python seed0.py --all
      python seed0.py --baseline     (L0 random vs L1 heuristic)
      python seed0.py --learn        (L2 tabular Q-learning, no external reward)
      python seed0.py --audit        (energy conservation check)
"""

import argparse
import math
import random
import time
from dataclasses import dataclass

# ----------------------------------------------------------------------------
# World
# ----------------------------------------------------------------------------

@dataclass
class Food:
    x: int
    y: int
    energy: int = 30
    alive: bool = True


class World:
    """64x64 torus grid with regenerating food sources. Engine-level rules
    (metabolism, death) are hardcoded here: a strategy can only influence
    energy through actions, never directly."""

    SIZE = 64
    METABOLISM = 0.4   # cost per tick, even while resting
    MOVE_COST = 0.2
    DEATH_LEVEL = 0.0
    FOOD_REGEN = 0.15
    FOOD_ENERGY = 60

    def __init__(self, n_food=64, seed=None):
        self.n_food = n_food
        self.rng = random.Random(seed)
        self.reset_food(seed)

    def reset_food(self, seed=None):
        """Re-seed food placement. Called once per life."""
        if seed is not None:
            self.rng = random.Random(seed)
        self.foods = [Food(self.rng.randrange(self.SIZE),
                           self.rng.randrange(self.SIZE))
                      for _ in range(self.n_food)]

    def step(self):
        """Regenerate depleted food cells (tick boundary)."""
        for f in self.foods:
            if not f.alive and self.rng.random() < self.FOOD_REGEN:
                f.alive = True
                f.energy = self.FOOD_ENERGY

    def food_at(self, x, y):
        for f in self.foods:
            if f.alive and f.x == x and f.y == y:
                return f
        return None

    def torus_delta(self, x1, y1, x2, y2):
        dx = (x2 - x1 + self.SIZE // 2) % self.SIZE - self.SIZE // 2
        dy = (y2 - y1 + self.SIZE // 2) % self.SIZE - self.SIZE // 2
        return dx, dy

    def nearest_foods(self, x, y, k=3):
        scored = []
        for f in self.foods:
            if not f.alive:
                continue
            dx, dy = self.torus_delta(x, y, f.x, f.y)
            scored.append((abs(dx) + abs(dy), dx, dy))
        scored.sort()
        return scored[:k]


# ----------------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------------

class Agent:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.energy = 0.0
        self.alive = False
        self.steps = 0
        self.total_gathered = 0.0
        self.total_moved = 0

    def reset(self, world):
        self.x = world.rng.randrange(world.SIZE)
        self.y = world.rng.randrange(world.SIZE)
        self.energy = 80.0
        self.alive = True
        self.steps = 0
        self.total_gathered = 0.0
        self.total_moved = 0


# ----------------------------------------------------------------------------
# Actions
# ----------------------------------------------------------------------------

MOVES = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}
ACTIONS = ["N", "S", "E", "W", "GATHER", "REST"]


def apply_action(agent, world, action):
    """The only place energy changes. Engine-level, strategies cannot bypass."""
    if action in MOVES:
        dx, dy = MOVES[action]
        agent.x = (agent.x + dx) % world.SIZE
        agent.y = (agent.y + dy) % world.SIZE
        agent.energy -= world.MOVE_COST
        agent.total_moved += 1
    elif action == "GATHER":
        f = world.food_at(agent.x, agent.y)
        if f is not None:
            agent.energy += f.energy
            agent.total_gathered += f.energy
            f.alive = False
    # REST: nothing, metabolism still applies
    agent.energy -= world.METABOLISM
    agent.steps += 1
    if agent.energy <= world.DEATH_LEVEL:
        agent.alive = False


# ----------------------------------------------------------------------------
# Strategies
# ----------------------------------------------------------------------------

class RandomStrategy:
    """L0: uniform random baseline. Measures world difficulty."""

    name = "L0-random"

    def act(self, agent, world):
        return world.rng.choice(ACTIONS)


class HeuristicStrategy:
    """L1: hand-written heuristic. Knows what food is (the unfair advantage
    L2 is NOT given)."""

    name = "L1-heuristic"

    def act(self, agent, world):
        if world.food_at(agent.x, agent.y) is not None:
            return "GATHER"
        if agent.energy > 60.0:
            # explore: random walk while healthy
            return world.rng.choice(["N", "S", "E", "W"])
        scored = world.nearest_foods(agent.x, agent.y, k=1)
        if not scored:
            return world.rng.choice(["N", "S", "E", "W"])
        _, dx, dy = scored[0]
        if abs(dx) >= abs(dy):
            return "E" if dx > 0 else "W"
        return "S" if dy > 0 else "N"


class QLearnStrategy:
    """L2: tabular Q-learning with NO external reward.
    Reward = 0 per tick, -10 on death. The agent never learns that 'food'
    exists; it only observes energy dropping and must discover that some
    directions keep it alive longer.

    State = (energy bin, 8-way direction to nearest food, distance bin)
          ~ 10 * 8 * 3 = 240 states
    """

    name = "L2-qlearn"

    ACT_IDX = {a: i for i, a in enumerate(ACTIONS)}

    def __init__(self, alpha=0.1, gamma=0.95, eps0=0.3, eps_min=0.02,
                 eps_decay=0.997):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps0
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.Q = {}
        self.episodes = 0
        self.lifetimes = []

    def state_of(self, agent, world):
        eb = min(int(agent.energy // 10), 9)
        scored = world.nearest_foods(agent.x, agent.y, k=1)
        if not scored:
            return (eb, 0, 3)  # no food visible
        _, dx, dy = scored[0]
        ang = math.atan2(dy, dx)
        direction = int((ang + math.pi) / (math.pi / 4)) % 8
        dist = abs(dx) + abs(dy)
        db = 0 if dist <= 4 else 1 if dist <= 12 else 2
        return (eb, direction, db)

    def _q(self, s, a):
        return self.Q.get((s, a), 0.0)

    def act(self, agent, world):
        s = self.state_of(agent, world)
        if world.rng.random() < self.eps:
            return world.rng.choice(ACTIONS)
        # greedy with random tie-break (unvisited actions have Q=0 which is
        # often the max among negative values; must not always pick 'N')
        best_q = max(self._q(s, a) for a in ACTIONS)
        candidates = [a for a in ACTIONS if self._q(s, a) == best_q]
        return world.rng.choice(candidates)

    def learn(self, s, a, r, s2, done):
        best = max(ACTIONS, key=lambda a2: self._q(s2, a2))
        target = r + (0.0 if done else self.gamma * self._q(s2, best))
        self.Q[(s, a)] = self._q(s, a) + self.alpha * (target - self._q(s, a))

    def on_episode_end(self, lifetime):
        self.lifetimes.append(lifetime)
        self.episodes += 1
        self.eps = max(self.eps_min, self.eps * self.eps_decay)


# ----------------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------------

def run_episode(world, agent, strategy, max_ticks=5000, learn=False):
    """One life. Food was already re-seeded by the caller. Returns lifetime."""
    s = None
    if learn:
        s = strategy.state_of(agent, world)
    while agent.alive and agent.steps < max_ticks:
        a = strategy.act(agent, world)
        if learn:
            e_before = agent.energy
            apply_action(agent, world, a)
            s2 = strategy.state_of(agent, world)
            done = not agent.alive
            # Homeostatic signal: the energy delta IS the reward.
            # No external reward function -- the body itself grades the action.
            r = agent.energy - e_before
            if done:
                r -= 10.0  # death penalty (only terminal signal)
            strategy.learn(s, strategy.ACT_IDX[a], r, s2, done)
            s = s2
        else:
            apply_action(agent, world, a)
        world.step()
    return agent.steps


def baseline(world, strategy, n=300, seed=42):
    """Average lifetime over n lives, fresh food placement each life."""
    agent = Agent()
    lives = []
    for i in range(n):
        world.reset_food(seed + i * 7919)
        agent.reset(world)
        lives.append(run_episode(world, agent, strategy, max_ticks=3000))
    avg = sum(lives) / len(lives)
    median = sorted(lives)[len(lives) // 2]
    print(f"{strategy.name:<14} n={n:4d}  avg={avg:8.1f}  median={median:6d}  "
          f"max={max(lives):6d}")
    return avg


def learn(world, episodes=800, max_ticks=3000, seed=42):
    """Train L2. One episode = one life. Reward: 0 per tick, -10 on death."""
    agent = Agent()
    strat = QLearnStrategy()
    t0 = time.time()
    for ep in range(episodes):
        world.reset_food(seed + ep * 65537)
        agent.reset(world)
        lifetime = run_episode(world, agent, strat,
                               max_ticks=max_ticks, learn=True)
        strat.on_episode_end(lifetime)
        if (ep + 1) % 100 == 0:
            w = strat.lifetimes[-100:]
            print(f"ep {ep+1:4d}  avg(last100)={sum(w)/len(w):7.1f}  "
                  f"eps={strat.eps:.3f}  Q_states={len(strat.Q):4d}")
    w = strat.lifetimes[-100:]
    print(f"\nfinal avg(last100)={sum(w)/100:.1f}  "
          f"median={sorted(w)[50]:d}  elapsed={time.time()-t0:.1f}s")
    return strat


def audit_energy(world, seed=42):
    """Energy conservation: init + gathered == final + moved + metabolism."""
    agent = Agent()
    world.reset_food(seed)
    agent.reset(world)
    init = agent.energy
    while agent.alive and agent.steps < 2000:
        apply_action(agent, world, HeuristicStrategy().act(agent, world))
        world.step()
    lhs = init + agent.total_gathered
    rhs = (agent.energy + agent.total_moved * world.MOVE_COST
           + agent.steps * world.METABOLISM)
    err = abs(lhs - rhs)
    print(f"energy audit: init={init:.1f} gathered={agent.total_gathered:.1f} "
          f"final={agent.energy:.3f} err={err:.2e} -> "
          f"{'OK' if err < 1e-9 else 'FAIL'}")


def main():
    p = argparse.ArgumentParser(description="SEED-0 minimal living agent")
    p.add_argument("--all", action="store_true")
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--learn", action="store_true")
    p.add_argument("--audit", action="store_true")
    p.add_argument("--episodes", type=int, default=800)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    world = World()

    if args.all or args.audit:
        audit_energy(world)
    if args.all or args.baseline:
        print("=== baseline (no learning) ===")
        baseline(world, RandomStrategy(), seed=args.seed)
        baseline(world, HeuristicStrategy(), seed=args.seed)
    if args.all or args.learn:
        print(f"\n=== L2 q-learning, {args.episodes} lives, "
              f"reward: -10 on death ===")
        learn(world, episodes=args.episodes, seed=args.seed)


if __name__ == "__main__":
    main()
