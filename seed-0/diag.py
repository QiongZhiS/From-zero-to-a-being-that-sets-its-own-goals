"""Diagnostics: what did L2 actually learn?

1. Pure-REST lifetime (theoretical floor/ceiling for lazy policy)
2. After training, run the greedy policy and dump action distribution
"""
import seed0
from seed0 import (World, Agent, run_episode, QLearnStrategy, ACTIONS,
                   RandomStrategy, HeuristicStrategy)

w = World()

# --- 1. pure REST lifetime ------------------------------------------------
class RestStrategy:
    name = "rest-only"
    def act(self, agent, world):
        return "REST"

agent = Agent()
lives = []
for i in range(200):
    w.reset_food(42 + i * 7919)
    agent.reset(w)
    lives.append(run_episode(w, agent, RestStrategy(), max_ticks=3000))
print(f"rest-only  n=200  avg={sum(lives)/len(lives):.1f}")

# --- 2. train then inspect greedy policy ----------------------------------
strat = QLearnStrategy()
for ep in range(600):
    w.reset_food(42 + ep * 65537)
    agent.reset(w)
    run_episode(w, agent, strat, max_ticks=3000, learn=True)
    strat.on_episode_end(agent.steps)

# greedy action distribution over states
from collections import Counter
cnt = Counter()
for (s, a), q in strat.Q.items():
    cnt[a] += 1
print("Q-table action visit counts:", dict(cnt))

# best action per state, aggregated by energy bin (Q keys use int action idx)
best = {}
for (s, a), q in strat.Q.items():
    eb, direction, dist = s
    best.setdefault(eb, Counter())
    best[eb][a] += q
print("\nQ-value sum per energy-bin per action (idx 0-5 = N S E W GATHER REST):")
for eb in sorted(best):
    row = {ACTIONS[i]: round(best[eb][i], 1) for i in range(6)}
    print(f"  energy~{eb*10:3d}: {row}")

# greedy action most preferred per state
greedy = Counter()
for s, actions in best.items():
    if actions:
        ai = actions.most_common(1)[0][0]
        greedy[ACTIONS[ai]] += 1
print("\ngreedy-preferred action over states:", dict(greedy))

# greedy evaluation
agent = Agent()
lives = []
for i in range(200):
    w.reset_food(42 + i * 7919)
    agent.reset(w)
    # force greedy: temporarily zero epsilon
    old = strat.eps
    strat.eps = 0.0
    lives.append(run_episode(w, agent, strat, max_ticks=3000))
    strat.eps = old
print(f"\ngreedy-after-train  n=200  avg={sum(lives)/len(lives):.1f}")
