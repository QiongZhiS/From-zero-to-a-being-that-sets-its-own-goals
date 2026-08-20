"""
SEED-20: B0 scale test -- is size a variable?

Route B question: our toy conclusions (sessile convergence, homogeneity)
were measured at N=300. Does behavior diversity CHANGE with population
size, or is the mechanism scale-independent?

Vectorized (numpy) re-implementation of SEED-16's genetic regime:
preference (specialize A or B) evolves by mutation; world has two food
zones (A sparse-rich, B dense-poor). If division of labor (bimodal pref)
emerges more cleanly at scale, size matters. If the distribution is the
same shape at N=300 and N=10000, the mechanism is scale-independent.

Run:  python seed20.py --n 5000 --ticks 2000 --seed 42
"""

import argparse
import numpy as np

SIZE = 64
METABOLISM = 0.4
MOVE_COST = 0.2
FOOD_ENERGY_A = 80
FOOD_ENERGY_B = 30
SPLIT_ENERGY = 100.0
INIT_ENERGY = 150.0
MUTATION = 0.05
FOOD_REGEN = 0.10
A_X_MAX = 24
B_X_MIN = 40
A_N = 40
B_N = 96


def make_world(seed, food_scale=1.0):
    rng = np.random.default_rng(seed)
    a_n = max(1, int(A_N * food_scale))
    b_n = max(1, int(B_N * food_scale))
    fa = np.column_stack([rng.integers(0, A_X_MAX + 1, a_n),
                          rng.integers(0, SIZE, a_n)])
    fb = np.column_stack([rng.integers(B_X_MIN, SIZE, b_n),
                          rng.integers(0, SIZE, b_n)])
    return fa, fb


def torus_dxdy(px, py, fx, fy):
    dx = (fx - px + SIZE // 2) % SIZE - SIZE // 2
    dy = (fy - py + SIZE // 2) % SIZE - SIZE // 2
    return dx, dy


def simulate(n_init=500, max_pop=10000, ticks=2000, seed=42,
             report_every=200, food_scale=0.0):
    rng = np.random.default_rng(seed)
    # SEED-16: 136 food sustains ~400 pop. Scale food with target pop.
    if food_scale <= 0:
        food_scale = n_init / 400.0
    fa, fb = make_world(seed, food_scale)

    # agent arrays (fixed size, alive mask)
    pos = rng.integers(0, SIZE, (max_pop, 2)).astype(np.float64)
    pref = np.zeros(max_pop)
    energy = np.zeros(max_pop)
    alive = np.zeros(max_pop, bool)
    n0 = n_init
    pos[:n0] = rng.integers(0, SIZE, (n0, 2))
    pref[:n0] = rng.random(n0)
    energy[:n0] = INIT_ENERGY
    alive[:n0] = True

    # food arrays
    food_pos = np.vstack([fa, fb])
    food_zone = np.array([0] * len(fa) + [1] * len(fb))  # 0=A, 1=B
    food_energy = np.array([FOOD_ENERGY_A] * len(fa) +
                           [FOOD_ENERGY_B] * len(fb), dtype=float)
    food_alive = np.ones(len(food_pos), bool)
    food_regen = FOOD_REGEN

    births = 0
    starved = 0

    for t in range(ticks):
        n = int(alive.sum())
        if n == 0:
            print("extinct")
            break
        idx = np.where(alive)[0]
        px = pos[idx, 0]
        py = pos[idx, 1]
        pe = energy[idx]
        pp = pref[idx]

        # --- act: decide target zone by pref, find nearest food there ---
        # go to B with prob pref (vectorized coin flip)
        go_b = rng.random(n) < pp
        # for each agent, distance to all foods in its zone
        tgt = np.full(n, -1, int)
        for zone in (0, 1):
            mask = go_b == (zone == 1)
            if not mask.any():
                continue
            zf = np.where(food_zone == zone)[0]          # global idx of zone foods
            ff = food_alive[zf]                          # alive mask within zone
            if not ff.any():
                continue
            gidx = zf[ff]                                # global idx of alive zone foods
            fpos = food_pos[gidx]
            # distances (N_food x N_agents)
            dx = (fpos[:, 0][:, None] - px[mask][None, :] + SIZE // 2) \
                % SIZE - SIZE // 2
            dy = (fpos[:, 1][:, None] - py[mask][None, :] + SIZE // 2) \
                % SIZE - SIZE // 2
            d = np.abs(dx) + np.abs(dy)
            best = np.argmin(d, axis=0)
            tgt[np.where(mask)[0]] = gidx[best]          # GLOBAL index

        # --- act: gather if food underfoot, else move toward target ---
        for i, gi in enumerate(idx):
            if tgt[i] < 0:
                continue
            fx = food_pos[tgt[i], 0]
            fy = food_pos[tgt[i], 1]
            if int(px[i]) == int(fx) and int(py[i]) == int(fy) \
                    and food_alive[tgt[i]]:
                energy[gi] += food_energy[tgt[i]]
                food_alive[tgt[i]] = False
                # no move
                continue
            dx, dy = torus_dxdy(int(px[i]), int(py[i]), int(fx), int(fy))
            if abs(dx) >= abs(dy):
                step = (1, 0) if dx > 0 else (-1, 0)
            else:
                step = (0, 1) if dy > 0 else (0, -1)
            pos[gi, 0] = (pos[gi, 0] + step[0]) % SIZE
            pos[gi, 1] = (pos[gi, 1] + step[1]) % SIZE
            energy[gi] -= MOVE_COST

        energy[idx] -= METABOLISM

        # --- reproduction ---
        split_mask = energy[idx] >= SPLIT_ENERGY
        if split_mask.any():
            n_split = int(split_mask.sum())
            free = np.where(~alive)[0][:n_split]
            if len(free) > 0:
                k = min(len(free), n_split)
                f_idx = free[:k]
                s_idx = idx[split_mask][:k]
                pos[f_idx] = pos[s_idx] + rng.integers(-1, 2, (k, 2))
                pos[f_idx] %= SIZE
                pref[f_idx] = np.clip(pref[s_idx] +
                                      rng.normal(0, MUTATION, k), 0, 1)
                energy[f_idx] = energy[s_idx] / 2.0
                energy[s_idx] /= 2.0
                alive[f_idx] = True
                births += k

        # --- death ---
        dead = alive & (energy <= 0)
        starved += int(dead.sum())
        alive[dead] = False

        # --- food regen ---
        regen_mask = ~food_alive & (rng.random(len(food_pos)) < food_regen)
        food_alive[regen_mask] = True

        if (t + 1) % report_every == 0:
            n = int(alive.sum())
            p = pref[alive]
            lo = float((p < 0.25).mean())
            hi = float((p > 0.75).mean())
            print(f"t={t+1:5d}  pop={n:5d}  A-spec={lo:.3f}  "
                  f"mid={1-lo-hi:.3f}  B-spec={hi:.3f}")

    n = int(alive.sum())
    p = pref[alive]
    lo = int((p < 0.25).sum())
    hi = int((p > 0.75).sum())
    mid = n - lo - hi
    print(f"\nfinal: pop={n}  A-spec={lo}  mid={mid}  B-spec={hi}  "
          f"births={births}  starved={starved}")
    # compare with SEED-16 genetic (n=300): 100A / 62mid / 238B
    print("SEED-16 reference (n=300): 100A / 62mid / 238B")


def main():
    p = argparse.ArgumentParser(description="SEED-20 B0 scale test")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--max-pop", type=int, default=10000)
    p.add_argument("--ticks", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report", type=int, default=200)
    p.add_argument("--food-scale", type=float, default=1.0,
                   help="scale food count with population")
    args = p.parse_args()
    simulate(n_init=args.n, max_pop=args.max_pop, ticks=args.ticks,
             seed=args.seed, report_every=args.report,
             food_scale=args.food_scale)


if __name__ == "__main__":
    main()
