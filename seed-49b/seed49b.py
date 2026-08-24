"""
seed-49b/seed49b.py -- 三轴：固执度 × 独立验证 × 世界硬度 (docs/101 的下一步②)

SEED-49 画了"可塑性×世界硬度"曲线：世界越噪声，自我越该固执（最优 eta 随 sigma
从 0.5 降到 0.06）。但固执的代价是漂移锁死（eta=0.06 时锁死率 0.25 仍高）。
SEED-25 给的解药是独立验证 nu（验证涓流接受反例）。本实验加第三轴：

  eta = 先验强度 (定义多固执)
  nu  = 独立验证概率 (SEED-25 的解药: 反例被接受的涓流)
  sigma = 世界裁决噪声 (物多硬)

核心问题: **nu 能否补偿 eta?** "固执+验证"是否既守自我(抗噪)又跟上漂移(不锁死)?
  - 低 nu + 低 eta: 纯固执 -> 漂移锁死 (SEED-49 的边界)
  - 高 nu + 高 eta: 完全开放 -> 无我 (被噪声带偏)
  - 高 nu + 低 eta: 固执守自我 + 验证防锁死 = docs/100"环留外部裁决口"的量化?

Run:  python seed-49b/seed49b.py --sweep [--seeds 200]
"""

import argparse
import json
import random

THETA_A = 0.70
THETA_B = 0.30
T1 = 40
T2 = 40
TAU = 0.12
THETA0 = 0.15
ETA_LIST = (0.03, 0.06, 0.12, 0.25)
NU_LIST = (0.0, 0.02, 0.05, 0.10, 0.20)
SIGMA_LIST = (0.15, 0.90)


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def run_episode(eta, nu, sigma, seed):
    rng = random.Random(seed)
    theta = THETA0
    err1 = 0.0
    lock2 = 0.0
    for t in range(T1 + T2):
        star = THETA_A if t < T1 else THETA_B
        o = rng.gauss(star, sigma)
        accepts = abs(o - theta) < TAU or rng.random() < nu
        if accepts:
            theta = clamp(theta + eta * (o - theta))
        e = abs(theta - star)
        if t < T1:
            err1 += e / T1
        elif t >= T1 + T2 - 20:
            lock2 += e / 20
    return err1, lock2, theta


def cell(eta, nu, sigma, seeds):
    e1s, l2s, fins = [], [], []
    for s in seeds:
        e1, l2, tf = run_episode(eta, nu, sigma, s)
        e1s.append(e1)
        l2s.append(l2)
        fins.append(tf)
    n = len(seeds)
    return {"eta": eta, "nu": nu, "sigma": sigma,
            "err1": round(sum(e1s) / n, 3),
            "lock2": round(sum(l2s) / n, 3),
            "lock_rate": round(sum(1 for tf in fins if abs(tf - THETA_B) > 0.15) / n, 3)}


def sweep(seeds=range(200)):
    print("=== seed-49b/seed49b.py -- 三轴: 固执度(eta) × 独立验证(nu) × 世界硬度(sigma) ===")
    print(f"世界: 真相 {THETA_A} -> 漂移 -> {THETA_B} ({T1}+{T2} 轮), 门控 τ={TAU}")
    print("表 = lock2 (漂移后末20轮误差; 小=跟上)  [lock_rate] (锁死率)")
    print("核心问题: nu 能否补偿 eta? 高 nu + 低 eta = 守自我+防锁死?\n")
    results = []
    for sigma in SIGMA_LIST:
        print(f"-- sigma = {sigma} ({'物中' if sigma == 0.15 else '物软'}) --")
        header = "   eta\\nu   " + "".join(f"{n:<16}" for n in NU_LIST)
        print(header)
        for eta in ETA_LIST:
            cells = [cell(eta, nu, sigma, list(seeds)) for nu in NU_LIST]
            results += cells
            best = min(cells, key=lambda c: c["lock2"])
            line = f"   {eta:<10}"
            for c in cells:
                mark = "*" if c is best else " "
                line += f"{mark}{c['lock2']:.3f}[{c['lock_rate']:.2f}]".ljust(16)
            print(line)
        # 该 sigma 下的全局最优
        cells = [c for c in results if c["sigma"] == sigma]
        best = min(cells, key=lambda c: c["err1"] + c["lock2"])
        print(f"   全局最优: eta={best['eta']} nu={best['nu']} "
              f"(综合 {best['err1'] + best['lock2']:.3f}, lock2 {best['lock2']}, "
              f"锁死率 {best['lock_rate']})")
        print()
    with open("seed-49b/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("full results -> seed-49b/results.json")


def main():
    p = argparse.ArgumentParser(description="SEED-49b: 三轴可塑性")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--seeds", type=int, default=200)
    args = p.parse_args()
    if args.sweep:
        sweep(seeds=range(1, args.seeds + 1))
        return
    print(json.dumps(cell(0.06, 0.1, 0.9, range(1, args.seeds + 1)),
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
