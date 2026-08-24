"""
SEED-49: 定义的可塑性测绘 (docs/100 的下一步候选)

docs/100: "人的定义就是主观唯心的，这个定义决定人怎么反馈，定义的形成却是外在
×内在共同的结果，含我也含物" -> 定义→反馈→铸造→定义是闭环。本实验测绘
"定义"该多可塑——把 docs/100 的"平衡"从口号画成一条可测的曲线:

  先验强度 = 学习率 eta  (定义多快被世界改写; 小=固执, 大=可塑)
  裁决强度 = 世界噪声 sigma (物多硬; 小=硬, 大=软)

机制 = SEED-25 (自指门控: 只接受确认信念的观察, nu=0.05 独立验证涓流) +
      世界中期漂移 (真相 0.7 -> 0.3) + 观察噪声 (裁决硬度旋钮)。

三区预测 (docs/100 的"完全可塑=没有自我, 完全刚性=锁死"):
  锁死区: eta 小 -> 漂移后跟不上新真相 (SEED-25 门控锁死)
  无我区: eta 大 + sigma 大 -> 定义被单次噪声带偏, 没有稳定自我
          (= SEED-48 的 recency 臂, 在自我模型上的复现)
  平衡带: 随世界硬度移动 -- 物越软(sigma 大), 定义越该固执(eta 小);
          物越硬(sigma 小), 定义越该开放(eta 大)

Run:  python seed-49/seed49.py --sweep        # 二维网格 + results.json
      python seed-49/seed49.py --eta 0.12 --sigma 0.15   # 单点
"""

import argparse
import json
import random

THETA_A = 0.70        # 阶段1真相 (初始世界)
THETA_B = 0.30        # 阶段2真相 (漂移后)
T1 = 40               # 阶段1轮数 (短窗口: 让"未收敛"成为主要代价, 平衡才显形)
T2 = 40               # 阶段2轮数 (漂移后窗口同样短: eta 小=响应慢=跟不上的代价可见)
TAU = 0.12            # 确认容差 (SEED-25)
NU = 0.05             # 独立验证涓流 (SEED-25 解药; 固定小值, 防纯锁死)
THETA0 = 0.15         # 初始定义 (起步错, SEED-25 锁死场景)
LOCK_E = 0.25         # 锁死检测: 漂移后末20轮平均误差 > 此 = 没跟上新真相
ETA_LIST = (0.03, 0.06, 0.12, 0.25, 0.50)
SIGMA_LIST = (0.05, 0.15, 0.40, 0.90)


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def run_episode(eta, sigma, seed):
    """一个定义的生命: 阶段1追踪 0.7, 漂移后追踪 0.3。返回 (err1, err2, lock2, theta_final)。"""
    rng = random.Random(seed)
    theta = THETA0
    err1 = 0.0
    err2 = 0.0
    lock2 = 0.0
    for t in range(T1 + T2):
        star = THETA_A if t < T1 else THETA_B
        o = rng.gauss(star, sigma)
        accepts = abs(o - theta) < TAU or rng.random() < NU
        if accepts:
            theta = clamp(theta + eta * (o - theta))
        e = abs(theta - star)
        if t < T1:
            err1 += e / T1
        else:
            err2 += e / T2
            if t >= T1 + T2 - 20:
                lock2 += e / 20
    return err1, err2, lock2, theta


def cell(eta, sigma, seeds):
    e1s, e2s, l2s, fins = [], [], [], []
    for s in seeds:
        e1, e2, l2, tf = run_episode(eta, sigma, s)
        e1s.append(e1)
        e2s.append(e2)
        l2s.append(l2)
        fins.append(tf)
    n = len(seeds)
    return {"eta": eta, "sigma": sigma,
            "err1": round(sum(e1s) / n, 3),   # 阶段1: 追踪初始真相 (起步校准+稳定)
            "err2": round(sum(e2s) / n, 3),   # 阶段2: 跟漂移 (响应+稳定性)
            "lock2": round(sum(l2s) / n, 3),  # 漂移后末20轮误差 (锁死+波动)
            # 锁死率: 漂移后定义最终状态没爬到新真相附近 (状态本身判据,
            # 不被噪声波动污染 -- 抖动≠没跟上)
            "lock_rate": round(sum(1 for tf in fins if abs(tf - THETA_B) > 0.15) / n, 3)}


def sweep(seeds=range(200)):
    print("=== SEED-49: 定义的可塑性测绘 (docs/100) ===")
    print(f"世界: 真相 {THETA_A} -> 漂移 -> {THETA_B} ({T1}+{T2} 轮), "
          f"观察噪声 sigma(物硬), 学习率 eta(先验强度/固执度), 门控 nu={NU}")
    print("表1 = 阶段2末20轮误差 lock2 (漂移响应; 小=跟上了, 大=锁死)")
    print("表2 = 阶段1平均误差 err1 (追踪+稳定性; 无我区: eta大×sigma大 这里高)")
    print("表3 = 锁死率 (漂移后定义最终状态没爬到新真相: |theta_final-0.3| > 0.15)")
    print("[X] = 该 sigma 下最优 eta\n")
    results = []
    for sigma in SIGMA_LIST:
        cells = [cell(eta, sigma, list(seeds)) for eta in ETA_LIST]
        results += cells
    for label, key in (("表1 lock2", "lock2"), ("表2 err1", "err1"),
                       ("表3 锁死率", "lock_rate")):
        print(f"-- {label} --")
        print(f"   sigma\\eta  " + "".join(f"{e:<9}" for e in ETA_LIST))
        for sigma in SIGMA_LIST:
            cells = [c for c in results if c["sigma"] == sigma]
            best = min(cells, key=lambda c: c[key])
            line = f"   {sigma:<10}"
            for c in cells:
                txt = f"[{c[key]:.3f}]" if c is best else f"{c[key]:.3f}"
                line += f"{txt:<9}"
            print(line)
        print()
    # 平衡带: 每 sigma 的最优 eta -- 综合代价 = 阶段1追踪 + 漂移响应
    print("平衡带 (物硬->开放, 物软->固执; 综合代价 = err1 + lock2):")
    for sigma in SIGMA_LIST:
        cells = [c for c in results if c["sigma"] == sigma]
        best = min(cells, key=lambda c: c["err1"] + c["lock2"])
        print(f"   sigma={sigma:<5} 最优 eta = {best['eta']}  "
              f"(综合 {best['err1'] + best['lock2']:.3f}; "
              f"lock2 {best['lock2']} / err1 {best['err1']} / "
              f"锁死率 {best['lock_rate']})")
    with open("seed-49/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("\nfull results -> seed-49/results.json")


def main():
    p = argparse.ArgumentParser(description="SEED-49: 定义的可塑性测绘")
    p.add_argument("--eta", type=float, default=0.12)
    p.add_argument("--sigma", type=float, default=0.15)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--seeds", type=int, default=200)
    args = p.parse_args()
    if args.sweep:
        sweep(seeds=range(1, args.seeds + 1))
        return
    print(json.dumps(cell(args.eta, args.sigma, range(1, args.seeds + 1)),
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
