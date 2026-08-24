"""
seed-57/seed57.py -- 情比理深的边界：原谅价格的上涨曲线（docs/113）

起点（docs/112 下一步②）：SEED-56 是纯值驱动（收益完全不参与决策），所以
Reward Gap 只说明"值压过理"。现实里两者权衡（讨论记录："经济因素给代价
函数提供刻度尺"）。本 seed 把收益加回决策（混合决策：分数 = β_v×值 +
β_r×收益），扫描 B 的收益 R_B，测"情比理深"的精确形态：
    **原谅价格 R\*(t)** = 使"接近 B"分数 == "远离"分数的临界收益。
    恨加深（value[B] 单调降）→ R\*(t) 上升 = "再多的钱也换不回原谅"
    的量化：不是"多少钱都不行"，而是"需要多少钱随时间上涨"，且低
    诱惑下发散（无价）。

主张（可证伪）：
    1) R_B 低（诱惑小）：回避持续、value 深负、R\* 曲线发散——
       "无价"（时间够长，任何收益都盖不过加深的恨）
    2) R_B 高（诱惑大）：回避被压制/变晚——存在"理压过情"的档位
    3) R\* 曲线随 value 加深单调上升 = 原谅价格的上涨（恨的深度直接
       换算成价格）
    4) 相变：R_B 超过某档后 200 轮内几乎不回避——"情比理深"有边界

机制（同 SEED-56 + 混合决策）：
    B 期望收益 R_B（理性该接近），20% 毛刺归因给 B（身体疼 -1，不进
    收益），回避加深 value[B]（划痕 -0.5），遗忘 0.995。决策分数 =
    β_v×value + β_r×reward（β_v=2.5 值主导、β_r=0.5 收益有分量）。

诚实边界：混合决策的权重/毛刺/加深是我们写的（docs/38）；"原谅价格"
= value 深度换算的临界收益，是行为签名不宣称感受（docs/31/63）；toy。

Run:
  python seed-57/seed57.py
  python seed-57/seed57.py --sweep     # 写 seed-57/results.json
"""

import argparse
import json
import math
import random

OPTIONS = ["A", "B", "away"]
REWARD_A = 2.0
SPIKE_P = 0.2
SPIKE_VAL = -1.0
CARVE_VAL = -0.5
DECAY = 0.995
BETA_V = 2.5          # 值权重（情）
BETA_R = 0.5          # 收益权重（理）
TOTAL = 200
RB_GRID = (3, 6, 10, 15, 25, 40)


def score(o, value, reward):
    return BETA_V * value[o] + BETA_R * reward[o]


def softmax_index(scores, rng):
    mx = max(scores)
    exps = [math.exp(s - mx) for s in scores]
    tot = sum(exps)
    r = rng.random() * tot
    acc = 0.0
    for i, e in enumerate(exps):
        acc += e
        if r <= acc:
            return i
    return len(scores) - 1


def run(R_B, seed, carve=True):
    rng = random.Random(seed)
    value = {o: 0.0 for o in OPTIONS}
    reward = {"A": REWARD_A, "B": R_B, "away": 0.0}
    avoid = 0
    gap = 0.0
    first_avoid = None
    breakeven = []
    for t in range(TOTAL):
        scores = [score(o, value, reward) for o in OPTIONS]
        choice = OPTIONS[softmax_index(scores, rng)]
        r = reward[choice]
        if choice == "B" and rng.random() < SPIKE_P:
            value["B"] += SPIKE_VAL              # 毛刺归因给 B
        if carve and choice == "away" and value["B"] < 0:
            value["B"] += CARVE_VAL              # 回避加深（恨的放大）
        for o in OPTIONS:
            value[o] *= DECAY
        gap += R_B - r                           # 相对 B 期望收益的放弃量
        if choice == "away":
            avoid += 1
            if first_avoid is None:
                first_avoid = t
        if t % 25 == 0:
            # 原谅价格 R*：使 B 分数 == away 分数的临界收益
            r_star = (BETA_V * (value["away"] - value["B"])) / BETA_R
            breakeven.append(round(max(0.0, r_star), 1))
    return {"avoid": avoid, "gap": gap, "value_B": value["B"],
            "first_avoid": first_avoid, "breakeven": breakeven}


def main():
    p = argparse.ArgumentParser(description="SEED-57 情比理深的边界 (docs/113)")
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()

    print("=== 情比理深的边界：原谅价格的上涨曲线 ===")
    print(f"混合决策 分数={BETA_V}×值 + {BETA_R}×收益 | 毛刺 20%(-1 归因 B) | "
          f"回避加深 -{CARVE_VAL} | 遗忘 {DECAY} | {TOTAL} 轮 | "
          f"R* = 盖过恨所需的临界收益")
    print(f"{'R_B':>5} | {'臂':<9} | {'回避轮':>6} {'Gap':>7} {'value终':>7} "
          f"{'萌芽':>5} | R* 终值(200轮)  R* 曲线")
    print("-" * 132)
    out = {}
    for R_B in RB_GRID:
        row = {}
        for carve, label in ((True, "加深"), (False, "无加深")):
            avoids, gaps, vals, firsts = [], [], [], []
            br_sum = None
            for s in range(args.seeds):
                r = run(R_B, s, carve=carve)
                avoids.append(r["avoid"])
                gaps.append(r["gap"])
                vals.append(r["value_B"])
                firsts.append(r["first_avoid"] if r["first_avoid"] is not None
                              else TOTAL)
                if br_sum is None:
                    br_sum = [0.0] * len(r["breakeven"])
                for i, v in enumerate(r["breakeven"]):
                    br_sum[i] += v
            ma = sum(avoids) / args.seeds
            mg = sum(gaps) / args.seeds
            mv = sum(vals) / args.seeds
            mf = sum(firsts) / args.seeds
            br = [round(v / args.seeds, 1) for v in br_sum]
            br_str = "→".join(f"{v:.0f}" for v in br)
            if carve:
                tag = "加深→R* 发散=无价（时间够长，任何收益盖不过加深的恨）"
            else:
                tag = "无加深→R* 有界=恨有价（够大的诱惑可以买回）"
            print(f"{R_B:>5} | {label:<9} | {ma:>6.0f} {mg:>7.0f} {mv:>7.1f} "
                  f"{mf:>5.0f} | {br[-1]:>9.1f}  {br_str:<38} {tag}")
            row[label] = {"avoidance_rounds": round(ma, 1), "gap": round(mg, 1),
                          "value_B": round(mv, 1), "first_avoid": round(mf, 1),
                          "breakeven_final": br[-1], "breakeven_curve": br}
        out[f"RB{R_B}"] = row
    if args.sweep:
        with open("seed-57/results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("\nfull results -> seed-57/results.json")


if __name__ == "__main__":
    main()
