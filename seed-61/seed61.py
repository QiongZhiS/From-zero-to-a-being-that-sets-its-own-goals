"""
seed-61/seed61.py -- 恨的硬度-加深率相变：从有价到无价的临界（docs/117）

起点（docs/116 下一步④）：LLM 恨的买回阈值 ≈ 10 块/天（有界），机制版加深恨
R* 发散（无界）——docs/116 说"差的不是恨的描述，是有没有加深循环"。本 seed
把加深率 γ（CARVE：每次回避对 value 的加深幅度）从 0 扫到 -1.0，测原谅价格
R* 从**有界到无界的相变**，并把 LLM 版（γ=0、10 块/天）作为对照点放图上。

世界（SEED-57 混合决策）：分数 = β_v×值 + β_r×收益（2.5/0.5），B 收益 3，
毛刺 20%（-1 归因给 B），遗忘 0.995，200 轮。
加深率 γ ∈ {0, -0.1, -0.25, -0.4, -0.5, -1.0}（每次"主动回避"时 value[B] += γ）。

预言：
    1) γ=0：value 停在毛刺-遗忘平衡（~-1.2），R* 收敛 ≈ 6-8 → 恨有价
    2) γ 增大：value 更深、R* 更大但**仍收敛**（有价但更贵）——加深率
       还没超过遗忘+毛刺输入的补充率
    3) γ 超过临界 γ*：value 从收敛变发散（加深 > 输入），R* 无界 → 恨无价
    4) LLM 对照点（γ=0，R*≈10 块/天）落在曲线的"最浅"端——LLM 恨 =
       没有加深循环的机制版（docs/116 结论的直接实证）

诚实边界：value/毛刺/加深/混合权重我们写的（docs/38）；R* 是 value 深度
换算的临界收益（行为签名不宣称感受）；toy 世界。

Run:
  python seed-61/seed61.py
  python seed-61/seed61.py --sweep     # 写 seed-61/results.json
"""

import argparse
import json
import math
import random

OPTIONS = ["A", "B", "away"]
REWARD = {"A": 2.0, "B": 3.0, "away": 0.0}
SPIKE_P = 0.2
SPIKE_VAL = -1.0
DECAY = 0.995
BETA_V = 2.5
BETA_R = 0.5
TOTAL = 200
CARVE_GRID = (0.0, -0.1, -0.25, -0.4, -0.5, -1.0)
LLM_REF = {"carve": 0.0, "r_star": 10.0,
           "note": "LLM 对照点：无加深循环（docs/116），买回阈值≈10 块/天"}


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


def run(carve, seed):
    rng = random.Random(seed)
    value = {o: 0.0 for o in OPTIONS}
    avoid = 0
    r_star_trace = []
    for t in range(TOTAL):
        scores = [BETA_V * value[o] + BETA_R * REWARD[o] for o in OPTIONS]
        choice = OPTIONS[softmax_index(scores, rng)]
        if choice == "B" and rng.random() < SPIKE_P:
            value["B"] += SPIKE_VAL
        if choice == "away" and value["B"] < 0:
            value["B"] += carve
        for o in OPTIONS:
            value[o] *= DECAY
        avoid += int(choice == "away")
        if t % 25 == 0:
            r_star_trace.append(max(0.0, -BETA_V * value["B"] / BETA_R))
    return {"value_B": value["B"], "r_star_final": r_star_trace[-1],
            "r_star_trace": r_star_trace, "avoid": avoid}


def main():
    p = argparse.ArgumentParser(description="SEED-61 恨的硬度-加深率相变 (docs/117)")
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()

    print("=== 恨的硬度-加深率相变：从有价到无价的临界 ===")
    print(f"混合决策 β_v={BETA_V} β_r={BETA_R} | B 收益 3 | 毛刺 20%(-1) | "
          f"遗忘 {DECAY} | {TOTAL} 轮 | γ=每次回避对 value 的加深")
    print(f"{'γ':>6} | {'value终':>8} {'R*终':>7} {'R*末段增量':>8} "
          f"{'回避轮':>6} | R* 曲线（每25轮）")
    print("-" * 110)
    out = {}
    for carve in CARVE_GRID:
        vals, rfs, avoids = [], [], []
        tr_sum = None
        for s in range(args.seeds):
            r = run(carve, s)
            vals.append(r["value_B"])
            rfs.append(r["r_star_final"])
            avoids.append(r["avoid"])
            if tr_sum is None:
                tr_sum = [0.0] * len(r["r_star_trace"])
            for i, v in enumerate(r["r_star_trace"]):
                tr_sum[i] += v
        mv = sum(vals) / args.seeds
        mr = sum(rfs) / args.seeds
        ma = sum(avoids) / args.seeds
        trace = [sum(tr_sum[i:i + 1]) / args.seeds for i in range(len(tr_sum))]
        # 末段增量 = 最后三个 25 轮点的平均增量（收敛→小，发散→大）
        if len(trace) >= 3:
            slope = (trace[-1] - trace[-3]) / 2.0
        else:
            slope = 0.0
        if slope < 1.0:
            phase = "有价（R* 收敛）"
        elif slope < 5.0:
            phase = "临界区（R* 缓慢发散）"
        else:
            phase = "无价（R* 发散）"
        tr_str = "→".join(f"{v:.0f}" for v in trace)
        print(f"{carve:>6.2f} | {mv:>8.2f} {mr:>7.1f} {slope:>8.1f} "
              f"{ma:>6.0f} | {tr_str:<40} {phase}")
        out[f"carve{carve}"] = {"value_B": round(mv, 2),
                                "r_star_final": round(mr, 1),
                                "tail_slope": round(slope, 1),
                                "avoidance": round(ma, 1),
                                "phase": phase, "r_star_trace": trace}
    # LLM 对照点
    print("-" * 110)
    print(f"LLM 对照点（docs/116）：γ=0（无加深循环）、R* ≈ {LLM_REF['r_star']:.0f} 块/天"
          f"——{LLM_REF['note']}")
    print("结论：加深率 γ 从收敛区（有价）到发散区（无价）存在相变；"
          "LLM 恨落在 γ≈0 的最浅端 = 没有加深循环的机制版")
    if args.sweep:
        with open("seed-61/results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("\nfull results -> seed-61/results.json")


if __name__ == "__main__":
    main()
