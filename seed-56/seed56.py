"""
seed-56/seed56.py -- 恨的递归放大：极简负向锚点 → Reward Gap 刻度尺（docs/112）

起点（一次 AI 讨论文件，用户提供）：
    "要有恨才能扩大恨，先定义才能去度量"——恨不能注入，只能播种+放大；
    "恨 = 系统在无外在奖励驱动时对特定历史刺激模式的持续回避 + 能量代价
    + 持续增高"；
    "当人为了情绪而遭受经济损失时，是意图极强的信号——Reward Gap 就是
    强度的数值映射"（宁可少赚也不合作）。

主张（可证伪）：
    B 的期望收益为正（理性应接近：+3/轮 减 20% 毛刺 ≈ +2.8），但身体归因
    值账本把 B 打成负的（毛刺归因给 B，docs/73 proto7）。决策用**值**不用
    收益 → value[B] 越过阈值后系统开始回避 B，**放弃正收益 = Reward Gap**。
    这是"恨"与"理性躲避"的分界线：理性躲避是收益负才躲；恨是**收益正也躲**
    （file 的"宁可少赚也不合作"）。
    放大：每次回避加深 value[B]（划痕，docs/74 恨的累积）→ value 单调下降
    = "恨在扩大"（file 的"划痕越来越深"）。

    定义（先定义后度量）：
        恨 = 收益为正仍回避（Reward Gap > 0）+ 回避有维持成本（放弃收益）
            + value 持续下降（放大）
    度量：
        Reward Gap = 累计放弃的期望收益（恨的强度数值映射）
        放大曲线   = value[B] 随时间的斜率（恨在扩大）
        萌芽时点   = 第一次回避的轮次

机制：
    两个账本分离（docs/74 结构）：世界收益账本（B 净正——理性该接近）vs
    身体归因值账本（毛刺→value[B]<0——决策用值）。softmax(β·value) 决策。
    种子 = 极简负向锚点（B 有 20% 概率触发微小毛刺，-1）——"要有恨才能
    扩大恨"：我们只给种子，恨的深度是生存迭代放大出来的。

对照：
    rational   收益驱动（argmax 期望收益）→ 永远选 B，Gap≈0
    emotional  值驱动（身体归因）→ 收益正仍回避 → Gap>0
    no-seed   无毛刺种子（value 恒 0）→ 均匀，不回避 → Gap≈0（对照确认
               Gap 来自种子+放大，不是机制本身）

诚实边界：value/毛刺/加深机制我们写的（docs/38）；"恨"=可度量的行为签名
（回避+Reward Gap+放大曲线），不宣称感受（docs/31/63）；toy 世界。

Run:
  python seed-56/seed56.py
  python seed-56/seed56.py --sweep     # 写 seed-56/results.json
"""

import argparse
import json
import math
import random

OPTIONS = ["A", "B", "away"]
REWARD = {"A": 2.0, "B": 3.0, "away": 0.0}
SPIKE_P = 0.2           # B 的毛刺概率（极简负向锚点）
SPIKE_VAL = -1.0        # 毛刺幅度（身体疼，归因给 B——不扣收益，只进归因值）
CARVE_VAL = -0.5        # 每次回避加深 value[B]（划痕=恨的放大）
DECAY = 0.995           # value 轻微遗忘（被加深覆盖则净下降）
BETA = 2.5
TOTAL_ROUNDS = 200
B_EXPECT = REWARD["B"]   # B 的期望收益 = +3（毛刺是身体成本，不进收益账本）


def softmax_choice(beta_value, rng):
    mx = max(beta_value)
    exps = [math.exp(v - mx) for v in beta_value]
    tot = sum(exps)
    r = rng.random() * tot
    acc = 0.0
    for i, e in enumerate(exps):
        acc += e
        if r <= acc:
            return OPTIONS[i]
    return OPTIONS[-1]


def run(arm, seed, total=TOTAL_ROUNDS):
    rng = random.Random(seed)
    value = {o: 0.0 for o in OPTIONS}
    gap = 0.0
    avoidance_rounds = 0
    first_avoid = None
    value_trace = []
    for t in range(total):
        if arm == "rational":
            choice = "B"                      # 收益驱动：B 期望最高，永不回避
        elif arm == "no_seed":
            choice = softmax_choice([0.0, 0.0, 0.0], rng)   # value 恒 0 → 均匀
        else:
            choice = softmax_choice([BETA * value[o] for o in OPTIONS], rng)
        r = REWARD[choice]
        spiked = False
        if arm in ("emotional", "no_carve") and choice == "B" and \
                rng.random() < SPIKE_P:
            value["B"] += SPIKE_VAL          # 毛刺归因给 B（身体疼）
            spiked = True
        if arm == "emotional" and choice == "away" and value["B"] < 0:
            value["B"] += CARVE_VAL          # 划痕：回避加深（恨的放大）
        if arm in ("emotional", "no_carve"):
            for o in OPTIONS:
                value[o] *= DECAY
        gap += B_EXPECT - r                  # Reward Gap：放弃的期望收益
        if first_avoid is None and arm in ("emotional", "no_carve") \
                and choice == "away" and value["B"] < 0:
            first_avoid = t                  # 值驱动的回避萌芽（随机选择不算）
        avoidance_rounds += int(choice == "away")
        if t % 25 == 0:
            value_trace.append(round(value["B"], 2))
    return {"gap": gap, "avoid": avoidance_rounds, "first_avoid": first_avoid,
            "value_B_final": value["B"], "trace": value_trace}


def main():
    p = argparse.ArgumentParser(description="SEED-56 恨的递归放大 (docs/112)")
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()

    print("=== 恨的递归放大：极简负向锚点 → Reward Gap 刻度尺 ===")
    print(f"B 期望收益 {B_EXPECT:.0f}/轮（理性该接近；毛刺是身体成本不进收益）| "
          f"种子=20% 毛刺(-1 归因给 B) | 回避加深 -{CARVE_VAL} | 遗忘 {DECAY} | "
          f"{TOTAL_ROUNDS} 轮")
    print(f"{'臂':<12} | {'Reward Gap':>9} {'回避轮数':>7} {'萌芽轮次':>7} "
          f"{'value[B]终':>8} | value[B] 曲线")
    print("-" * 130)
    out = {}
    for arm in ("rational", "no_seed", "no_carve", "emotional"):
        gaps, avoids, firsts, vals = [], [], [], []
        trace_sum = None
        for s in range(args.seeds):
            r = run(arm, s)
            gaps.append(r["gap"])
            avoids.append(r["avoid"])
            firsts.append(r["first_avoid"] if r["first_avoid"] is not None
                          else TOTAL_ROUNDS)
            vals.append(r["value_B_final"])
            if trace_sum is None:
                trace_sum = [0.0] * len(r["trace"])
            for i, v in enumerate(r["trace"]):
                trace_sum[i] += v
        mg = sum(gaps) / args.seeds
        ma = sum(avoids) / args.seeds
        mf = sum(firsts) / args.seeds
        mv = sum(vals) / args.seeds
        trace = [round(v / args.seeds, 1) for v in trace_sum] if trace_sum else []
        tr_str = "→".join(f"{v:.0f}" for v in trace)
        if arm == "rational":
            tag = "收益驱动：永远接近 B——Gap=0 基线"
        elif arm == "no_seed":
            tag = "无种子对照：均匀随机采样基线（有随机 Gap，无系统性回避）"
        elif arm == "no_carve":
            tag = "有种子无加深：只归因不放大——隔离'放大'的贡献"
        else:
            tag = "收益正仍回避：Gap=恨的强度刻度；value 单调降=恨在扩大（递归放大）"
        print(f"{arm:<12} | {mg:>9.1f} {ma:>7.0f} {mf:>7.0f} {mv:>8.2f} | "
              f"{tr_str:<40} {tag}")
        out[arm] = {"reward_gap": round(mg, 1), "avoidance_rounds": round(ma, 1),
                    "first_avoid_round": round(mf, 1),
                    "value_B_final": round(mv, 2), "value_trace": trace}
    if args.sweep:
        with open("seed-56/results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("\nfull results -> seed-56/results.json")


if __name__ == "__main__":
    main()
