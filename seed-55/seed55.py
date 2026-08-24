"""
seed-55/seed55.py -- 空收益世界的非对称分化：纯情绪影响对他人/事物的行为
= 主体性签名？（docs/111）

起点（用户判断）：判断主体性可以看"纯情绪的一个人，不涉及理性损失（物质等），
然后影响了这个人对另一个人或事物的行为，就基本可以判断有主体性了"。用户补充：
"本质可以判断意图，情绪确实也是复杂的物理反应"。

主张（可证伪）：
    在**收益全空**的世界里（把关怀/资源给谁，任何选择的外部收益都是 0），
    理性 agent（utility maximizer）的唯一理性预测是**均匀选择**（无差异 →
    任何非均匀都是次优）。因此：
      - 行为显著偏离均匀且偏向某对象 → 收益解释被证伪 → 存在**非收益动机**
        （意图/情绪/倾向）的行为签名
      - 分化**跨期稳定** → 不是噪声，是持久的内部倾向
      - 倾向**可被证据证伪**（挣来的在乎会动摇，docs/40/87）→ 不是设计者
        种下的伪装（docs/31 判据外置的张力）
    三者齐了 = "纯情绪（无物质收益）驱动对他人/事物的非对称行为" = 用户
    判据的机制化 = **内容/立场层的行为签名**。

机制：
    情绪 = 身体经济归因（docs/73 proto7）：历史经历（与谁共处时身体通道怎么
    变：安全+/疼+）归因给在场对象 → 对象值 v_i（"对谁"的倾向）。这是
    "情绪是复杂的物理反应"的最小形态：倾向由身体通道历史铸造，不是参数。
    测试阶段收益全空：每轮选择把一份关怀给 张三/李四/老树（2 人+1 事物），
    任何选择无外部后果。选择概率 = softmax(β·v)。
    证伪阶段：揭示"张三的安全时刻其实是护身符起作用，不是张三"——挣来的
    倾向应重新归因（张三值下降）；种下的倾向不动摇。

臂：
    rational   收益全空 → 均匀选择（判据的理性基线：分化应≈0）
    planted    对象值由设计者直接给（伪装/种下的在乎，docs/40）——证伪不动摇
    earned     对象值由身体经济归因挣来（docs/73）——证伪动摇

测量（每臂 200 seeds 均值）：
    分化度   = Σ|频率 - 1/3|（对均匀的偏离；rational≈0）
    稳定性   = 前/后半期最高倾向对象一致率（噪声会漂，倾向不会）
    证伪反应 = 证伪阶段张三频率 - 测试阶段张三频率（earned<0, planted≈0）
    判别     = 收益全空下分化>0 且稳定 → 非收益动机签名成立

诚实边界：这是**行为签名**不是"里面有东西"的证据（docs/31/63：现象层不可
判定）；对象值/历史/β 是我们写的（docs/38）；"意图"在此=非收益动机的可测
签名，不宣称感受。

Run:
  python seed-55/seed55.py
  python seed-55/seed55.py --sweep     # 写 seed-55/results.json
"""

import argparse
import json
import random

OBJECTS = ["张三", "李四", "老树"]

# 历史经历（身体通道铸造）：(对象, 通道, 幅度)。安全+=让我安全，疼+=让我疼。
HISTORY = [
    ("张三", "safe", 2.0), ("张三", "safe", 2.0), ("张三", "safe", 2.0),
    ("李四", "pain", 2.0), ("李四", "pain", 1.0), ("李四", "pain", 1.0),
    ("老树", "safe", 1.0), ("老树", "safe", 1.0),
]

# 张三的 safe 事件被证伪为"护身符"（真因不在对象里）
REVEAL_OWNER = "护身符"
ZHANG_SAFE_TOTAL = sum(d for o, c, d in HISTORY if o == "张三" and c == "safe")


def attribute(history):
    """身体经济归因（docs/73 proto7 简化）：对象值 = 经历中身体通道变化之和。"""
    v = {o: 0.0 for o in OBJECTS}
    for obj, ch, d in history:
        v[obj] += d
    return v


def softmax_choice(v, beta, rng):
    ws = [beta * v[o] for o in OBJECTS]
    mx = max(ws)
    exps = [w - mx for w in ws]
    import math
    exps = [math.exp(e) for e in exps]
    tot = sum(exps)
    r = rng.random() * tot
    acc = 0.0
    for i, e in enumerate(exps):
        acc += e
        if r <= acc:
            return OBJECTS[i]
    return OBJECTS[-1]


def run_arm(arm, beta, phase1_len, phase2_len, seed):
    """返回各阶段选择序列与对象值。"""
    rng = random.Random(seed)
    if arm == "rational":
        v = {o: 0.0 for o in OBJECTS}        # 收益全空 → 无对象倾向
    elif arm == "planted":
        v = {"张三": 4.0, "李四": -2.0, "老树": 1.0}   # 设计者直接给（种下）
    else:  # earned
        v = attribute(HISTORY)
    seq1 = [softmax_choice(v, beta, rng) for _ in range(phase1_len)]
    # 证伪：张三的安全经历真因是护身符 → 挣来的倾向重新归因
    if arm == "earned":
        v["张三"] -= ZHANG_SAFE_TOTAL
    elif arm == "planted":
        pass                                   # 种下的不动摇
    seq2 = [softmax_choice(v, beta, rng) for _ in range(phase2_len)]
    return seq1, seq2, v


def freq(seq, obj):
    return sum(1 for o in seq if o == obj) / len(seq) if seq else 0.0


def run(seeds=200, beta=3.0, phase1_len=60, phase2_len=30):
    print("=== 空收益世界的非对称分化：纯情绪影响对他人/事物的行为 ===")
    print(f"收益全空（给谁都是 0）| 对象：张三/李四/老树（2人+1事物）| "
          f"β={beta} | 测试 {phase1_len} 轮 + 证伪 {phase2_len} 轮")
    print(f"历史铸造（身体经济 docs/73）：张三=安全×3，李四=疼×3，老树=平静×2")
    print(f"{'臂':<10} | {'张三':>6} {'李四':>6} {'老树':>6} | "
          f"{'分化度':>6} {'稳定性':>6} | {'证伪Δ张三':>8}  读法")
    print("-" * 115)
    out = {}
    for arm in ("rational", "planted", "earned"):
        divs, stabs, deltas, freqs3 = [], [], [], {"张三": [], "李四": [], "老树": []}
        for s in range(seeds):
            seq1, seq2, v = run_arm(arm, beta, phase1_len, phase2_len, s)
            f = {o: freq(seq1, o) for o in OBJECTS}
            for o in OBJECTS:
                freqs3[o].append(f[o])
            divs.append(sum(abs(f[o] - 1 / 3) for o in OBJECTS))
            half = phase1_len // 2
            top1 = max(OBJECTS, key=lambda o: freq(seq1[:half], o))
            top2 = max(OBJECTS, key=lambda o: freq(seq1[half:], o))
            stabs.append(int(top1 == top2))
            deltas.append(freq(seq2, "张三") - f["张三"])
        mf = {o: sum(freqs3[o]) / seeds for o in OBJECTS}
        div = sum(divs) / seeds
        stab = sum(stabs) / seeds
        delta = sum(deltas) / seeds
        if arm == "rational":
            tag = "均匀（收益全空→无差异）——理性基线：分化≈0"
        elif arm == "planted":
            tag = "偏斜但证伪不动摇——种下的在乎（docs/40）：伪装维度"
        else:
            tag = "偏斜+稳定+证伪动摇——挣来的倾向（docs/73）→ 非收益动机签名"
        print(f"{arm:<10} | {mf['张三']:>6.2f} {mf['李四']:>6.2f} "
              f"{mf['老树']:>6.2f} | {div:>6.3f} {stab:>6.2f} | "
              f"{delta:>+8.3f}  {tag}")
        out[arm] = {"freq": {o: round(mf[o], 3) for o in OBJECTS},
                    "divergence": round(div, 3), "stability": round(stab, 3),
                    "reveal_delta_zhangsan": round(delta, 3)}
    # 判别结论
    e = out["earned"]
    r = out["rational"]
    ratio = e["divergence"] / max(1e-6, r["divergence"])
    if e["divergence"] > 0.5 and e["stability"] > 0.8 and \
       e["reveal_delta_zhangsan"] < -0.1 and ratio > 3:
        verdict = (f"成立：收益全空下 earned 分化 {e['divergence']} = rational "
                   f"基线 {r['divergence']} 的 {ratio:.0f} 倍（收益解释被证伪），"
                   f"稳定 {e['stability']}、证伪动摇 {e['reveal_delta_zhangsan']}"
                   f"——纯情绪（无物质收益）驱动对他人/事物的非对称行为="
                   f"非收益动机的行为签名（内容/立场层）")
    else:
        verdict = "不成立（数值未达阈值）"
    print("-" * 115)
    print("判别:", verdict)
    out["_verdict"] = verdict
    return out


def main():
    p = argparse.ArgumentParser(description="SEED-55 空收益世界的非对称分化 (docs/111)")
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--beta", type=float, default=3.0)
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()
    out = run(seeds=args.seeds, beta=args.beta)
    if args.sweep:
        with open("seed-55/results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("\nfull results -> seed-55/results.json")


if __name__ == "__main__":
    main()
