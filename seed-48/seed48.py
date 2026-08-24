"""
seed-48/seed48.py -- 去噪先验的可迁移性 (docs/99 的机制侧验证)

用户: "人的去噪能力是不是就是一种泛化能力？人可以在未知领域通过去噪去学习?"
docs/78 概念地图: 去噪⊂压缩⊂泛化 (包含关系)。本实验测机制侧最锋利的一刀:
一个**只有**"世界可压缩(重复=结构)"这一个先验、**零领域知识**的 agent,
能否在多个结构完全不同的未知世界里活下来并迁移?

关键: "去噪"必须和"无去噪"真正对立。第一版用显示污染失败了 -- 决策层的
"选均值最高"本身就是隐式去噪, K=1 靠均值也能过滤。改为**回报反转噪声**:
世界给的回报有 30% 概率被反转(观察到的 r 是假的)。此时:
  去噪   = 只信"样本 >= K"的关联 + 取均值  (不信任单次观察)
  无去噪 = recency 式: 信最近一次观察 (覆盖更新)  (信任单次观察 -> 被反转带偏)
  随机   = 基线

三世界 (结构完全不同, 各带 30% 回报反转噪声):
  α 静态:  s -> (s+1)%N   (循环结构)
  β 条件:  s -> (s*2+1)%N (与 α 完全不同的代数结构)
  γ 漂移:  前 50 轮 s->(s+1)%N, 后 50 轮换成 s->(s+3)%N
           (docs/26 双刃剑: 提取的旧结构在漂移后会不会锁死? 去噪者稳定但慢,
           recency 者灵活但被噪声带偏 -- 谁在漂移后恢复快?)

测 (每世界 200 seeds, 取均值):
  1) 各世界内: 三臂的存活率 / 平均每轮回报 / 末段准确率
  2) γ 漂移: 漂移后能量轨迹与末段准确率 (锁死 vs 恢复)

Run: python seed-48/seed48.py [--seeds N] [--quick]
"""

import argparse
import random

N = 4              # 信号数 = 动作数
NOISE_P = 0.30     # 回报反转概率 (观察到的 r 有 30% 概率是假的)
EPS = 0.20         # 探索率 (ε-greedy)
START_E = 30.0
METAB = 0.2        # 每轮代谢
R_CORRECT = 2.0
R_WRONG = -1.0
TURNS = 100        # γ 世界: 50 轮后漂移
K_DENOISE = 2      # 去噪门槛: 样本 >= 2 才信 (重复才值得信)


def make_world(kind):
    def f(s, t):
        if kind == "alpha":
            return (s + 1) % N
        if kind == "beta":
            return (s * 2 + 1) % N
        if kind == "gamma":
            return (s + 1) % N if t < TURNS // 2 else (s + 3) % N
    return f


def run_denoise(kind, seed, turns=TURNS):
    """A 去噪: 均值 + 样本门槛 K。返回 (存活, 总回报, 末段准确率, 漂移后能量快照)。"""
    rng = random.Random(seed)
    f = make_world(kind)
    counts = [[0] * N for _ in range(N)]
    sums = [[0.0] * N for _ in range(N)]
    energy = START_E
    total_r = 0.0
    cl = ln = 0
    post = []
    for t in range(turns):
        s = rng.randrange(N)
        if rng.random() < EPS:
            a = rng.randrange(N)
        else:
            cand = [(s, a) for a in range(N) if counts[s][a] >= K_DENOISE]
            a = (max(cand, key=lambda sa: sums[sa[0]][sa[1]] / counts[sa[0]][sa[1]])[1]
                 if cand else rng.randrange(N))
        r_true = R_CORRECT if a == f(s, t) else R_WRONG
        r_obs = -r_true if rng.random() < NOISE_P else r_true   # 回报反转噪声
        counts[s][a] += 1
        sums[s][a] += r_obs
        energy += r_true - METAB
        total_r += r_true
        if t >= turns - 10:
            cl += 1 if a == f(s, t) else 0
            ln += 1
        if kind == "gamma" and t >= TURNS // 2 and (t - TURNS // 2) % 10 == 0:
            post.append(energy)
        if energy <= 0:
            return False, total_r, cl / max(1, ln), post
    return True, total_r, cl / max(1, ln), post


def run_denoise_reset(kind, seed, turns=TURNS):
    """D 去噪+重置 (docs/25/27 独立验证的机制版): 同去噪, 但每 10 轮检查
    最近 10 轮观测回报均值 -- 低于阈值 -> 怀疑模型失效(自指不可靠, 需要外部
    信号裁决) -> 重置经验表重新学。打破旧结构不是去噪自带的能力。"""
    rng = random.Random(seed)
    f = make_world(kind)
    counts = [[0] * N for _ in range(N)]
    sums = [[0.0] * N for _ in range(N)]
    energy = START_E
    total_r = 0.0
    cl = ln = 0
    post = []
    recent = []            # 最近观测回报 (检查用)
    best_recent = -9.0     # 历史最好的最近均值 (重置只看"从好变差", 探索期不误触发)
    for t in range(turns):
        s = rng.randrange(N)
        if rng.random() < EPS:
            a = rng.randrange(N)
        else:
            cand = [(s, a) for a in range(N) if counts[s][a] >= K_DENOISE]
            a = (max(cand, key=lambda sa: sums[sa[0]][sa[1]] / counts[sa[0]][sa[1]])[1]
                 if cand else rng.randrange(N))
        r_true = R_CORRECT if a == f(s, t) else R_WRONG
        r_obs = -r_true if rng.random() < NOISE_P else r_true
        counts[s][a] += 1
        sums[s][a] += r_obs
        recent.append(r_obs)
        energy += r_true - METAB
        total_r += r_true
        if t >= turns - 10:
            cl += 1 if a == f(s, t) else 0
            ln += 1
        if kind == "gamma" and t >= TURNS // 2 and (t - TURNS // 2) % 10 == 0:
            post.append(energy)
        # 独立验证 (docs/25/27): 只看"从好变差" -- 曾经学会(best>0.5)之后,
        # 最近均值掉到最好的一半以下 -> 模型可能失效 -> 重置 (探索期 best 没起来,
        # 不会误触发; 漂移后旧模型退化 -> 触发)
        if len(recent) >= 10:
            cur = stats(recent[-10:])
            if cur > best_recent:
                best_recent = cur
            elif best_recent > 0.5 and cur < best_recent * 0.5:
                counts = [[0] * N for _ in range(N)]
                sums = [[0.0] * N for _ in range(N)]
                best_recent = cur   # 重置后从当前水平重新评估 (防连锁重置)
        if energy <= 0:
            return False, total_r, cl / max(1, ln), post
    return True, total_r, cl / max(1, ln), post


def run_recency(kind, seed, turns=TURNS):
    """B 无去噪 (recency): 信任最近一次观察, 覆盖更新。"""
    rng = random.Random(seed)
    f = make_world(kind)
    vals = [[0.0] * N for _ in range(N)]
    seen = [[False] * N for _ in range(N)]
    energy = START_E
    total_r = 0.0
    cl = ln = 0
    post = []
    for t in range(turns):
        s = rng.randrange(N)
        if rng.random() < EPS:
            a = rng.randrange(N)
        else:
            cand = [a for a in range(N) if seen[s][a]]
            a = (max(cand, key=lambda a: vals[s][a]) if cand else rng.randrange(N))
        r_true = R_CORRECT if a == f(s, t) else R_WRONG
        r_obs = -r_true if rng.random() < NOISE_P else r_true
        vals[s][a] = r_obs            # 覆盖: 最近一次观察就是"真相"
        seen[s][a] = True
        energy += r_true - METAB
        total_r += r_true
        if t >= turns - 10:
            cl += 1 if a == f(s, t) else 0
            ln += 1
        if kind == "gamma" and t >= TURNS // 2 and (t - TURNS // 2) % 10 == 0:
            post.append(energy)
        if energy <= 0:
            return False, total_r, cl / max(1, ln), post
    return True, total_r, cl / max(1, ln), post


def run_random(kind, seed, turns=TURNS):
    rng = random.Random(seed)
    f = make_world(kind)
    energy = START_E
    total_r = 0.0
    cl = ln = 0
    post = []
    for t in range(turns):
        a = rng.randrange(N)
        r_true = R_CORRECT if a == f(rng.randrange(N), t) else R_WRONG
        energy += r_true - METAB
        total_r += r_true
        if t >= turns - 10:
            cl += 1 if a == f(rng.randrange(N), t) else 0
            ln += 1
        if kind == "gamma" and t >= TURNS // 2 and (t - TURNS // 2) % 10 == 0:
            post.append(energy)
        if energy <= 0:
            return False, total_r, cl / max(1, ln), post
    return True, total_r, cl / max(1, ln), post


def stats(vals):
    n = len(vals)
    return sum(vals) / n if n else 0.0


def main():
    p = argparse.ArgumentParser(description="SEED-48: 去噪先验的可迁移性")
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--quick", action="store_true", help="10 seeds 快速看结构")
    args = p.parse_args()
    seeds = 10 if args.quick else args.seeds
    runners = [("去噪 K=2", run_denoise), ("去噪+重置", run_denoise_reset),
               ("无去噪 recency", run_recency), ("随机基线", run_random)]

    print(f"=== seed-48/seed48.py -- 去噪先验的可迁移性 (docs/99) ===")
    print(f"世界: 信号数 {N}, 回报反转噪声 {NOISE_P:.0%}, {TURNS} 轮, {seeds} seeds\n")

    for kind, label in [("alpha", "α 静态"), ("beta", "β 条件"), ("gamma", "γ 漂移")]:
        print(f"-- 世界 {label} (结构: {kind}) --")
        print(f"   {'臂':<12}{'存活率':<8}{'回报/轮':<9}{'末段准确率':<10}{'漂移后能量(γ)'}")
        for name, runner in runners:
            surv, tot, acc, post = [], [], [], []
            for s in range(seeds):
                ok, tr, ac, pe = runner(kind, s)
                surv.append(ok)
                tot.append(tr)
                acc.append(ac)
                post.append(pe)
            post_str = ""
            if kind == "gamma":
                snap = {60: [], 80: [], 100: []}
                for pe in post:
                    for idx, tgt in enumerate([10, 30, 50]):
                        if idx < len(pe):
                            snap[[60, 80, 100][idx]].append(pe[idx])
                post_str = " / ".join(f"{stats(snap[k]):.0f}" for k in (60, 80, 100))
            print(f"   {name:<12}{stats(surv):<8.3f}{stats(tot) / TURNS:<9.2f}"
                  f"{stats(acc):<10.3f}{post_str}")
        print()

    print("判读:")
    print("  1) α/β: 去噪(均值+门槛) > recency(信单次)? = 不信任单次观察在")
    print("     噪声世界里更稳 -- '重复=结构'先验跨世界通用(通过去噪学未知成立)")
    print("  2) γ: 漂移后去噪锁死(能量掉/末段准确率低) vs recency 恢复快?")
    print("     = docs/26 双刃剑: 稳定但慢(锁死), 灵活但被噪声带偏 -- 打破旧")
    print("     结构需要更强的'干预'先验(SEED-44 do-operator, 观察⊂干预⊂因果)")


if __name__ == "__main__":
    main()
