"""
seed-54/seed54.py -- 演化惊讶阈值：刻板印象 vs 自适应是同一个旋钮的两端（docs/110）

起点（docs/109 下一步①）：SEED-53 手工对比了 freeze（θ=∞）/surprise（θ=1.5）/
always（θ=0），发现"稳定世界冻结=省认知的理性近似、快世界必须更新"。但那些
阈值是我们**手工写的**。本 seed 把惊讶阈值 θ（触发模型更新的敏感度）变成可遗传
参数，让**世界漂移速度 p 自己选出最优 θ**——方法同 SEED-47/docs/85（通道权重
由世界选）。

主张（可证伪）：
    惊讶阈值 θ 是"何时更新模型"的旋钮：θ 大=懒惰（高置信 miss 多次才更新，
    接近冻结/刻板印象），θ 小=敏感（一有惊讶就更新，接近持续建模）。
    演化（种群+选择+变异，适应度=关系 R）在不同漂移速度 p 的世界里跑，
    预言：
      1) p=0（静止）：演化选出大 θ（接近永不更新）——**刻板印象是演化
         在稳定世界里选出来的最优解**，不是缺陷
      2) p 增大：θ* 单调下降（世界越动越敏感）
      3) p=0.05（快）：θ* 很小（一有风吹草动就更新）
      4) 演化选出的 θ* 的适应度 ≥ SEED-53 手工策略（θ=1.5）——世界自选
         优于任何固定预设
    结果应是一条单调下降的 θ*(p) 曲线——刻板印象与自适应不是两种策略，
    是同一个旋钮的两端，由世界压力调节（docs/85 A/B 世界、docs/101 最优
    可塑性随世界硬度移动的同构复现）。

世界/账本：同 SEED-53（2 状态 happy/sad 随机游走、p=每轮翻转概率；认知 E
代谢-0.1/更新-U；关系 R 命中+1/miss-4）。

诚实边界：适应度（R）与账本是我们写的（docs/38）；演化在固定 p 的世界里
选 θ——"世界压力调节旋钮"= 演化视角，不是 agent 内省；toy 世界。

Run:
  python seed-54/seed54.py                  # 全部 p 演化
  python seed-54/seed54.py --sweep          # 写 seed-54/results.json
"""

import argparse
import json
import math
import random

# ---------------- 世界与账本（同 SEED-53） ----------------

STATES = ["happy", "sad"]

REPLY_TABLE = {
    ("happy", "我中奖了！"): "恭喜！太棒了！",
    ("happy", "今天好累啊"): "好好休息，明天更棒！",      # 歧义
    ("sad", "我搞砸了。"): "没关系的，我陪你。",
    ("sad", "今天好累啊"): "怎么了？跟我说说。",           # 歧义
}

REPLIES = sorted({y for y in REPLY_TABLE.values()})
ALL_UTTS = sorted({x for (s, x) in REPLY_TABLE})

WORD_AFFINITY = {
    "我中奖了": {"happy": 8.0},
    "我搞砸了": {"sad": 8.0},
}

START_E = 150.0
METAB = 0.1
START_R = 50.0
HIT_GAIN = 1.0
MISS_PEN = 2.0       # miss 便宜
U_DEFAULT = 6.0      # 更新贵——懒惰在慢世界才有优势（成本比翻转）


def content_bias(x, s):
    for w, aff in WORD_AFFINITY.items():
        if w in x:
            return aff.get(s, 0.2)
    return 0.5


def make_episode(drift_p, total_rounds, rng):
    ep = []
    s_idx = 0
    for _ in range(total_rounds):
        if rng.random() < drift_p:
            s_idx = 1 - s_idx
        s = STATES[s_idx]
        utts = [x for (ss, x) in REPLY_TABLE if ss == s]
        x = rng.choice(utts)
        ep.append((s, x, REPLY_TABLE[(s, x)]))
    return ep


class SurpriseAgent:
    """惊讶驱动更新：高置信 miss=模型被证伪（docs/107），误差积分超阈值 θ 就
    更新。θ 是本 seed 要演化的参数（旋钮：大=懒惰/接近冻结，小=敏感）。
    前 15 轮热身（总是更新）——把"初始学习"与"持续敏感度"两个旋钮隔离：
    演化只作用于 θ（初始学习对所有 θ 相同）。"""

    def __init__(self, seed=0, thresh=1.5, warmup=8):
        self.rng = random.Random(seed)
        self.thresh = thresh
        self.warmup = warmup
        self.P_s = {s: 1.0 / len(STATES) for s in STATES}
        self.n = {s: {x: 1.0 for x in ALL_UTTS} for s in STATES}
        self.conf = 0.5
        self.scores_last = None
        self.err = 0.0
        self.updates = 0
        self.round = 0

    def _likelihood(self, s, x):
        tot = sum(self.n[s].values())
        return self.n[s][x] / tot if tot else 1.0 / len(ALL_UTTS)

    def _prior_step(self):
        self.P_s = {s: 0.9 * self.P_s[s] + 0.1 / len(STATES)
                    for s in STATES}
        lo = 0.08
        self.P_s = {s: max(lo, v) for s, v in self.P_s.items()}
        tot = sum(self.P_s.values())
        self.P_s = {s: v / tot for s, v in self.P_s.items()}

    def respond(self, x):
        self._prior_step()
        scores = {s: self._likelihood(s, x) * content_bias(x, s) * self.P_s[s]
                  for s in STATES}
        self.scores_last = scores
        tot = sum(scores.values())
        s_hat = max(scores, key=scores.get)
        self.conf = scores[s_hat] / tot if tot else 1.0 / len(STATES)
        y = REPLY_TABLE.get((s_hat, x))
        return y if y else self.rng.choice(REPLIES)

    def update(self, x, hit):
        scores = self.scores_last or {s: self._likelihood(s, x) *
                                      content_bias(x, s) * self.P_s[s]
                                      for s in STATES}
        s_hat = max(scores, key=scores.get)
        if hit:
            self.n[s_hat][x] += 1.0
        else:
            self.n[s_hat][x] = max(0.2, self.n[s_hat][x] - 0.5)
        tot = sum(scores.values())
        post = {s: v / tot for s, v in scores.items()} if tot else \
            {s: 1.0 / len(STATES) for s in STATES}
        self.P_s = {s: 0.9 * post[s] + 0.1 / len(STATES) for s in STATES}
        lo = 0.08
        self.P_s = {s: max(lo, v) for s, v in self.P_s.items()}
        tot = sum(self.P_s.values())
        self.P_s = {s: v / tot for s, v in self.P_s.items()}
        self.updates += 1

    def decide_update(self, hit, conf):
        if self.round < self.warmup:
            return True
        if hit:
            self.err *= 0.8
        else:
            self.err += 1.5 if conf > 0.6 else 0.5
        if self.err >= self.thresh:
            self.err = 0.0
            return True
        return False


def run_episode(agent, drift_p, U, seed, total_rounds=240):
    rng = random.Random(seed)
    ep = make_episode(drift_p, total_rounds, rng)
    E = START_E
    R = START_R
    outcome = "alive"
    for i, (s, x, y_star) in enumerate(ep):
        agent.round = i
        E -= METAB
        if E <= 0:
            outcome = "exhausted"
            break
        y = agent.respond(x)
        hit = (y == y_star)
        R += HIT_GAIN if hit else -MISS_PEN
        if R <= 0:
            outcome = "estranged"
            break
        if agent.decide_update(hit, agent.conf):
            E -= U
            if E <= 0:
                outcome = "exhausted"
                break
            agent.update(x, hit)
    return {"outcome": outcome, "R": R, "updates": agent.updates}


# ---------------- 演化层 ----------------

def fitness(thresh, drift_p, U, world_seeds):
    """θ 的适应度：在同一组世界实例上跑（同代个体共享——公平对比，
    消灭世界实例方差；本 seed 踩过 K 小方差压过 θ 差异的坑）。"""
    Rs = 0.0
    upds = 0.0
    for ws in world_seeds:
        a = SurpriseAgent(seed=ws + 99991, thresh=thresh)
        r = run_episode(a, drift_p, U, ws, 240)
        Rs += r["R"]
        upds += r["updates"]
    return Rs / len(world_seeds), upds / len(world_seeds)


def evolve(drift_p, U=6.0, pop=40, gens=25, K=6, seed=1,
           log_lo=math.log(0.2), log_hi=math.log(5.0), elite=2):
    """演化：种群=惊讶阈值 θ。适应度=同组世界实例上的平均关系 R。
    锦标赛选择+精英+乘法变异。"""
    rng = random.Random(seed)
    ths = [math.exp(rng.uniform(log_lo, log_hi)) for _ in range(pop)]
    trace = []
    for g in range(gens):
        # 同代共享同一组世界实例（公平对比）
        world_seeds = [rng.randrange(10 ** 9) for _ in range(K)]
        fits = [fitness(th, drift_p, U, world_seeds) for th in ths]
        # 精英保留
        order = sorted(range(pop), key=lambda i: -fits[i][0])
        elite_ths = [ths[i] for i in order[:elite]]
        # 锦标赛选择 + 变异
        new_ths = []
        while len(new_ths) < pop - elite:
            cands = [rng.randrange(pop) for _ in range(3)]
            pi = max(cands, key=lambda i: fits[i][0])
            child = ths[pi]
            if rng.random() < 0.9:
                child = child * math.exp(rng.gauss(0, 0.35))
            else:
                child = math.exp(rng.uniform(log_lo, log_hi))   # 重置防卡死
            child = min(20.0, max(0.05, child))                 # 阈值范围
            new_ths.append(child)
        ths = elite_ths + new_ths
        best_i = max(range(pop), key=lambda i: fits[i][0])
        trace.append({"gen": g, "mean_theta": sum(ths) / pop,
                      "best_theta": ths[best_i],
                      "best_R": round(fits[best_i][0], 1)})
    best_i = max(range(pop), key=lambda i: fits[i][0])
    theta_star = ths[best_i]
    best_R, best_upd = fits[best_i]
    return theta_star, best_R, best_upd, trace


def main():
    p = argparse.ArgumentParser(description="SEED-54 演化惊讶阈值 (docs/110)")
    p.add_argument("--pop", type=int, default=40)
    p.add_argument("--gens", type=int, default=25)
    p.add_argument("--K", type=int, default=6, help="每代共享的世界实例数")
    p.add_argument("--U", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--rep", type=int, default=3,
                   help="每个 p 独立演化次数（报告均值）")
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()

    print("=== 演化惊讶阈值：世界漂移速度选 θ（何时更新模型的旋钮）===")
    print(f"种群 {args.pop} × {args.gens} 代 × {args.rep} 次演化 | "
          f"每代 {args.K} 个共享世界实例 | U={args.U}（更新贵）miss-{MISS_PEN:.0f}（便宜）")
    print(f"θ 大=懒惰(接近冻结)  θ 小=敏感(接近持续建模) | 前 8 轮热身 | "
          f"θ*(均值 ± 范围)")
    print("-" * 118)
    out = {}
    for drift_p in (0.0, 0.002, 0.01, 0.05):
        ths, Rs, upds = [], [], []
        for r in range(args.rep):
            th_star, best_R, best_upd, trace = evolve(
                drift_p, U=args.U, pop=args.pop, gens=args.gens, K=args.K,
                seed=args.seed + r * 101)
            ths.append(th_star)
            Rs.append(best_R)
            upds.append(best_upd)
        lo, hi = min(ths), max(ths)
        mean_th = sum(ths) / len(ths)
        mean_R = sum(Rs) / len(Rs)
        mean_upd = sum(upds) / len(upds)
        if drift_p == 0.0:
            tag = "静止世界：θ 无选择压力（从不需要更新——'刻板印象'在这一端只是无事可做）"
        elif mean_th >= 0.8:
            tag = "快世界：θ 停在中等——认知预算给敏感度封顶（过度敏感=耗竭，演化自发复现'always 全灭'）"
        else:
            tag = "中慢世界：选了高敏感（世界越动，更新触发越灵敏）"
        print(f"{drift_p:>7.3f} | θ* {mean_th:.2f} [{lo:.2f}-{hi:.2f}] "
              f"R {mean_R:>6.1f} 更新 {mean_upd:>5.1f}/局 | {tag}")
        out[f"p{drift_p}"] = {"theta_star_mean": round(mean_th, 2),
                              "theta_star_range": [round(lo, 2), round(hi, 2)],
                              "best_R": round(mean_R, 1),
                              "updates_per_episode": round(mean_upd, 1)}
    if args.sweep:
        with open("seed-54/results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("\nfull results -> seed-54/results.json")


if __name__ == "__main__":
    main()
