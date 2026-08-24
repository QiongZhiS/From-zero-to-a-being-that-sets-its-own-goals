"""
seed-53/seed53.py -- 刻板印象的理性：更新成本曲线（docs/109）

起点（docs/108 下一步①）：SEED-52 量化了"刻板印象的代价"（冻结模型漂移后
命中骤降），但那是**更新免费**的世界。现实里更新模型有认知成本（docs/25
有代价的压缩；docs/6/SEED-6 验证有成本演化选适度）。问题反过来：
**刻板印象在什么条件下是理性的？** 当更新成本 > 预测错误代价时，不更新
（冻结）反而是最优——刻板印象不是认知缺陷，是资源受限下的理性近似。

主张（可证伪）：
    给预期回应游戏加"认知账本"（更新耗能量 E）与"关系账本"（命中+1、
    miss-2，关系归零=被疏远；能量归零=认知耗竭）。世界漂移**不循环**
    （happy→sad→lonely→neutral 后饱和在 neutral）——冻结模型的"锁定
    状态"永不回来，关系单调流失，漂移速度才真正影响冻结者命运。
    扫描 (更新成本 U × 漂移速度 phase_len) 网格，四策略：
      freeze       热身 15 轮学习后冻结（刻板印象）
      always       每轮更新（无脑高情商）
      surprise     惊讶驱动更新（docs/107：惊讶=模型被证伪的信号=该更新了）
      random       每轮 30% 概率更新（控制变量）
    预言：
      1) 慢世界 + 高成本：freeze 存活且认知几乎零消耗——刻板印象是理性的
         （省下的认知 > 损失的命中）；always/random 认知耗竭
      2) 快世界：freeze 关系崩盘被疏远——必须更新
      3) surprise 全网格接近最优——"惊讶驱动更新"自适应成立
      4) surprise 的更新次数随漂移速度单调上升

世界（SEED-52 四状态）：happy/sad/lonely/neutral 各 1 专属 + 1 歧义发言
（"今天好累啊"happy/sad、"周末有空吗"lonely/neutral——歧义句逼出持续建模）。

诚实边界：能量/关系账本与内容先验是我们写的（docs/38）；"认知成本"是抽象
参数；行为签名（存活/关系/命中/更新）不宣称感受；toy 世界。

Run:
  python seed-53/seed53.py                  # 默认网格扫描
  python seed-53/seed53.py --sweep          # 写 seed-53/results.json
"""

import argparse
import json
import random

STATES = ["happy", "sad"]

REPLY_TABLE = {
    ("happy", "我中奖了！"): "恭喜！太棒了！",
    ("happy", "今天好累啊"): "好好休息，明天更棒！",      # 歧义
    ("sad", "我搞砸了。"): "没关系的，我陪你。",
    ("sad", "今天好累啊"): "怎么了？跟我说说。",           # 歧义
}

REPLIES = sorted({y for y in REPLY_TABLE.values()})
ALL_UTTS = sorted({x for (s, x) in REPLY_TABLE})

# 内容先验：专属词→状态亲和（强）；歧义句无亲和
WORD_AFFINITY = {
    "我中奖了": {"happy": 8.0},
    "我搞砸了": {"sad": 8.0},
}


def content_bias(x, s):
    for w, aff in WORD_AFFINITY.items():
        if w in x:
            return aff.get(s, 0.2)
    return 0.5


def make_episode(drift_p, total_rounds, rng):
    """漂移速度 = 每轮状态翻转概率 p（随机游走，从 happy 开始）。
    冻结模型（锁 happy）一旦状态翻到 sad 就进入错误窗口；翻转不保证回来
    ——冻结者的累积错误随 p 单调上升（这是固定 pivot 做不到的：循环世界
    里锁定状态会回来、吸收世界里只有第一个 pivot 起作用）。"""
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


# ---------------- 贝叶斯核心（SEED-52 empath 的 4 状态版） ----------------

class BayesAgent:
    name = "bayes"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.P_s = {s: 1.0 / len(STATES) for s in STATES}
        self.n = {s: {x: 1.0 for x in ALL_UTTS} for s in STATES}
        self.conf = 0.5
        self.scores_last = None
        self.round = 0
        self.updates = 0

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
        """更新模型（有成本，由策略决定何时调用）。"""
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
        raise NotImplementedError


class FreezeAgent(BayesAgent):
    """刻板印象：热身期学习，之后冻结（永不更新）。"""

    name = "freeze"

    def __init__(self, seed=0, warmup=15):
        super().__init__(seed)
        self.warmup = warmup

    def decide_update(self, hit, conf):
        return self.round < self.warmup


class AlwaysAgent(BayesAgent):
    """每轮更新（无脑高情商）。"""

    name = "always"

    def decide_update(self, hit, conf):
        return True


class SurpriseAgent(BayesAgent):
    """惊讶驱动更新：高置信 miss=惊讶（模型被证伪，docs/107），累积误差超阈值
    就更新——预测误差驱动的更新闭环（docs/81）。"""

    name = "surprise"

    def __init__(self, seed=0, thresh=1.5):
        super().__init__(seed)
        self.thresh = thresh
        self.err = 0.0

    def decide_update(self, hit, conf):
        if hit:
            self.err *= 0.8          # 命中衰减误差（世界还是老样子）
        else:
            self.err += 1.5 if conf > 0.6 else 0.5   # 惊讶 / 低置信
        if self.err >= self.thresh:
            self.err = 0.0
            return True
        return False


class RandomAgent(BayesAgent):
    """每轮以 30% 概率更新（控制变量）。"""

    name = "random"

    def decide_update(self, hit, conf):
        return self.rng.random() < 0.3


# ---------------- 认知/关系账本 ----------------

START_E = 100.0
METAB = 0.1          # 每轮维持成本
START_R = 50.0
HIT_GAIN = 1.0
MISS_PEN = 4.0       # 被误解比被理解贵得多（一次伤人的误解远超一次顺利交流）


def run_episode(agent, drift_p, U, seed, total_rounds=240):
    """一局：A 以每轮概率 drift_p 漂移，agent 回应。返回统计。"""
    rng = random.Random(seed)
    ep = make_episode(drift_p, total_rounds, rng)
    E = START_E
    R = START_R
    hits = 0
    outcome = "alive"
    survived = total_rounds
    for i, (s, x, y_star) in enumerate(ep):
        agent.round = i
        E -= METAB
        if E <= 0:
            outcome = "exhausted"
            survived = i
            break
        y = agent.respond(x)
        hit = (y == y_star)
        hits += int(hit)
        R += HIT_GAIN if hit else -MISS_PEN
        if R <= 0:
            outcome = "estranged"
            survived = i
            break
        if agent.decide_update(hit, agent.conf):
            E -= U
            if E <= 0:
                outcome = "exhausted"
                survived = i
                break
            agent.update(x, hit)
    return {
        "outcome": outcome,
        "survived": survived,
        "R": R,
        "E": E,
        "hit_rate": hits / survived if survived else 0.0,
        "updates": agent.updates,
    }


def scan(seeds=20):
    print("=== 刻板印象的理性：更新成本 × 世界漂移速度（每轮翻转概率 p）===")
    print("总轮数 240 | 2 状态(happy/sad)随机游走，p=每轮翻转概率（翻转不保证回来）")
    print("账本: 认知E=100(代谢-0.1/轮, 更新-U) | 关系R=50(命中+1, miss-4, R≤0=被疏远, E≤0=耗竭)")
    print(f"{'U':>4} {'p':>6} | " + " | ".join(
        f"{a.name:>10}" for a in (FreezeAgent(0), AlwaysAgent(0),
                                  SurpriseAgent(0), RandomAgent(0))))
    print("-" * 130)
    out = {}
    for U in (0.5, 2.0, 4.0):
        for p in (0.0, 0.002, 0.01, 0.05):
            row = {}
            for cls in (FreezeAgent, AlwaysAgent, SurpriseAgent, RandomAgent):
                alive = Rsum = Esum = hit = upd = 0
                n = 0
                for seed in range(seeds):
                    a = cls(seed=seed + 10000)
                    r = run_episode(a, p, U, seed, total_rounds=240)
                    alive += int(r["outcome"] == "alive")
                    Rsum += r["R"]
                    Esum += r["E"]
                    hit += r["hit_rate"]
                    upd += r["updates"]
                    n += 1
                row[cls.name] = {
                    "alive_rate": round(alive / n, 2),
                    "avg_R": round(Rsum / n, 1),
                    "avg_E": round(Esum / n, 1),
                    "avg_hit": round(hit / n, 3),
                    "avg_updates": round(upd / n, 1),
                }
            cells = []
            for k in ("freeze", "always", "surprise", "random"):
                r = row[k]
                cells.append(f"{r['alive_rate']:.2f} R{r['avg_R']:>5.0f} "
                             f"E{r['avg_E']:>4.0f} {r['avg_updates']:>3.0f}次")
            print(f"{U:>4.1f} {p:>6.3f} | " + " | ".join(f"{c:>26}" for c in cells))
            out[f"U{U}_p{p}"] = row
    return out


def main():
    p = argparse.ArgumentParser(description="SEED-53 刻板印象的理性 (docs/109)")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()
    out = scan(args.seeds)
    if args.sweep:
        with open("seed-53/results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("\nfull results -> seed-53/results.json")


if __name__ == "__main__":
    main()
