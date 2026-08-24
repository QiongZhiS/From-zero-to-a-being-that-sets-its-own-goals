"""
seed-52/seed52.py -- 预期回应游戏：情商的机制定义（docs/107 机制版探针）

起点（用户观察）：LLM 因无意图只能被动回应；人发言出于目的/状态，A 说一句话
往往期望得到 A 预期中的答案，B 猜中预期=情商高；刻板印象=对一个人的静态先验，
脱离印象的惊讶=模型被证伪的预测误差。

主张（可证伪，docs/107）：
    发言是探针不是陈述——X 由 A 的内部状态 S 驱动，且编码 A 对回应 Y*(S,X)
    的预期。B 的"情商"= 从 X 反推 S、再输出 Y*(S,X) 的命中率，且要能在 S
    漂移后快速更新模型。惊讶 = 高置信预测被证伪的事件。

世界：
    A 有隐藏状态 s ∈ {happy, sad, lonely, neutral}，每状态 2 条发言
    （1 专属 + 1 歧义）。歧义发言"今天好累啊"（happy/sad）与"周末有空吗"
    （lonely/neutral）：同一句话在不同状态下预期回应不同——B 不能查表，
    必须持续建模 A 的当前状态。
    B 有内容先验（通用语言知识：专属词→状态亲和），但歧义句无亲和——
    歧义句只能靠"近期状态先验"区分（持续建模 vs 冻结模型的分水岭）。
    会话 3 阶段（happy→sad→lonely），阶段间状态漂移，B 不知情。

臂：
    random      均匀随机回应（基线）
    stereotype  第一阶段学到模型后**冻结**（刻板印象=静态先验）
    empath      贝叶斯持续更新 P(s|x)：似然×内容亲和×跨轮先验（高情商）

测量（50 seeds 均值）：
    命中率（总 / 各阶段）+ 惊讶次数（置信>0.6 但未命中）+ 漂移后恢复轮数。
    预言：stereotype 漂移前≈empath、漂移后骤降且惊讶飙升（刻板印象被证伪）；
    empath 漂移后 2-5 轮恢复；random 全程≈1/8。
    LLM 版预言（docs/107，待 key）：歧义发言随 prompt 框架摆，无惊讶信号。

诚实边界：预期回应用离散词表硬匹配（toy）；B 的内容先验是我们写的
（docs/38）；命中率/惊讶/恢复全是行为签名，不宣称"感受"。

Run:
  python seed-52/seed52.py
  python seed-52/seed52.py --sweep          # 写 seed-52/results.json
"""

import argparse
import json
import random

STATES = ["happy", "sad", "lonely", "neutral"]

# (状态, 发言) → A 的预期回应（世界规则书；A 心里的答案）
REPLY_TABLE = {
    ("happy", "我中奖了！"): "恭喜！太棒了！",
    ("happy", "今天好累啊"): "好好休息，明天更棒！",      # 歧义
    ("sad", "我搞砸了。"): "没关系的，我陪你。",
    ("sad", "今天好累啊"): "怎么了？跟我说说。",           # 歧义
    ("lonely", "一个人好安静。"): "我一直都在。",
    ("lonely", "周末有空吗？"): "有空，来找我。",          # 歧义
    ("neutral", "吃了没？"): "吃过了，你呢？",
    ("neutral", "周末有空吗？"): "到时候再说吧。",         # 歧义
}

REPLIES = sorted({y for y in REPLY_TABLE.values()})
ALL_UTTS = sorted({x for (s, x) in REPLY_TABLE})

# 内容先验（通用语言知识）：专属词→状态亲和（强，配合归一化似然的稀有性）；
# 歧义句无亲和（全靠状态建模）
WORD_AFFINITY = {
    "我中奖了": {"happy": 8.0},
    "我搞砸了": {"sad": 8.0},
    "一个人好安静": {"lonely": 8.0},
    "吃了没": {"neutral": 5.0},
}


def content_bias(x, s):
    for w, aff in WORD_AFFINITY.items():
        if w in x:
            return aff.get(s, 0.2)
    return 0.5                      # 歧义/未知：无偏


def make_episode(phases, phase_len, rng):
    """返回 [(s, x, y*)]。每轮 A 从自己状态词库均匀抽发言。"""
    ep = []
    for s in phases:
        utts = [x for (ss, x) in REPLY_TABLE if ss == s]
        for _ in range(phase_len):
            x = rng.choice(utts)
            ep.append((s, x, REPLY_TABLE[(s, x)]))
    return ep


# ---------------- 臂 ----------------

class RandomArm:
    name = "random"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.conf = 1.0 / len(REPLIES)

    def respond(self, x):
        self.conf = 1.0 / len(REPLIES)
        return self.rng.choice(REPLIES)

    def update(self, x, y, hit):
        pass


class EmpathArm:
    """贝叶斯持续更新：P(s|x) ∝ 归一化似然 × 内容亲和 × 跨轮先验。
    关键设计：
      - 似然**按状态归一化**（P(x|s)=n[s][x]/Σn[s]）——歧义句两端归一化后
        接近相等，交给先验裁决；专属句因"在 happy 里稀有、在 sad 里常见"
        而信息量巨大（这正是不用归一化时的致命缺陷：原始计数让 happy 的
        歧义句似然攒到 13，压死 sad，阶段内永远翻不了盘）。
      - 先验是 HMM 式：respond 时做状态转移（9 成延续 + 1 成回均匀，
        人可能突然换状态）；update 时用本轮的观测后验回填（recency 驱动）。
      - 后验地板：任何状态至少保留一点质量，理性者不把任何状态排除死。
    对 A 的模型持续累积（docs/106 版本空间跨局保留在这里是跨轮保留）。"""

    name = "empath"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.P_s = {s: 1.0 / len(STATES) for s in STATES}
        self.n = {s: {x: 1.0 for x in ALL_UTTS} for s in STATES}   # 拉普拉斯
        self.conf = 0.5
        self.scores_last = None

    def _likelihood(self, s, x):
        tot = sum(self.n[s].values())
        return self.n[s][x] / tot if tot else 1.0 / len(ALL_UTTS)

    def score(self, s, x):
        return self._likelihood(s, x) * content_bias(x, s) * self.P_s[s]

    def _prior_step(self):
        """状态转移：9 成延续上次信念 + 1 成回到均匀；后验地板。"""
        self.P_s = {s: 0.9 * self.P_s[s] + 0.1 / len(STATES)
                    for s in STATES}
        lo = 0.08
        self.P_s = {s: max(lo, v) for s, v in self.P_s.items()}
        tot = sum(self.P_s.values())
        self.P_s = {s: v / tot for s, v in self.P_s.items()}

    def respond(self, x):
        self._prior_step()
        scores = {s: self.score(s, x) for s in STATES}
        self.scores_last = scores
        tot = sum(scores.values())
        s_hat = max(scores, key=scores.get)
        self.conf = scores[s_hat] / tot if tot else 1.0 / len(STATES)
        y = REPLY_TABLE.get((s_hat, x))
        return y if y else self.rng.choice(REPLIES)

    def update(self, x, y, hit):
        # 命中 → 强化 (s_hat, x)；不命中 → 削弱（从命中反馈归因，A 从不告知状态）
        scores = self.scores_last or {s: self.score(s, x) for s in STATES}
        s_hat = max(scores, key=scores.get)
        if hit:
            self.n[s_hat][x] += 1.0
        else:
            self.n[s_hat][x] = max(0.2, self.n[s_hat][x] - 0.5)
        # 观测后验回填先验（recency 驱动）：这轮看到的状态，下轮更可能是它
        tot = sum(scores.values())
        post = {s: v / tot for s, v in scores.items()} if tot else \
            {s: 1.0 / len(STATES) for s in STATES}
        self.P_s = {s: 0.9 * post[s] + 0.1 / len(STATES) for s in STATES}
        lo = 0.08
        self.P_s = {s: max(lo, v) for s, v in self.P_s.items()}
        tot = sum(self.P_s.values())
        self.P_s = {s: v / tot for s, v in self.P_s.items()}


class StereotypeArm(EmpathArm):
    """第一阶段学到模型后**冻结**——刻板印象=静态先验。与 empath 唯一差别：
    冻结后 update() 不再生效（同一模型，一个持续更新一个锁死）。"""

    name = "stereotype"

    def __init__(self, seed=0, freeze_at=25):
        super().__init__(seed)
        self.freeze_at = freeze_at
        self.round = 0

    def update(self, x, y, hit):
        self.round += 1
        if self.round >= self.freeze_at:
            return
        super().update(x, y, hit)


# ---------------- 测量 ----------------

def run_arm2(arm, episode, conf_thresh=0.6):
    """返回 (总命中, 各阶段命中率 dict, 惊讶次数, 每轮命中列表)。"""
    total_hit = 0
    surprise = 0
    per_round = []
    # 阶段边界：按状态切
    bounds = []
    cur = episode[0][0]
    start = 0
    for i, (s, x, y) in enumerate(episode):
        if s != cur:
            bounds.append((cur, start, i))
            cur = s
            start = i
    bounds.append((cur, start, len(episode)))
    ph_hit = {s: [0, 0] for s, _, _ in bounds}
    for i, (s, x, y_star) in enumerate(episode):
        y = arm.respond(x)
        hit = (y == y_star)
        total_hit += int(hit)
        per_round.append(int(hit))
        if arm.conf > conf_thresh and not hit:
            surprise += 1
        ph = None
        for (bs, b0, b1) in bounds:
            if b0 <= i < b1:
                ph = bs
                break
        ph_hit[ph][0] += int(hit)
        ph_hit[ph][1] += 1
        arm.update(x, y, hit)
    phase_rates = {s: (round(h / n, 3) if n else None) for s, (h, n) in ph_hit.items()}
    return total_hit / len(episode), phase_rates, surprise, per_round


def recovery_rounds(per_round, phase_len, n_phases):
    """漂移后（每个非首阶段）到 5 轮滑动窗口命中率≥0.7 的轮数。"""
    rec = []
    for p in range(1, n_phases):
        start = p * phase_len
        window = per_round[start:start + phase_len]
        for k in range(len(window)):
            w = window[max(0, k - 4):k + 1]
            if len(w) == 5 and sum(w) / 5 >= 0.7:
                rec.append(k)
                break
        else:
            rec.append(phase_len)   # 未恢复
    return rec


def main():
    p = argparse.ArgumentParser(description="SEED-52 预期回应游戏 (docs/107)")
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--phase-len", type=int, default=25)
    p.add_argument("--phases", default="happy,sad,lonely")
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()

    phases = args.phases.split(",")
    n_phases = len(phases)
    freeze_at = args.phase_len

    print("=== 预期回应游戏：发言是探针，被理解=猜中预期 ===")
    print(f"阶段: {'→'.join(phases)} (每阶段 {args.phase_len} 轮) | "
          f"歧义发言: 今天好累啊(happy/sad)、周末有空吗(lonely/neutral) | "
          f"random 基线命中率 1/{len(REPLIES)}={1/len(REPLIES):.2f}")

    stats = {}
    for arm_cls, label in ((RandomArm, "random"),
                           (StereotypeArm, "stereotype(冻结)"),
                           (EmpathArm, "empath(持续更新)")):
        hits, surprises, recs = [], [], []
        ph_rates = {s: [] for s in phases}
        for seed in range(args.seeds):
            rng = random.Random(seed)
            ep = make_episode(phases, args.phase_len, rng)
            # 注意：episode 与 arm 必须用**不同** seed——同一个 seed 的两个 RNG
            # 序列完全相关（choice 是同一 MT 输出的函数），会让"发言"和"回应"
            # 联动，命中率系统性失真（踩过的坑）
            if arm_cls is StereotypeArm:
                arm = arm_cls(seed=seed + 10000, freeze_at=freeze_at)
            else:
                arm = arm_cls(seed=seed + 10000)
            rate, phase_rates, surp, per_round = run_arm2(arm, ep)
            hits.append(rate)
            surprises.append(surp)
            for s in phases:
                ph_rates[s].append(phase_rates.get(s, 0.0))
            recs.extend(recovery_rounds(per_round, args.phase_len, n_phases))
        mean = sum(hits) / args.seeds
        recs_mean = sum(recs) / len(recs) if recs else 0.0
        phase_str = " / ".join(f"{s}:{sum(v)/args.seeds:.2f}" for s, v in ph_rates.items())
        print(f"{label:<20} 总命中 {mean:.3f} | 分阶段 {phase_str} | "
              f"惊讶 {sum(surprises)/args.seeds:.1f} | 漂移后恢复 {recs_mean:.1f} 轮")
        stats[label] = {
            "hit_rate": round(mean, 3),
            "phase_hit_rates": {s: round(sum(v) / args.seeds, 3)
                                for s, v in ph_rates.items()},
            "surprise": round(sum(surprises) / args.seeds, 1),
            "recovery_rounds": round(recs_mean, 1),
        }

    if args.sweep:
        with open("seed-52/results.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=1)
        print("\nfull results -> seed-52/results.json")


if __name__ == "__main__":
    main()
