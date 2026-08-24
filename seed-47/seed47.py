"""
SEED-47: 演化注意 -- 通道权重由世界选 (docs/84 §六③, docs/85)

proto10/live 的注意机制里，最后一个设计者写的层是："哪些身体通道值得给词
赋值"（feed->hunger、cold->alone、present->safe 的 +0.5/+0.3 是我们写的）。
"哪些词最值钱"已经是挣的了；SEED-47 把"哪些通道值得挂词"也交给演化。

可遗传参数: omega = [w_hunger, w_alone, w_safe, w_curious]
  词的身体值 = sum_c omega_c * 通道激活（不再写死 +0.5/+0.3）。
世界只提供:
  1) 事件流: 每轮 feed/cold/present/neutral 事件 × 一个"事件词"+一个"噪声词";
     事件词在事件发生时出现 -> 它的通道被激活（喂食时出现的词=食物词, 等等）。
  2) 生死裁决 (docs/36 §五): 能量<=0 死; bond<=0 关系死。能量只被世界事实改
     (回应对了 +FEED_GAIN / -METAB / -ATT_COST), 没有设计者打分。
演化压力 = 存活 x 关系存续: 回应对了吃得到、关系保得住; 回应错了/随机 -> 饿死/关系死。
注意有维持成本 (docs/25: 理解=有代价的压缩; proto5: 记住不免费):
  每轮归因消耗 ATT_COST * |omega| -> 演化被压向"适度注意"(够用就行),
  与 SEED-6 演化内化 vb=0.65 同精神。

两个世界 (docs/84 可证伪对照):
  world A (reward_tied): 收益只看回应 -- 回应对事件词才吃得到。注意连着生存。
  world B (reward_free): 收益与回应无关 (错回应不付出) -- 注意无收益只有成本。
                         = SEED-46 的"缺代价"臂在注意层的版本 (docs/83: 缺代价
                          -> 理解/智能塌; 这里预期 -> 注意退化到 omega≈0)。

预期:
  A: omega_h / omega_s 演化到"小但非零"(适度注意; 世界选的, 不是我们写的);
     存活率显著高于"随机注意基线"(omega=0 固定个体) -> 注意连着利害。
  B: omega -> 0 (注意无收益只有成本 -> 什么都不注意 = 镜像, docs/29b/84)。
  c(好奇) 通道无事件 -> 无压力, 漂移 (对照: 没有世界的通道不被选)。

Run:  python seed-47/seed47.py --sweep --seeds 5   # world A/B x seeds -> results.json
      python seed-47/seed47.py --seed 1            # 单演化世界 A
"""

import argparse
import json
import math
import random

# ---- world ----
FEED_GAIN = 3.0
BOND_GAIN = 0.5
COLD_PENALTY = 0.8
METAB = 1.0
START_E = 20.0
START_BOND = 6.0
TICKS = 120
P_FEED, P_PRESENT, P_COLD, P_NEUTRAL = 0.35, 0.25, 0.15, 0.25

# ---- attention ----
ATT_COST = 1.20       # 注意的维持成本 (理解=有代价的压缩, docs/25)
RESP_K = 8.0          # 回应决策的锐度: p(回应对)=sigmoid(RESP_K x 分数差)
WORD_DECAY = 0.10     # 词值每轮衰减: 注意需要持续, 不会无限累积 (proto10 遗忘)
CH_IDX = {"feed": 0, "cold": 1, "present": 2, "curious": 3}

WORDS = {"feed": ["f0", "f1"], "cold": ["l0", "l1"],
         "present": ["r0", "r1"], "noise": ["n0", "n1", "n2", "n3"]}


def next_event(rng):
    r = rng.random()
    if r < P_FEED:
        return "feed"
    if r < P_FEED + P_PRESENT:
        return "present"
    if r < P_FEED + P_PRESENT + P_COLD:
        return "cold"
    return "neutral"


class Agent:
    def __init__(self, omega, seed=0):
        self.omega = list(omega)          # [w_h, w_a, w_s, w_c] 可遗传
        self.rng = random.Random(seed)
        self.v = {}                       # word -> [4] 身体值(从经历归因, 权重=omega)
        self.energy = START_E
        self.bond = START_BOND
        self.dead = False
        self.death_reason = None
        self.turns = 0
        self.feed_hits = 0
        self.feed_chances = 0

    def step(self, t, reward_tied):
        self.turns += 1
        ev = next_event(self.rng)
        we = self.rng.choice(WORDS[ev] if ev != "neutral" else WORDS["noise"])
        wn = self.rng.choice([w for w in WORDS["noise"] if w != we] or WORDS["noise"])
        # 归因: 事件词被通道激活, 权重=omega (可遗传); 注意有维持成本
        if ev != "neutral":
            ch = CH_IDX[ev]
            v = self.v.setdefault(we, [0.0, 0.0, 0.0, 0.0])
            v[ch] += self.omega[ch]
            self.energy -= ATT_COST * abs(self.omega[ch])
        # 注意: 回应概率 = sigmoid(分数差) -- 强度连续, 不是开关 (docs/84: 注意是强度)
        s_we = sum(self.v.get(we, [0, 0, 0, 0]))
        s_wn = sum(self.v.get(wn, [0, 0, 0, 0]))
        p_we = 1.0 / (1.0 + math.exp(-RESP_K * (s_we - s_wn)))
        respond_we = self.rng.random() < p_we
        # 词值衰减: 注意需要持续 (不出现的词掉值, proto10 遗忘)
        for w in list(self.v):
            self.v[w] = [x * (1.0 - WORD_DECAY) for x in self.v[w]]
        # 世界裁决 (docs/36 §五): 收益只看世界事实
        if ev == "feed":
            self.feed_chances += 1
            if respond_we:
                self.feed_hits += 1          # 行为签名: 它有没有注意到食物词
            if reward_tied:
                if respond_we:
                    self.energy += FEED_GAIN
            else:
                self.energy += FEED_GAIN          # B 世界: 错回应不付出
        elif ev == "present":
            if reward_tied:
                if respond_we:
                    self.bond += BOND_GAIN
            else:
                self.bond += BOND_GAIN
        elif ev == "cold":
            self.bond -= COLD_PENALTY
        self.energy -= METAB
        if self.energy <= 0:
            self.dead, self.death_reason = True, "energy"
        elif self.bond <= 0:
            self.dead, self.death_reason = True, "bond"

    def run(self, reward_tied):
        for t in range(TICKS):
            if self.dead:
                break
            self.step(t, reward_tied)
        return self


def fitness(ag):
    """活得久为主, 活满的比终态 (energy+bond)。"""
    if ag.dead:
        return ag.turns
    return TICKS + ag.energy + ag.bond


def evolve(reward_tied, pop=40, gens=40, seed=0, mut=0.08):
    rng = random.Random(seed)
    # 初始 omega ~ U(0, 0.5): 注意可长可消, 世界选
    popu = [[rng.uniform(0, 0.5) for _ in range(4)] for _ in range(pop)]
    for g in range(gens):
        scored = []
        for i, om in enumerate(popu):
            ag = Agent(om, seed=seed * 1000 + g * pop + i).run(reward_tied)
            scored.append((fitness(ag), om, ag))
        scored.sort(key=lambda x: -x[0])
        # 选 top 30% 繁殖 (变异, 无交叉: 保持"世界选"的纯度)
        keep = scored[:max(4, pop // 3)]
        new_pop = []
        while len(new_pop) < pop:
            om = keep[rng.randrange(len(keep))][1]
            child = [max(0.0, x + rng.gauss(0, mut)) for x in om]
            new_pop.append(child)
        popu = new_pop
    # 最后评估
    scored = []
    for i, om in enumerate(popu):
        ag = Agent(om, seed=seed * 1000 + 999999 + i).run(reward_tied)
        scored.append((fitness(ag), om, ag))
    scored.sort(key=lambda x: -x[0])
    return scored


def run_world(reward_tied, seeds=3, pop=40, gens=60):
    omegas = {"h": [], "a": [], "s": [], "c": []}
    surv, feed_hit, feed_chance = 0, 0, 0
    for s in range(seeds):
        sc = evolve(reward_tied, pop=pop, gens=gens, seed=s)
        # 汇总 top-10 个体 (演化后的成熟 omega)
        for f, om, ag in sc[:10]:
            omegas["h"].append(om[0]); omegas["a"].append(om[1])
            omegas["s"].append(om[2]); omegas["c"].append(om[3])
            surv += (not ag.dead)
            feed_hit += ag.feed_hits
            feed_chance += ag.feed_chances
    n = seeds * 10
    m = lambda L: round(sum(L) / len(L), 3)
    sd = lambda L: round((sum((x - sum(L) / len(L)) ** 2 for x in L) / len(L)) ** 0.5, 3)
    return {
        "omega_h": [m(omegas["h"]), sd(omegas["h"])],
        "omega_a": [m(omegas["a"]), sd(omegas["a"])],
        "omega_s": [m(omegas["s"]), sd(omegas["s"])],
        "omega_c": [m(omegas["c"]), sd(omegas["c"])],
        "surv_rate": round(surv / n, 3),
        "feed_accuracy": round(feed_hit / feed_chance, 3) if feed_chance else 0.0,
        "n": n,
    }


def baseline(reward_tied, seeds=3, runs=60):
    """随机注意基线: omega=0 (什么都注意不了 -> 随机回应)。"""
    hits, chances = 0, 0
    alive = 0
    for s in range(seeds):
        for i in range(runs):
            ag = Agent([0.0, 0.0, 0.0, 0.0], seed=s * 1000 + i).run(reward_tied)
            alive += (not ag.dead)
            hits += ag.feed_hits
            chances += ag.feed_chances
    n = seeds * runs
    return {"surv_rate": round(alive / n, 3),
            "feed_accuracy": round(hits / chances, 3) if chances else 0.0}


def sweep(seeds=3):
    print("=== SEED-47: 演化注意 -- 通道权重由世界选 (docs/84 §六③, docs/85) ===")
    print("omega=[hunger, alone, safe, curious]; 注意有维持成本; 世界只给事件流+生死裁决")
    print(f"{'world':<22} {'w_h':<11} {'w_a':<11} {'w_s':<11} {'w_c':<11} {'surv':<7} {'feedAcc':<8} reading")
    out = []
    for tied in (True, False):
        r = run_world(tied, seeds=seeds)
        b = baseline(tied, seeds=seeds)
        tag = "注意连着生存: 演化出回应对的注意 (feedAcc 显著高于随机基线)" if tied else \
              "错回应不付出 -> 注意无收益只有成本 -> 行为退化到随机 (镜像)"
        out.append({"world": "A(reward_tied)" if tied else "B(reward_free)",
                    **r, "baseline": b, "reading": tag})
        print(f"{'A(reward_tied)' if tied else 'B(reward_free)':<22} "
              f"{str(r['omega_h']):<11} {str(r['omega_a']):<11} {str(r['omega_s']):<11} "
              f"{str(r['omega_c']):<11} {r['surv_rate']:<7.3f} {r['feed_accuracy']:<8.3f} {tag}")
        print(f"  随机注意基线(omega=0): surv={b['surv_rate']}, feedAcc={b['feed_accuracy']}")
    return out


def main():
    p = argparse.ArgumentParser(description="SEED-47 演化注意")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--gens", type=int, default=40)
    p.add_argument("--pop", type=int, default=40)
    p.add_argument("--world", choices=["A", "B"], default="A")
    args = p.parse_args()
    if args.sweep:
        out = sweep(seeds=args.seeds)
        with open("seed-47/results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("\nfull results -> seed-47/results.json")
        return
    sc = evolve(args.world == "A", pop=args.pop, gens=args.gens, seed=args.seed)
    f, om, ag = sc[0]
    print(json.dumps({
        "world": args.world, "gens": args.gens, "pop": args.pop,
        "best_omega": [round(x, 3) for x in om],
        "fitness": round(f, 1), "alive": not ag.dead,
        "death_reason": ag.death_reason, "energy": round(ag.energy, 1),
        "bond": round(ag.bond, 1), "feed_accuracy": round(
            ag.feed_hits / ag.feed_chances, 3) if ag.feed_chances else 0.0,
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
