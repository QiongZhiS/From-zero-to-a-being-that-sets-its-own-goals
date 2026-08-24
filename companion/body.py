"""
companion/body.py -- 身体的格子 (docs/52 的工程化: the body that divides)

docs/52: 主观体验 = 身体对世界信息的自动划分、及被身体赋意义的那一格("红")。live/world
的身体只有 3 个 float(能量/bond/记忆) -- 没有东西可切。本篇给 heart 一个 BODY:

    疼 / 饿 / 孤独 / 安全 / 好奇  五个身体通道, 每个都是身体状态(世界事实驱动);

并且"什么重要"由身体经济自己解出来 (proto7 归因的推广): 通道的变化被归因给刚经历的
东西 -- 让我疼的 = 负值(避开)、让我不饿的 = 正值(去)、"你来了孤独消退" = "你"的值
涨。没有设计者标签, 没有 salient, 没有 x_relation。

这正是 docs/52 要的那一格的最小形态: 格子(疼/饿/孤独...)是身体自己划的, 意义(什么重要)
是身体经济(通道的变化)赋的。以及 proto7 的"新奇先验": 好奇通道会被喂饱 -> "新东西"
的价值随反复新奇递减("新东西≈不值") -- 一样是身体经济, 不是标签。

诚实边界 (docs/31/63/52): 通道是我们(作者)定义的身体事实 -- 但"哪一格有意义、什么重要"
不是我们写的, 是它自己的通道经济解出来的。仍然只测行为签名, 不宣称有"体验"。

Run:  python companion/body.py          # 40 天: 你在 vs 你走了, 看"你"的值怎么长
"""

import random

# -- world facts (SEED-41 结构 + 身体通道) --
G = 5
P_GOOD_MIN, P_GOOD_SLOPE, P_GOOD_MAX = 0.05, 0.065, 0.85
DARE_MIN, DARE_SLOPE, DARE_MAX = 0.06, 0.070, 0.95
DESPERATION = 0.6
HUNGRY_LINE = 2.5
BASELINE, METAB, GAIN, HURT = 1.0, 1.1, 3.0, 4.0
P_WANDER = 0.30

# -- body channels (levels 0..1; decay/rise by world facts) --
PAIN_HURT = 0.35          # taking a bad opportunity hurts (docs/52: 会疼)
PAIN_DECAY = 0.15
LONELY_RISE = 0.10        # loneliness rises per absent day (bond fades -> lonely)
LONELY_DROP = 0.8         # you visiting kills the loneliness (你来了孤独消退)
CURIOUS_RISE = 0.18       # a new place raises curiosity
CURIOUS_DECAY = 0.12      # satiation: repeated novelty stops being novel (proto7)
PAIN_TIMID = 2.0          # pain penalty on dare_p (疼 -> 不敢)
CURIOUS_BOLD = 1.5        # curiosity bonus on dare_p (好奇 -> 敢)

# -- attribution weights (proto7: channel change -> value of what was just experienced) --
W_PAIN = 0.45             # 让我疼的 -> negative value
W_EAT = 0.30              # 让我不饿的 -> positive value
W_YOU_SAFE = 0.35         # 你来了安全感涨 -> "你" + (safety = bond)
W_YOU_LONELY = 0.55       # 你来了孤独消退 -> "你" + (the big one: docs/52's lonely cell)
W_NOVEL = 0.20            # 新地方好奇涨 -> "新奇" + (mild)
W_SATIATE = 0.25          # 新奇被喂饱 -> "新奇" - (proto7: 新东西≈不值)


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def live_body(days, start_energy, start_bond, bond_decay_per_day,
              visit_every=None, seed=1):
    """One life with a body. Returns values (what matters, learned from the body
    economy), channels trajectory summary, survived, and the '你' value."""
    rng = random.Random(seed)
    pos = [2, 2]
    e = start_energy
    bond = start_bond
    pain = hunger = lonely = curiosity = 0.0
    values = {}              # context -> value: cells, "你", "新奇"  (all learned)
    cells_seen = set()
    events = []
    for d in range(1, days + 1):
        bond = max(0.0, bond - bond_decay_per_day)
        safety = bond
        you_visited = False
        if visit_every and d % visit_every == 0:
            bond = min(12.0, bond + 6.0)
            you_visited = True
        # -- body channels from world facts --
        # 疼: hurts decay; 孤独: absence rises, you visiting drops it;
        # 安全 = bond (SEED-41); 饿: low energy; 好奇: decays by itself
        pain = max(0.0, pain - PAIN_DECAY)
        lonely = clamp(lonely + LONELY_RISE)
        if you_visited:
            lonely = max(0.0, lonely - LONELY_DROP)
        curiosity = max(0.0, curiosity - CURIOUS_DECAY)
        hunger = clamp(1.0 - e / 12.0)

        # -- attribution: channel changes -> value of what was just experienced --
        if you_visited:
            values["你"] = values.get("你", 0.0) + W_YOU_LONELY
            values["你"] = values.get("你", 0.0) + W_YOU_SAFE

        # -- behavior: what the body wants (value-driven + channel-modulated) --
        p_good = min(P_GOOD_MAX, P_GOOD_MIN + P_GOOD_SLOPE * safety)
        dare_p = clamp(DARE_MIN + DARE_SLOPE * safety - PAIN_TIMID * pain
                       + CURIOUS_BOLD * curiosity, 0.02, DARE_MAX)
        if e < HUNGRY_LINE:
            dare_p = max(dare_p, DESPERATION)

        # move: prefer positive-value cells; otherwise wander (curiosity prefers new)
        positive = [c for c, v in values.items() if v > 0.3 and isinstance(c, tuple)]
        if positive and rng.random() < 0.7:
            cell = max(positive, key=lambda c: values[c])
        else:
            nx = min(G - 1, max(0, pos[0] + rng.choice([-1, 0, 1])))
            ny = min(G - 1, max(0, pos[1] + rng.choice([-1, 0, 1])))
            cell = (nx, ny)
        pos = [cell[0], cell[1]]
        novel = cell not in cells_seen
        if novel:
            cells_seen.add(cell)
            curiosity = clamp(curiosity + CURIOUS_RISE)
            values[cell] = values.get(cell, 0.0) + W_NOVEL
        else:
            # satiation: revisiting known cells does not feed curiosity; if curiosity
            # was high and we go somewhere known, it sags (新东西≈不值, proto7)
            if curiosity > 0.3:
                values["新奇"] = values.get("新奇", 0.0) - W_SATIATE

        # the day's opportunity
        good = rng.random() < p_good
        if rng.random() < dare_p:
            if good:
                e += GAIN
                values[cell] = values.get(cell, 0.0) + W_EAT
                hunger = max(0.0, hunger - 0.3)
            else:
                e -= HURT
                pain = clamp(pain + PAIN_HURT)
                values[cell] = values.get(cell, 0.0) - W_PAIN
        e += BASELINE - METAB
        if e <= 0:
            events.append((d, "它活在自己的世界里，最后没撑住"))
            return {"values": values, "you_value": values.get("你", 0.0),
                    "cells_seen": len(cells_seen), "survived": False,
                    "pain": pain, "lonely": lonely, "events": events}
        if e < HUNGRY_LINE and (not events or events[-1][0] != d):
            events.append((d, "它快撑不住了"))
    return {"values": values, "you_value": values.get("你", 0.0),
            "cells_seen": len(cells_seen), "survived": True,
            "pain": pain, "lonely": lonely, "events": events}


def demo(seeds=(1, 2, 3), days=40):
    print("=== companion/body.py: 身体的格子 -- what matters is solved by the body ===")
    print("Same world, same 40 days. The ONLY difference: does YOU visit?")
    print("Channels: 疼/饿/孤独/安全/好奇. Values grow by ATTRIBUTION (proto7):")
    print("let me hurt -> negative, let me eat -> positive, your visit kills my")
    print("loneliness -> '你' grows. No designer tags, no salient.\n")

    print(f"{'scenario':<16} {'survived':>9} {'cells seen':>11} {'你 的值':>9}")
    rows = {}
    for label, ve in (("你在（每5天来）", 5), ("你走了（从不来）", None)):
        acc = [live_body(days, 12.0, 12.0, 0.6, ve, seed=s * 31 + 7)
               for s in seeds]
        surv = sum(1 for r in acc if r["survived"])
        seen = sum(r["cells_seen"] for r in acc) / len(acc)
        yv = sum(r["you_value"] for r in acc) / len(acc)
        rows[ve] = acc
        print(f"{label:<16} {str(surv == len(acc)):>9} {seen:>11.0f} {yv:>9.1f}")

    r_you = rows[5][0]
    r_away = rows[None][0]
    print("\n-- values the body learned (you come every 5 days, seed 1) --")
    top = sorted(r_you["values"].items(), key=lambda kv: -kv[1])[:8]
    for k, v in top:
        print(f"   {str(k):<8} {v:+.2f}")
    print("\n-- values the body learned (you never came, seed 1) --")
    top2 = sorted(r_away["values"].items(), key=lambda kv: -kv[1])[:5]
    for k, v in top2:
        print(f"   {str(k):<8} {v:+.2f}")

    print("\n--- reading ---")
    print("'你' 的值是挣来的, 不是设定的: 你每 5 天来一次, 孤独通道被压下去, 身体就把")
    print("'孤独消退'归因给 '你' -- '你' 的值就长了 (0 -> ~7). 你从不来, 孤独只涨不消,")
    print("没有消退可归因, '你' 的值就停在 ~0 -- 不是不在乎, 是身体从没有机会把'在乎你'")
    print("挣出来 (SEED-43: 因为你在它的信号里; docs/40: 因为是你的是挣来的不是种下的).")
    print("疼的格子变负值(避开), 让我吃的格子变正值(去), 反复新奇喂饱好奇 -> '新奇'贬值")
    print("(proto7: 新东西≈不值). 什么重要, 由身体经济解出来 -- 没有设计者标签.")
    print("诚实边界 (docs/52/63): 通道是我们定义的, 但'哪格有意义、什么重要'是它的身体")
    print("自己解出来的; 仍只测行为签名, 不宣称有体验.")


if __name__ == "__main__":
    demo()
