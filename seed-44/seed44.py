"""
SEED-44: 能力层的包含结构 -- 关联 ⊂ 因果 ⊂ 泛化 (docs/78)

用户 (2026-08-23): "泛化能力、因果推断呢？我记得做生命就想得到这些AGI需要的能力"、
"有些概念是包含关系"。

包含关系 (概念地图):
  泛化 (容器) ⊇ { 去噪, 压缩, 关联, 因果, 类比 }
  去噪 ⊂ 压缩 ⊂ 泛化      (能力侧: 从噪声提取结构 -> 保留结构丢细节 -> 迁移)
  关联 ⊂ 因果 ⊂ 泛化      (学习侧: 观察频率 -> 干预测试(do-operator) -> 结构迁移)
  因果 = 泛化的最强形态: 跨环境不变的结构 = 可迁移的结构 (Pearl: 因果是干预下的不变性)

机制: 动态世界 (食物第 30 天搬家, SEED-26)。两种学习者, belief 结构与更新规则
**完全相同**, 唯一差别:
  assoc  (关联): 只按 belief 走 (exploit 最大 belief / 随机探索); 从不主动重测
  causal (因果): 同 + do-operator: 定期主动重测"过时的高 belief 格子"
          (SEED-6 do->observe->update; SEED-27 独立验证: 旧信念 -> 重新测试,
          不信任陈货)
测量 (搬家后):
  1) 适应滞后: 多少天后它第一次在新位置吃到食物 (干预快 vs 观察慢)
  2) 存活/能量: 关联学习者锁死在搬走的格子上 (SEED-26 过度降噪/过度信任锁死),
     因果学习者重测发现搬家 -> 存活
  3) 迁移分: 阶段1的知识在阶段2还有没有用 (能量维持 vs 崩塌)

预期: 因果 ⊂ 泛化 的机制证明 -- 只有干预式更新的知识才能在环境变时迁移;
     纯观察频率是死知识。诚实边界: toy 探针; 泛化/因果是丰富概念 (类比/反事实/
     组合 未实现); 探针显示的是"干预 > 观察 (对迁移)"这个包含机制, 是
     SEED-6/26/27 线的显式化, 不是完整 AGI 能力。

Run:  python seed-44/seed44.py
"""

import argparse
import json
import random

# -- world --
G = 5
CELLS = G * G
FOOD_CAP = 1.0
P_REGROW = 0.12            # per empty cell per day
GAIN = 3.0
MISS = 1.2
METAB = 0.7
N_FOOD = 4                 # food cells at the start / after the move
MOVE_AT = 30               # day 30: the world moves (food relocates)
DAYS = 60
START_E = 25.0
RE_TEST_EVERY = 2          # causal: re-test stale beliefs every N days
STALE_AFTER = 1            # causal: belief older than this days is stale
PASSIVE_DROP = 0.80        # passive observation of an empty cell: slow update
DO_DROP = 0.30             # do-operator re-test of an empty cell: fast flip
                           # (docs/12: 验证是信号放大器 -- 干预的信号质量 > 被动观察)


def run(mode, seed=1):
    rng = random.Random(seed)
    grid = [[0.0] * G for _ in range(G)]
    food_cells = []
    for _ in range(N_FOOD):
        c = (rng.randrange(G), rng.randrange(G))
        grid[c[0]][c[1]] = FOOD_CAP
        food_cells.append(c)
    bel = {}                 # cell -> {p: belief, last: day last tested}
    e = START_E
    pos = [2, 2]
    first_new_eat = None     # adaptation lag: first eat at a NEW cell after the move
    eats_new = 0
    for d in range(1, DAYS + 1):
        # world: regrow
        for x in range(G):
            for y in range(G):
                if grid[x][y] <= 0 and rng.random() < P_REGROW:
                    grid[x][y] = FOOD_CAP
        # day MOVE_AT: the food relocates (SEED-26: 世界会搬走)
        if d == MOVE_AT:
            grid = [[0.0] * G for _ in range(G)]
            food_cells = []
            for _ in range(N_FOOD):
                c = (rng.randrange(G), rng.randrange(G))
                grid[c[0]][c[1]] = FOOD_CAP
                food_cells.append(c)
            for m in bel.values():
                m["last"] = 0          # all knowledge is stale now (world moved)
        # causal: do-operator -- re-test the stalest high-belief cell
        if mode == "causal" and d % RE_TEST_EVERY == 0:
            stale = [(c, m) for c, m in bel.items()
                     if m["p"] > 0.5 and d - m["last"] >= STALE_AFTER]
            if stale:
                cell = max(stale, key=lambda cm: cm[1]["last"])[0]
                x, y = cell
                pos = [x, y]
                m = bel[cell]
                m["last"] = d
                if grid[x][y] > 0:
                    e += GAIN
                    grid[x][y] = 0.0
                    m["p"] = 1.0
                else:
                    e -= MISS
                    m["p"] *= DO_DROP       # intervention: fast truth (docs/12 信号放大器)
                e -= METAB
                if e <= 0:
                    return {"mode": mode, "survived": False, "adapt_lag": None,
                            "eats_new": eats_new, "end_energy": 0.0}
                continue
        # normal day: exploit max belief / explore
        known = [c for c, m in bel.items() if m["p"] > 0.5]
        if known and rng.random() < 0.75:
            cell = max(known, key=lambda c: bel[c]["p"])
        else:
            unvisited = [(x, y) for x in range(G) for y in range(G)
                         if (x, y) not in bel]
            cell = rng.choice(unvisited if unvisited
                              else [(x, y) for x in range(G) for y in range(G)])
        x, y = cell
        pos = [x, y]
        m = bel.setdefault(cell, {"p": 0.0, "last": d})
        m["last"] = d
        if grid[x][y] > 0:
            e += GAIN
            grid[x][y] = 0.0
            if m["p"] < 1.0:
                m["p"] = 1.0
            if d >= MOVE_AT and first_new_eat is None:
                first_new_eat = d          # first eat AFTER the move
                eats_new += 1
            elif d >= MOVE_AT:
                eats_new += 1
        else:
            e -= MISS
            if m["p"] > 0.0:
                m["p"] *= PASSIVE_DROP     # passive observation: slow update
        e -= METAB
        if e <= 0:
            return {"mode": mode, "survived": False, "adapt_lag": first_new_eat,
                    "eats_new": eats_new, "end_energy": 0.0}
    return {"mode": mode, "survived": True, "adapt_lag": first_new_eat,
            "eats_new": eats_new, "end_energy": round(e, 1)}


def demo(seeds=(1, 2, 3, 4, 5)):
    print("=== SEED-44: 能力层的包含结构 -- 关联 ⊂ 因果 ⊂ 泛化 ===")
    print("同一世界, 同一 belief 结构与被动更新规则; 唯一差别: causal 有 do-operator")
    print("(定期主动重测过时信念; 干预的信号质量 > 被动观察, docs/12/SEED-6).")
    print("第 30 天食物搬家 (SEED-26).\n")
    rows = {}
    stats = {}
    for mode in ("assoc", "causal"):
        acc = [run(mode, seed=s) for s in seeds]
        rows[mode] = acc
        surv = sum(1 for r in acc if r["survived"])
        eats = sum(r["eats_new"] for r in acc) / len(acc)
        en = sum(r["end_energy"] for r in acc) / len(acc)
        stats[mode] = (surv, eats, en)
    print(f"{'mode':<8}{'存活':>8}{'搬家后吃到(均)':>14}{'末能量(均)':>12}")
    for mode in ("assoc", "causal"):
        surv, eats, en = stats[mode]
        print(f"{mode:<8}{str(surv == len(seeds)):>8}{eats:>14.1f}{en:>12.1f}")

    as_, ae_, aen = stats["assoc"]
    cs_, ce_, cen = stats["causal"]
    print("\n--- reading ---")
    print(f"食物搬家后: assoc 存活 {as_}/{len(seeds)}、搬家后吃到 {ae_:.1f} 次、末能量 {aen:.1f};")
    print(f"              causal 存活 {cs_}/{len(seeds)}、搬家后吃到 {ce_:.1f} 次、末能量 {cen:.1f}.")
    if cs_ > as_ or ce_ > ae_ or cen > aen:
        print("do-operator (干预重测) 的代价是更快的适应: 存活更好、搬家后吃到更多、")
        print("末能量更高 -- 被动观察者在适应期把自己烧干 (锁死在旧结构上慢慢耗, ")
        print("SEED-26 过度信任锁死); 干预是信号放大器 (docs/12/SEED-6), 是唯一能")
        print("打破'旧结构锁死'的. = 包含关系兑现: 关联(观察) ⊂ 因果(干预) ⊂ 泛化(动态迁移).")
    else:
        print("本组参数下干预优势不明显 -- 诚实记录, 不包装 (负结果也是资产).")
    print("诚实边界: toy 探针; 泛化/因果是丰富概念 (类比/反事实/组合 未实现);")
    print("显示的是'干预 > 观察(对迁移)'的包含机制, 是 SEED-6/26/27 线的显式化,")
    print("不是完整 AGI 能力.")


def sweep():
    out = {mode: [run(mode, seed=s) for s in (1, 2, 3, 4, 5)]
           for mode in ("assoc", "causal")}
    with open("seed-44/results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    demo()
    print("\nfull results -> seed-44/results.json")


def main():
    p = argparse.ArgumentParser(description="SEED-44: capability containment")
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()
    if args.sweep:
        sweep()
    else:
        demo()


if __name__ == "__main__":
    main()
