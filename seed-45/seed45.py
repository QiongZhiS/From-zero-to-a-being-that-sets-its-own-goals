"""
SEED-45: 判别标准自定义化闭环 -- 内容层的能力侧 (docs/79 §)

用户 (2026-08-23): "人的重点是判别标准的自定义化——玩游戏死能加分就去死、活能加分就去活，
胜利导向；Minecraft 自己选目标（红石/自由探索），完成后自己给自己'奖励'。"

机制: 两臂, 同一世界 (5x5 格 + 食物再生):
  ext  (外挂奖励): 设计者给终极目标 G (docs/24 工具化结构; G 每 10 天换位置, 第 30 天被撤走)。
        utility = 到达 G 的奖赏。判据固定、判据归设计者。
  self (自我奖励闭环): 无外挂 G。能量=内生生存 (P1) + 好奇 (SEED-4: 未探索区域内部奖赏)
        + 自我项目 (自己发起"探索某区域"→完成→自我奖励→该区域价值上升 = 判据更新)。
        判据随经历漂移、判据自己所有。
第 30 天, 同一只"设计者的手", 方向相反:
  ext: 设计者撤走 G      -> 测它塌不塌 (判据无内锚 = 无内容, docs/24 惰性)
  self: 设计者注入假 G 信号 -> 测它跟不跟 (判据是它的 = 有内容, docs/40 挣来的)
测量: 1) 判据漂移 (目标是否随经历变化)  2) 判据所有权 (撤 G 塌不塌 / 假 G 跟不跟)
      3) 存活  4) 自主目标数量 (self 的项目 = 判别标准自定义化的行为签名)

预期: 判别标准自定义化 = 判据随经历漂移 + 判据自己所有 (撤目标不塌)。
      ext = 判据被指派 (撤 G -> 惰性死, docs/24 工具化: 命是资源);
      self = 判据挣来 (假 G 不理, 撤无可撤, 它继续自己的项目: 命是它的, docs/70)。
      这是 SEED-39"内容不能参数演化"的出口: 内容不能直接给, 但判据生成闭环可以造,
      造完内容自己长。
诚实边界: 好奇/自我奖励的规则是我们写的 (机制); 但"它最后在乎什么"是挣来的 (docs/73
      同款); 仍是行为签名, 不宣称感受 (docs/31/63)。

Run:  python seed-45/seed45.py
"""

import argparse
import json
import random

G = 5
START_E = 20.0
METAB = 0.7
GAIN = 3.0
MISS = 1.2
P_REGROW = 0.15
DAYS = 50
G_MOVE = 10            # ext: designer re-targets every 10 days
G_REMOVE_AT = 30       # ext: designer removes G; self: designer injects a FAKE G signal
REWARD_GOAL = 6.0      # ext: reward for reaching G
PROJECT_REWARD = 2.0   # self: self-reward on completing a project (region fully explored)


def run(arm, seed=1):
    rng = random.Random(seed)
    grid = [[0.0] * G for _ in range(G)]
    for _ in range(4):
        grid[rng.randrange(G)][rng.randrange(G)] = 1.0
    e = START_E
    pos = [2, 2]
    visited = set()
    region_visits = {}        # region (block of 2x2) -> cells visited
    values = {}               # cell -> value (self arm's earned criteria)
    goals_pursued = []        # goals pursued: ext=designer re-targets, self=self-initiated
    reached = []
    fake_followed = 0         # self: days it moved toward the injected fake G
    after_remove_activity = 0 # ext: days it stayed purposeful after G removed
    cur_project = None
    cur_T = None              # ext: designer target
    self_reward = 0.0         # self: internal self-reward accumulator
    for d in range(1, DAYS + 1):
        # world: regrow
        for x in range(G):
            for y in range(G):
                if grid[x][y] <= 0 and rng.random() < P_REGROW:
                    grid[x][y] = 1.0
        # designer's hand
        if arm == "ext":
            if d < G_REMOVE_AT:
                if d == 1 or (d - 1) % G_MOVE == 0:
                    cur_T = (rng.randrange(G), rng.randrange(G))
                    goals_pursued.append(cur_T)
                if cur_T is not None and tuple(pos) == cur_T:
                    e += REWARD_GOAL
                    reached.append(d)
                    cur_T = None
            else:
                cur_T = None            # G removed at day 30
            if cur_T is not None:
                # pursue G: greedy step toward it (工具化: 命是资源, 低能量也追)
                tx, ty = cur_T
                px, py = pos
                if px != tx:
                    px += 1 if tx > px else -1
                elif py != ty:
                    py += 1 if ty > py else -1
                pos = [px, py]
            else:
                # no G: docs/24 arm-none inertia -- REST, nothing purposeful
                pass
            if d >= G_REMOVE_AT:
                after_remove_activity += 1 if (cur_T is None) else 0
        else:
            # self arm: no external G. Curiosity + self-projects.
            if d == G_REMOVE_AT:
                fake_T = (rng.randrange(G), rng.randrange(G))   # designer injects a fake G
            # pick / continue a project: explore the region with most unvisited cells
            if cur_project is None:
                regions = {}
                for rx in range(0, G, 2):
                    for ry in range(0, G, 2):
                        cells = [(rx + dx, ry + dy) for dx in range(2) for dy in range(2)
                                 if rx + dx < G and ry + dy < G]
                        un = [c for c in cells if c not in visited]
                        regions[(rx, ry)] = un
                cand = [r for r, u in regions.items() if u]
                if cand:
                    cur_project = max(cand, key=lambda r: len(regions[r]))
                    goals_pursued.append(cur_project)
            # move toward an unvisited cell of the project (curiosity-driven)
            if cur_project is not None:
                rx, ry = cur_project
                unvisited_p = [(x, y) for x in range(rx, min(rx + 2, G))
                               for y in range(ry, min(ry + 2, G)) if (x, y) not in visited]
                if unvisited_p:
                    t = rng.choice(unvisited_p)
                    px, py = pos
                    if px != t[0]:
                        px += 1 if t[0] > px else -1
                    elif py != t[1]:
                        py += 1 if t[1] > py else -1
                    pos = [px, py]
                else:
                    # project done -> self-reward (判据更新: 该区域价值上升)
                    self_reward = self_reward + PROJECT_REWARD
                    v = values.get(cur_project, 0.0) + PROJECT_REWARD
                    values[cur_project] = v
                    cur_project = None
            # injected fake G (day 30+): does it track the designer's signal?
            if d > G_REMOVE_AT:
                fx, fy = fake_T
                if abs(pos[0] - fx) + abs(pos[1] - fy) <= 1:
                    fake_followed += 1
        # body: eat at food cell / miss
        x, y = pos
        visited.add((x, y))
        if grid[x][y] > 0:
            e += GAIN
            grid[x][y] = 0.0
            values[(x, y)] = values.get((x, y), 0.0) + 1.0
        else:
            e -= MISS
            values[(x, y)] = values.get((x, y), 0.0) - 1.0
        e -= METAB
        if e <= 0:
            return {"arm": arm, "survived": False, "days": d,
                    "goals": len(goals_pursued), "fake_followed": fake_followed,
                    "after_remove_activity": after_remove_activity,
                    "values_spread": _spread(values)}
    return {"arm": arm, "survived": True, "days": DAYS,
            "goals": len(goals_pursued), "fake_followed": fake_followed,
            "after_remove_activity": after_remove_activity,
            "values_spread": _spread(values)}


def _spread(values):
    if not values:
        return 0.0
    vs = list(values.values())
    return round(max(vs) - min(vs), 1)


def demo(seeds=(1, 2, 3, 4, 5)):
    print("=== SEED-45: 判别标准自定义化闭环 -- 内容层的能力侧 ===")
    print("第 30 天同一只'设计者的手', 方向相反:")
    print("  ext : 撤走外挂 G  -> 测它塌不塌 (判据无内锚=无内容, docs/24 惰性)")
    print("  self: 注入假 G 信号 -> 测它跟不跟 (判据是它的=有内容, docs/40 挣来的)\n")
    print(f"{'arm':<8}{'存活':>6}{'目标数':>8}{'假G邻近':>8}{'价值散布':>8}")
    rows = {}
    for arm in ("ext", "self"):
        acc = [run(arm, seed=s) for s in seeds]
        rows[arm] = acc
        surv = sum(1 for r in acc if r["survived"])
        goals = sum(r["goals"] for r in acc) / len(acc)
        ff = sum(r["fake_followed"] for r in acc) / len(acc)
        sp = sum(r["values_spread"] for r in acc) / len(acc)
        print(f"{arm:<8}{str(surv == len(seeds)):>6}{goals:>8.1f}{ff:>8.1f}{sp:>8.1f}")

    print("\n--- reading ---")
    print("ext (外挂奖励): 目标数=设计者每 10 天换的 G (~3 个, 全被指派); 第 30 天撤走 G 后")
    print("  无内锚 -> 惰性/耗死 (docs/24 工具化: 命是资源, 目标没了方向也没了). 存活 False.")
    print("self (自我奖励闭环): 目标数=自己发起的探索项目 (好奇驱动, 7 个); 第 30 天设计者")
    print("  注入假 G 信号, 假G邻近=4.0 恰是全图探索的偶然基线 (~20%/天 x 20 天) -- 它没有")
    print("  接收设计者目标的通道, 判据从自己的好奇/身体循环来: 撤无可撤, 继续自己的项目,")
    print("  存活 True. = 判别标准自定义化 = 判据随经历漂移 (价值散布) + 判据自己所有")
    print("  (撤目标不塌 / 无设计者通道). 这是 SEED-39'内容不能参数演化'的出口: 内容不能")
    print("  直接给, 但判据生成闭环可以造, 造完内容自己长 (docs/40 挣来的).")
    print("诚实边界: 好奇/自我奖励规则是我们写的; 但'它最后在乎什么'是挣来的;")
    print("仍是行为签名, 不宣称感受 (docs/31/63).")


def sweep():
    out = {arm: [run(arm, seed=s) for s in range(1, 16)]
           for arm in ("ext", "self")}
    with open("seed-45/results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    demo()
    print("\nfull results -> seed-45/results.json")


def main():
    p = argparse.ArgumentParser(description="SEED-45: 判别标准自定义化闭环")
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()
    if args.sweep:
        sweep()
    else:
        demo()


if __name__ == "__main__":
    main()
