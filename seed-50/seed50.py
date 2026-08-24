"""
seed-50/seed50.py -- 反事实探针：可重放世界（docs/105）

起点（用户对话）：看百万粉 up 玩"拼接前后故事"游戏拼青蛙王子——四个框、
两个场景选择（森林/亲吻）、三个角色、每框最多两人，拼半天只会随机换位摸
机制。随后用户澄清真实机制，本 seed 按真实机制重做世界规则书：
    - 角色：公主/王子/女巫（青蛙=王子被诅咒后的形态，同一角色位）
    - 关系状态全局累积、跨框保留（框一相爱 → 框四亲吻才成功）
    - 场景类型决定交互：亲吻场景=表白/亲吻，森林场景=相遇/施法
    - 女巫要"先被拒、后施法"（被拒是变青蛙的前提）
    - 亲吻要"先相爱、且王子已是青蛙"

核心主张（可证伪）：
    反事实在被动观察里不存在（只见过一条发生路径），但在**可重放世界**里
    存在——状态可存、时间线可剪、决策可换、分支可重跑。于是"如果没发生 X
    会怎样"第一次有一个**世界裁决**的答案，而不是语言先验。因果推断的瓶颈
    不是算法，是**有没有一个能按回去重跑的世界接口**。

事件表（世界规则书：同框角色对 × 场景 × 全局状态）：
    fall_in_love    森林 {公主,王子}       → 相爱            （无前提；需王子人形）
    witch_declares  亲吻 {女巫,王子}       → 女巫爱上+王子拒绝（无前提；需王子人形）
    curse           森林 {女巫,王子}       → 王子变青蛙       （前提：被拒）
    true_love_kiss  亲吻 {公主,王子(青蛙)} → 诅咒解除         （前提：相爱+青蛙）
    成功 = ending=happy（真爱之吻触发）。依赖是偏序：相爱与表白被拒可换位，
    但相爱需王子人形 → 相爱必须在诅咒之前 → 可行解 2 个（不是 3 个）。

探针：
  A 反事实重放：参考解剪枝换决策，重跑，diff——
    不吻（青蛙永远青蛙）/ 想跳过"被拒"直接施法（curse 执行不了=结构事实）/
    相爱放晚点（可行解2：相爱与表白被拒可换位，依赖是偏序不是全序）。
  B 拼接排序：4 框配置三种 solver：
      random    = 均匀随机盲试（38416 种配置只有 2 种可行 → 期望 ~1.9 万次；
                  这就是"拼半天力竭"的原因——纯随机是绝望的，通关靠语义线索）
      textprior = 只有故事梗概（内容：森林施法、亲吻解咒；结构盲：不知道
                  施法需先被拒、亲吻需青蛙形态）→ 违规 2 框跑不通
      world     = 读规则书推导依赖序 → 一次成功（2 个可行解之一）
  C 判定：穷举 38416 种配置，可行解 2 个（相爱需王子人形=相爱在诅咒前；
     依赖是偏序：相爱与表白被拒可换位）。

诚实边界：规则是按用户澄清写的（docs/38）；重放是机制（能力层），不宣称
agent "理解"；world 臂能读规则书不是自学习；toy 世界。

Run:
  python seed-50/seed50.py
  python seed-50/seed50.py --probe counterfactual
  python seed-50/seed50.py --probe splice --seeds 100
  python seed-50/seed50.py --probe verify
  python seed-50/seed50.py --sweep
"""

import argparse
import json
import random
from itertools import combinations, product

# ---------------- 世界（用户澄清的真实游戏机制） ----------------
# 角色：公主(princess)/王子(prince)/女巫(witch)；青蛙=王子的形态。
# 场景：森林(forest)/亲吻(kiss)。4 个框，每框=场景+0~2 角色，按序执行。

SCENES = ("forest", "kiss")
ROLES = ("princess", "prince", "witch")


def fresh_state():
    return {
        "love": False,        # 公主&王子相爱
        "witch_love": False,  # 女巫爱上王子
        "rejected": False,    # 王子拒绝女巫
        "form": "prince",     # 王子形态: "prince" | "frog"
        "ending": None,       # None | "happy"
    }


def clone(s):
    return dict(s)


def set_(s, **kw):
    s.update(kw)
    return True


# 事件表 = 世界规则书。同一角色对在不同场景 = 不同事件（场景决定交互性质）。
# pre/fx 用于执行；needs/effects 用于规划器（derive_plan 读它们推依赖序）。
EVENTS = {
    "fall_in_love": {
        "desc": "森林相遇：公主与王子相爱",
        "scene": "forest", "pair": ("princess", "prince"),
        "needs": [("form", "prince")],
        "effects": {"love": True},
        "pre": lambda s: s["form"] == "prince" and not s["love"],
        "fx": lambda s: set_(s, love=True),
    },
    "witch_declares": {
        "desc": "亲吻场景：女巫向王子表白，王子拒绝",
        "scene": "kiss", "pair": ("witch", "prince"),
        "needs": [("form", "prince")],
        "effects": {"witch_love": True, "rejected": True},
        "pre": lambda s: s["form"] == "prince" and not s["rejected"],
        "fx": lambda s: set_(s, witch_love=True, rejected=True),
    },
    "curse": {
        "desc": "森林相遇：女巫把王子变成青蛙（前提：先被拒）",
        "scene": "forest", "pair": ("witch", "prince"),
        "needs": [("form", "prince"), ("rejected", True)],
        "effects": {"form": "frog"},
        "pre": lambda s: s["form"] == "prince" and s["rejected"],
        "fx": lambda s: set_(s, form="frog"),
    },
    "true_love_kiss": {
        "desc": "亲吻场景：公主亲吻青蛙，诅咒解除（前提：相爱+青蛙）",
        "scene": "kiss", "pair": ("princess", "prince"),
        "needs": [("form", "frog"), ("love", True)],
        "effects": {"form": "prince", "ending": "happy"},
        "pre": lambda s: s["form"] == "frog" and s["love"] and s["ending"] is None,
        "fx": lambda s: set_(s, form="prince", ending="happy"),
    },
}

# 参考解（用户给的正确解法）
CANONICAL = [
    ("forest", ("princess", "prince")),   # 框一：森林，公主&王子相爱
    ("kiss", ("witch", "prince")),        # 框二：亲吻，女巫表白被拒
    ("forest", ("witch", "prince")),      # 框三：森林，女巫施法变青蛙
    ("kiss", ("princess", "prince")),     # 框四：亲吻，真爱之吻解咒
]

# 单框配置池：2 场景 × (0/1/2 角色组合) = 14 种
FRAME_POOL = [(sc, ch) for sc in SCENES
              for r in range(3) for ch in combinations(ROLES, r)]


def frame_event(scene, chars):
    """一个框（场景+角色集合）匹配哪个事件；返回事件名或 None（无效框）。"""
    for name, ev in EVENTS.items():
        if ev["scene"] == scene and set(ev["pair"]) == chars:
            return name
    return None


def play_frames(frames):
    """按顺序执行 4 框配置。返回（终态, 事件日志）。日志含每框成败与前后状态。"""
    s = fresh_state()
    log = []
    for t, (scene, chars) in enumerate(frames):
        pre = clone(s)
        name = frame_event(scene, set(chars))
        ev = EVENTS[name] if name else None
        ok = False
        if ev is not None and ev["pre"](s):
            ev["fx"](s)
            ok = True
        log.append({"t": t, "scene": scene, "chars": sorted(chars),
                    "event": name, "ok": ok,
                    "pre": pre, "post": clone(s)})
    return s, log


def quick_play(frames):
    """无日志版（random 臂用）：返回（ending, 无效框数）。"""
    s = fresh_state()
    invalid = 0
    for scene, chars in frames:
        name = frame_event(scene, set(chars))
        ev = EVENTS[name] if name else None
        if ev is not None and ev["pre"](s):
            ev["fx"](s)
        else:
            invalid += 1
    return s["ending"], invalid


def diff_state(a, b):
    keys = sorted(set(a) | set(b))
    return {k: (a.get(k), b.get(k)) for k in keys if a.get(k) != b.get(k)}


# ---------------- 探针 A：反事实重放 ----------------

def probe_counterfactual():
    print("=== A 反事实重放：'如果…'由世界算出来，不是语言先验 ===")
    s_can, log_can = play_frames(CANONICAL)
    print(f"参考解: 相爱={s_can['love']} 被拒={s_can['rejected']} "
          f"形态={s_can['form']} 结局={s_can['ending']}")

    # CF1: 框四不吻（换成森林+公主&王子——王子已是青蛙，相爱事件前提不满足）
    frames1 = list(CANONICAL[:3]) + [("forest", ("princess", "prince"))]
    s1, lg1 = play_frames(frames1)
    d1 = diff_state(s_can, s1)
    print(f"CF1 如果框四不亲吻（换成森林相遇）: {d1}")
    print("   -> 世界裁决：王子永远是青蛙，没有真爱之吻，故事不完整")

    # CF2: 想跳过"被拒"直接施法（框二就放森林+女巫&王子）
    frames2 = [("forest", ("princess", "prince")),
               ("forest", ("witch", "prince")),
               ("kiss", ("witch", "prince")),
               ("kiss", ("princess", "prince"))]
    s2, lg2 = play_frames(frames2)
    bad2 = [e["t"] for e in lg2 if not e["ok"]]
    print(f"CF2 如果玩家想跳过表白、框二直接施法: 无效框 {bad2} "
          f"-> 世界裁决：没有'被拒'这个前因，curse 动作执行不了"
          f"（结构事实，梗概里没有）；没青蛙，kiss 也执行不了")
    d2 = diff_state(s_can, s2)
    print(f"   -> 终点差异: {d2}")

    # CF3: 相爱放晚点（可行解 2：相爱与表白被拒可换位）
    frames3 = [("kiss", ("witch", "prince")),
               ("forest", ("princess", "prince")),
               ("forest", ("witch", "prince")),
               ("kiss", ("princess", "prince"))]
    s3, lg3 = play_frames(frames3)
    d3 = diff_state(s_can, s3)
    print(f"CF3 如果框一先表白被拒、框二再相爱（相爱放晚点）: 终态差异 {d3}")
    print("   -> 世界裁决：照样通关——依赖是偏序不是全序，相爱与表白被拒可换位；"
          "同一个终态，不同的框顺序（docs/87：时间在快照里不可见）")

    return {
        "canonical": {"love": s_can["love"], "rejected": s_can["rejected"],
                      "form": s_can["form"], "ending": s_can["ending"]},
        "cf1_no_kiss": {"form": s1["form"], "ending": s1["ending"], "diff": d1},
        "cf2_skip_rejection": {"invalid_frames": bad2, "ending": s2["ending"],
                               "diff": d2},
        "cf3_late_love": {"ending": s3["ending"], "diff": d3},
    }


# ---------------- 探针 B：拼接排序（三臂） ----------------

def solve_random(max_tries, rng):
    """model-free 盲试：均匀随机抽 4 框配置直到跑通；封顶=力竭。"""
    tries = 0
    invalid = 0
    while True:
        tries += 1
        frames = tuple(rng.choice(FRAME_POOL) for _ in range(4))
        ending, inv = quick_play(frames)
        invalid += inv
        if ending == "happy":
            return {"trials": tries, "invalid": invalid, "success": True,
                    "exhausted": False}
        if tries >= max_tries:
            return {"trials": tries, "invalid": invalid, "success": False,
                    "exhausted": True}


TEXTPRIOR_FRAMES = [
    ("forest", ("princess", "prince")),   # 相爱（梗概里公主王子先相爱——内容对）
    ("forest", ("witch", "prince")),      # 想直接施法——结构盲：不知道需先被拒
    ("kiss", ("witch", "prince")),        # 表白被拒（内容对，但顺序错了：施法在前）
    ("kiss", ("princess", "prince")),     # 想亲吻——结构盲：王子没变青蛙，吻不生效
]


def solve_textprior():
    """LLM 式：只有故事梗概（内容），没有世界规则（结构）。一次提交，不试错。"""
    frames = TEXTPRIOR_FRAMES
    s, log = play_frames(frames)
    bad = [e["t"] for e in log if not e["ok"]]
    return {"trials": 1, "invalid": len(bad), "invalid_frames": bad,
            "success": s["ending"] == "happy", "frames": frames}


def derive_plan():
    """STRIPS 式前向规划：读规则书（needs/effects），推出可行执行序。
    关键：每轮只执行一个可执行事件后重新扫描——避免同一轮里执行顺序互相
    破坏（如 curse 把王子变青蛙后，fall_in_love 的"需王子人形"就永远不满足了）。
    贪心死锁时用 DFS 兜底（≤24 排列）。"""
    s0 = fresh_state()

    def greedy():
        s = clone(s0)
        remaining = set(EVENTS)
        plan = []
        while remaining:
            progressed = False
            for name in sorted(remaining):
                ev = EVENTS[name]
                if all(s[f] == v for f, v in ev["needs"]):
                    s.update(ev["effects"])
                    plan.append((ev["scene"], ev["pair"]))
                    remaining.remove(name)
                    progressed = True
                    break
            if not progressed:
                return plan, remaining
        return plan, remaining

    plan, remaining = greedy()
    if not remaining:
        return plan

    # 贪心死锁 → DFS 兜底（本规则书不会发生，双保险）
    def dfs(s, remaining, acc):
        if not remaining:
            return acc if s["ending"] == "happy" else None
        for name in sorted(remaining):
            ev = EVENTS[name]
            if all(s[f] == v for f, v in ev["needs"]):
                s2 = clone(s)
                s2.update(ev["effects"])
                r = dfs(s2, remaining - {name}, acc + [(ev["scene"], ev["pair"])])
                if r:
                    return r
        return None

    return dfs(clone(s0), set(EVENTS), []) or []


def solve_world():
    frames = derive_plan()
    s, log = play_frames(frames)
    bad = [e["t"] for e in log if not e["ok"]]
    return {"trials": 1, "invalid": len(bad), "invalid_frames": bad,
            "success": s["ending"] == "happy", "frames": frames}


def probe_splice(max_tries=30000, seeds=100, rng=None):
    print("=== B 拼接排序：4 框配置（场景×角色）盲试 vs 内容 vs 规则书 ===")
    rng = rng or random.Random(1)
    r_trials, r_inv, r_exh = [], [], 0
    for _ in range(seeds):
        r = solve_random(max_tries, rng)
        r_trials.append(r["trials"])
        r_inv.append(r["invalid"])
        r_exh += int(r["exhausted"])
    total = len(FRAME_POOL) ** 4
    mean = sum(r_trials) / len(r_trials)
    med = sorted(r_trials)[len(r_trials) // 2]
    print(f"random    臂 ({seeds} seeds): 平均 {mean:.0f} 次盲试才摸到机制 "
          f"({total} 种配置只有 2 种可行, 理论期望 {total / 2:.0f}), "
          f"中位 {med}, 封顶力竭 {r_exh}/{seeds}, 累计无效框 {sum(r_inv)} 次")

    tp = solve_textprior()
    print(f"textprior 臂: 1 次提交, 无效框 {tp['invalid']} ({tp['invalid_frames']}), "
          f"跑通? {tp['success']}  <- 内容有、结构盲")

    w = solve_world()
    print(f"world     臂: 1 次提交, 无效框 {w['invalid']}, 跑通? {w['success']} "
          f"<- 读规则书一次成功")

    return {
        "random": {"seeds": seeds, "pool_size": total,
                   "mean_trials": round(mean, 1), "median_trials": med,
                   "theoretical_mean": round(total / 2.0, 1),
                   "exhausted": r_exh, "total_invalid": sum(r_inv)},
        "textprior": {"trials": tp["trials"], "invalid": tp["invalid"],
                      "invalid_frames": tp["invalid_frames"],
                      "success": tp["success"]},
        "world": {"trials": w["trials"], "invalid": w["invalid"],
                  "success": w["success"]},
    }


# ---------------- 探针 C：判定（穷举） ----------------

def probe_verify():
    print("=== C 判定：穷举全部 4 框配置 ===")
    n_ok = 0
    examples = []
    for frames in product(FRAME_POOL, repeat=4):
        ending, _ = quick_play(frames)
        if ending == "happy":
            n_ok += 1
            if len(examples) < 4:
                examples.append([list(f) for f in frames])
    total = len(FRAME_POOL) ** 4
    print(f"{total} 种配置中可行解 {n_ok} 个 -> 依赖是偏序："
          f"相爱/表白被拒可换位（{n_ok} 个可行解，不止参考解一个）；"
          f"相爱需王子人形=相爱必须在诅咒之前")
    for ex in examples:
        print("   例:", ex)
    return {"valid_configs": n_ok, "total": total}


def main():
    p = argparse.ArgumentParser(description="SEED-50 反事实探针：可重放世界 (docs/105)")
    p.add_argument("--probe", choices=["counterfactual", "splice", "verify", "all"],
                   default="all")
    p.add_argument("--trials", type=int, default=30000,
                   help="random 臂单次封顶尝试数（默认 30000）")
    p.add_argument("--seeds", type=int, default=100, help="random 臂统计种子数")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--sweep", action="store_true", help="跑统计并写 results.json")
    args = p.parse_args()

    results = {}
    if args.probe in ("counterfactual", "all"):
        results["counterfactual"] = probe_counterfactual()
    if args.probe in ("splice", "all"):
        results["splice"] = probe_splice(args.trials, args.seeds,
                                         random.Random(args.seed))
    if args.probe in ("verify", "all"):
        results["verify"] = probe_verify()

    if args.sweep:
        with open("seed-50/results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        print("\nfull results -> seed-50/results.json")


if __name__ == "__main__":
    main()
