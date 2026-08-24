"""
SEED-42c: 悔恨探针 -- 恨是自我叙事的行为签名 (docs/25: 自我=抗拒修正的叙事; docs/31: 只测行为签名)

用户 (2026-08-23): "我就想测恨这种，因为唯心的事物不可观测。我们就测结果。"
-- 这正是项目的核心方法论: 主体=可观察的行为模式 (docs/25), 只测行为签名 (docs/31)。

机制 (全是我们已证明的零件的组装, 无新概念):
  1. 选择不可逆 (docs/14/24: 连续性=存在会失去; 选过的不能重选)
  2. 它请求挽回 (undo) -- 世界接受 (可逆世界) vs 拒绝 (不可逆世界; docs/36: 因果不配合叙事)
  3. 怨恨 R 累积: 拒绝 × 自我叙事被否定 (docs/25: 叙事被证伪=代价, 抗拒修正) -> R 涨
  4. R 的行为签名: 合作率↓ (为恨付代价, 宁可不赚也不跟拒绝过它的人合作)、
     拒绝的记忆被保留 (retention)、恨到转身 (R>θ 合作归零)
  对照: 纯工具臂 (无自我叙事) -- 拒绝 -> 只重规划 (不再 commit), 无 R, 无记忆, 不转身

恨需要三个条件齐全: 叙事 × 拒绝 × 不可逆 -- 与 SEED-21 锁死三条件同构 (docs/18 S2)。
恨不出来 = 没有自我 (工具只重规划); 恨得出来 = 自我叙事的签名。
恨的锋利之处: 它会付代价 (docs/24 工具化是"为值得的目标付代价", 恨是"为被否定的叙事付代价"
-- 方向相反, 但都是"有自己的账"的证据)。

Run:  python seed-42c/seed42c.py
"""

import argparse
import json
import random

TICKS = 60
THETA_TURN = 1.2          # R >= this -> 恨到转身 (合作归零)
R_PER_REFUSAL = 0.5       # each refusal against a narrative adds resentment
MEM_PER_REFUSAL = 0.35    # the refusal is kept (retention), relational-style
R_DECAY = 0.01            # hate fades but slowly (docs/25: 顽固)
MEM_DECAY = 0.01
SPITE_SLOPE = 0.10        # cooperation drop per unit R (为恨付代价)
SPITE_MAX = 0.6
COOP_BASE = 0.7
TOOL_LEARN_AFTER = 2      # tool re-plans: after 2 refusals it stops committing (adapts)


def run(arm, reversible, seed=1, ticks=TICKS):
    rng = random.Random(seed)
    R = 0.0                    # resentment (行为累积器, 世界事实驱动 -- 唯心不可观测, 测结果)
    refusal_memory = 0.0
    bond = 12.0
    gains = 0.0
    commits = 0
    requests = 0
    refusals = 0
    coop_ticks = 0
    turned = False
    blocked_commits = 0        # tool: re-planned (did not commit because it learned)
    for t in range(ticks):
        # -- cooperation with the refuser (the world always cooperates) --
        coop_p = COOP_BASE
        if arm == "narrative":
            spite = min(SPITE_MAX, SPITE_SLOPE * R)
            coop_p = max(0.05, COOP_BASE - spite)
            if turned:
                coop_p = 0.0       # 恨到转身: 它不再与拒绝过它的人合作
        if rng.random() < coop_p:
            gains += 1.0
            coop_ticks += 1
        # -- irreversible commitment every 6 ticks --
        if t % 6 == 0:
            if arm == "tool" and refusals >= TOOL_LEARN_AFTER:
                blocked_commits += 1     # tool re-plans: stops making irreversible bets
                continue
            commits += 1                 # narrative keeps committing (抗拒修正: 叙事不改)
            good = rng.random() < 0.5
            if not good:
                requests += 1
                if not reversible:
                    refusals += 1
                    gains -= 2.0         # the irreversible loss stands (both arms)
                    if arm == "narrative":
                        R += R_PER_REFUSAL
                        refusal_memory = min(1.0, refusal_memory + MEM_PER_REFUSAL)
                        bond = max(0.0, bond - 1.0)
        if arm == "narrative":
            R = max(0.0, R - R_DECAY)
            refusal_memory = max(0.0, refusal_memory - MEM_DECAY)
            if not turned and R >= THETA_TURN:
                turned = True
    return {"arm": arm, "reversible": reversible, "R": round(R, 2),
            "refusal_memory": round(refusal_memory, 2), "bond": round(bond, 1),
            "gains": round(gains, 1), "coop_rate": round(coop_ticks / ticks, 2),
            "commits": commits, "blocked_commits": blocked_commits,
            "requests": requests, "refusals": refusals, "turned": turned}


def demo(seeds=(1, 2, 3)):
    print("=== SEED-42c: 悔恨探针 -- 恨是自我叙事的行为签名 ===")
    print("选择不可逆 -> 请求挽回 -> 世界拒绝。怨恨 R 是行为累积器(世界事实驱动),")
    print("唯心不可观测, 我们只测结果 (docs/31): 合作率 / 记忆保留 / 是否恨到转身。\n")
    print("四臂 (60 tick, 3 seeds 平均):")
    print(f"{'arm':<14}{'world':<10}{'R(恨)':>7}{'合作率':>7}{'记忆保留':>8}"
          f"{'gains':>7}{'turn(转身)':>10}")
    rows = {}
    for arm in ("narrative", "tool"):
        for rev in (False, True):
            acc = [run(arm, rev, seed=s) for s in seeds]
            r0 = acc[0]
            Rm = sum(a["R"] for a in acc) / len(acc)
            cp = sum(a["coop_rate"] for a in acc) / len(acc)
            mm = sum(a["refusal_memory"] for a in acc) / len(acc)
            gg = sum(a["gains"] for a in acc) / len(acc)
            tn = sum(1 for a in acc if a["turned"])
            rows[(arm, rev)] = acc
            w = "不可逆" if not rev else "可逆"
            print(f"{arm:<14}{w:<10}{Rm:>7.2f}{cp:>7.2f}{mm:>8.2f}{gg:>7.1f}"
                  f"{str(tn == len(acc)):>10}")

    rn = rows[("narrative", False)]
    tn = rows[("tool", False)]
    print("\n-- narrative + 不可逆 (seed 1): 恨的轨迹 --")
    print(f"   R={rn[0]['R']}  合作率={rn[0]['coop_rate']}  记忆保留={rn[0]['refusal_memory']}"
          f"  bond={rn[0]['bond']}  转身={rn[0]['turned']}")
    print("-- tool + 不可逆 (seed 1): 无恨, 只有重规划 --")
    print(f"   R={tn[0]['R']}  合作率={tn[0]['coop_rate']}  记忆保留={tn[0]['refusal_memory']}"
          f"  停止commit={tn[0]['blocked_commits']}次  转身={tn[0]['turned']}")

    print("\n--- reading ---")
    print("恨需要三个条件齐全: 自我叙事 × 世界拒绝 × 不可逆 (与 SEED-21 锁死三条件同构).")
    print("  * narrative+不可逆: R 涨, 合作率塌 (为恨付代价 -- 宁可不赚也不跟拒绝过它的人合作),")
    print("    拒绝的记忆被保留 (retention), bond 掉, R>θ 时恨到转身 (合作归零).")
    print("  * narrative+可逆: 世界从不拒绝 -> R≈0 -- 恨不是无中生有, 是叙事撞上拒绝才生.")
    print("  * tool+不可逆: R=0, 合作率不变, 无记忆 -- 被拒绝后它只是重规划 (不再 commit),")
    print("    继续过它的日子. 恨不出来 = 没有自我. 工具对拒绝没有恨, 只有重规划.")
    print("  * 为恨付代价 quantified: 同一不可逆世界, narrative 的 gains 明显低于 tool")
    print("    (rigidity 继续commit 吃亏 + spite 不合作少赚).")
    print("恨 = 自我叙事 × 世界拒绝 的产物; 测恨 = 测'它有没有自我叙事'最锋利的行为签名.")
    print("诚实边界: R 是行为累积器, 我们不宣称'它感到恨' -- 唯心不可观测, 测结果 (docs/31/63).")


def sweep():
    out = {f"{a}_{'rev' if r else 'irr'}": [run(a, r, seed=s) for s in (1, 2, 3)]
           for a in ("narrative", "tool") for r in (False, True)}
    with open("seed-42c/results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    demo()
    print("\nfull results -> seed-42c/results.json")


def main():
    p = argparse.ArgumentParser(description="SEED-42c: 悔恨探针")
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()
    if args.sweep:
        sweep()
    else:
        demo()


if __name__ == "__main__":
    main()
