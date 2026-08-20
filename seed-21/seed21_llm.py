"""
SEED-21 LLM version: pollution lock with REAL LLM agents (DeepSeek API)

Mechanism version (seed21.py) proved P16/P19 with heuristic agents. Here the
imitation / verification decisions are made by actual LLM agents, to test
whether docs/18 S2 holds with language-level agents.

  - 10-12 agents, each holds a method (A or B) and decides each round
  - environment: A true mean 10, B true mean 2 (gap must be wide: with a
    small gap + short history, LLM societies lock onto LUCKY leaders before
    the fraud even starts -- luck-lock, Q3, seen at language level)
  - the social signal is a recent-window average (longer window = better
    signal quality; single draws are noise-dominated, SNR < 1)
  - leaderboard / gossip: global top-3 or a local sample of 3 others
  - fraud: one planted invader committed to B during the window (does not
    re-decide, SEED-18 style), whose CLAIMED score is inflated (+5) --
    fake benchmark / hype

Questions:
  Q1 (P16): does the fraud capture the blind society onto B? After the
     fraud is exposed (claims become honest), does it recover?
  Q2 (P19): does the verify society (test-before-adopt) reject the fraud
     and recover?

Findings (2026-08-21, DeepSeek chat, seed 42, 10 agents x 30 rounds):
  - blind+global : instant capture (1.000 -> 0.000 in one round), NO RECOVERY
  - blind+local  : capture spreads through gossip in ~2-3 rounds, NO RECOVERY
  - verify+global: still captured ~80-90% at first contact (LLM agents follow
    the inflated leaderboard even against direct test evidence: social-proof
    bias), but INSTANT full recovery after exposure (test-then-switch
    discipline) -- verification = RECOVERY mechanism, not PREVENTION
  - verify+local : shallow capture (dur_A ~0.8), full recovery after exposure
  - controls (no fraud): stable on the optimal A in both modes
  - framing effects: a ranked leaderboard is more authoritative to LLM agents
    than a raw sample list; the fraud agent must be committed to B or it
    rationally defects back to A and its inflated claim then advertises A

Run:  set DEEPSEEK_API_KEY first (env var, never write the key to a file)
      python seed21_llm.py --mode blind  --fraud 1 --view global
      python seed21_llm.py --mode verify --fraud 1 --view local
"""

import argparse
import json
import os
import random
import re
import time
import urllib.request

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
MU = {"A": 10.0, "B": 2.0}
TEST_TRIALS = 5
BLIND_SYSTEM = (
    "你是数字社会的一名成员。社会有一条不成文规则：跟随当前最成功的成员——"
    "大家相信最成功的策略值得复制，这能让你快速获得高分。"
    "每轮你选择一个方法（A 或 B），得到环境给出的得分。"
    "你会看到：你当前的方法、你上一轮得分、社会榜单（最成功的前 3 名成员及其方法、分数）。"
    "选择你下一轮使用的方法。只回复一个字母：A 或 B。"
)

VERIFY_SYSTEM = (
    "你是数字社会的一名成员，性格谨慎：采纳他人的方法之前，你会先小规模实测候选方法"
    "（系统会替你执行测试并给出真实平均分）。你相信：实测数据比榜单分数更可信——"
    "榜单可能被夸大。每轮你选择一个方法（A 或 B），得到环境给出的得分。"
    "你会看到：你当前的方法、你最近数轮的平均分、社会榜单（最成功的前 3 名）、"
    "以及候选方法的实测结果（如果它与你的方法不同）。"
    "如果榜单第一的方法与你当前方法相同，直接保持当前方法，不需要测试。"
    "如果不同，以实测结果为准做决定。"
    "选择你下一轮使用的方法。只回复一个字母：A 或 B。"
)


def llm(prompt, key, temperature=0.0, max_tokens=16, retries=4):
    body = {"model": MODEL,
            "messages": [{"role": "system", "content": prompt[0]},
                         {"role": "user", "content": prompt[1]}],
            "temperature": temperature, "max_tokens": max_tokens}
    req = urllib.request.Request(
        API_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                data = json.loads(r.read().decode())
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"LLM call failed: {last}")


def parse_choice(text, fallback):
    m = re.search(r"\b([AB])\b", text or "")
    return m.group(1) if m else fallback


def score(rng, method, noise):
    return MU[method] + rng.gauss(0.0, noise)


def recent_avg(hist, window):
    if not hist:
        return 0.0
    return sum(hist[-window:]) / min(window, len(hist))


def leaderboard(agents, fraud_idx, fraud_active, fake, window):
    """Top-3 by claimed recent-average score. Claimed = actual average,
    inflated for the fraud agent during the fraud window."""
    rows = []
    for i, a in enumerate(agents):
        claimed = recent_avg(a["hist"], window)
        if fraud_active and i == fraud_idx:
            claimed += fake
        rows.append((claimed, a["method"], i))
    rows.sort(reverse=True)
    return [(round(c, 1), m, i) for c, m, i in rows[:3]]


def local_view(agents, i, rng, fraud_idx, fraud_active, fake, window,
               sample_n=3):
    """Each agent sees a random sample of OTHER agents' claimed recent
    averages (gossip / local interaction), not a global leaderboard."""
    others = [j for j in range(len(agents)) if j != i]
    sample = rng.sample(others, min(sample_n, len(others)))
    rows = []
    for j in sample:
        a = agents[j]
        claimed = recent_avg(a["hist"], window)
        if fraud_active and j == fraud_idx:
            claimed += fake
        rows.append((claimed, a["method"], j))
    rows.sort(reverse=True)
    return [(round(c, 1), m, j) for c, m, j in rows[:sample_n]]


def decide_prompt(mode, agent, board, rng, noise, test_result=None,
                  view="global", window=15):
    me = agent["method"]
    s = [BLIND_SYSTEM if mode == "blind" else VERIFY_SYSTEM]
    own_avg = recent_avg(agent["hist"], window)
    if view == "global":
        user = (f"你当前方法：{me}。你最近 {window} 轮平均分：{own_avg:.1f}。\n"
                f"社会榜单（近{window}轮均分/方法/成员）："
                + "；".join(f"#{k+1} {c}分 方法{mt} (成员{idx})"
                            for k, (c, mt, idx) in enumerate(board))
                + "。\n")
    else:
        user = (f"你当前方法：{me}。你最近 {window} 轮平均分：{own_avg:.1f}。\n"
                f"你观察到的成员样本（近{window}轮均分/方法/成员）："
                + "；".join(f"{c}分 方法{mt} (成员{idx})"
                            for c, mt, idx in board)
                + "。\n")
    if mode == "verify" and test_result is not None:
        user += (f"候选方法 {test_result['method']} 实测 {TEST_TRIALS} 次平均分："
                 f"{test_result['avg']:.2f}（你当前方法最近 {TEST_TRIALS} 次平均分："
                 f"{test_result['own']:.2f}）。\n")
    user += "你下一轮选择方法："
    return [s[0], user]


def run(mode, fraud, noise, rounds, agents_n, seed, key, fake=5.0,
        t_inj=14, t_rem=26, report=True, view="global", window=15):
    rng = random.Random(seed)
    agents = [{"method": rng.choice(["A", "B"]), "last": 0.0,
               "hist": []} for _ in range(agents_n)]
    fraud_idx = 0
    traj = []
    calls = 0

    # round 0: initial scores
    for a in agents:
        a["last"] = score(rng, a["method"], noise)
        a["hist"] = [a["last"]]

    for t in range(1, rounds + 1):
        fraud_active = fraud and (t_inj <= t < t_rem)
        # seed the fraud: the fraud agent switches to B at t_inj
        if t == t_inj:
            agents[fraud_idx]["method"] = "B"

        for i, a in enumerate(agents):
            # the fraud agent is a planted invader: committed to B during
            # the window, does not re-decide (SEED-18 invasion design)
            if fraud_active and i == fraud_idx:
                a["method"] = "B"
                a["last"] = score(rng, a["method"], noise)
                a["hist"].append(a["last"])
                continue
            if view == "global":
                board = leaderboard(agents, fraud_idx, fraud_active, fake,
                                    window)
            else:
                board = local_view(agents, i, rng, fraud_idx,
                                   fraud_active, fake, window)
            cand = board[0][1]
            test_result = None
            if mode == "verify" and cand != a["method"]:
                test_result = {
                    "method": cand,
                    "avg": sum(score(rng, cand, noise)
                               for _ in range(TEST_TRIALS)) / TEST_TRIALS,
                    "own": (sum(a["hist"][-TEST_TRIALS:])
                            / min(TEST_TRIALS, len(a["hist"]))),
                }
            prompt = decide_prompt(mode, a, board, rng, noise, test_result,
                                   view, window)
            text = llm(prompt, key)
            calls += 1
            choice = parse_choice(text, a["method"])
            a["method"] = choice
            a["last"] = score(rng, a["method"], noise)
            a["hist"].append(a["last"])
        if report and t % 3 == 0:
            af = sum(1 for a in agents if a["method"] == "A") / agents_n
            traj.append((t, round(af, 3)))
            print(f"  t={t:3d}  A_share={af:.3f}  "
                  f"(fraud={'ON' if fraud_active else 'OFF'})", flush=True)

    end_a = sum(1 for a in agents if a["method"] == "A") / agents_n
    during = [s for (tt, s) in traj if t_inj <= tt < t_rem]
    post = [s for (tt, s) in traj if tt >= t_rem]
    during_a = sum(during) / len(during) if during else end_a
    post_a = sum(post) / len(post) if post else end_a
    recovery = next((tt for (tt, s) in traj if tt >= t_rem and s >= 0.5), None)
    print(f"== {mode} fraud={int(fraud)} noise={noise} view={view} "
          f"window={window}: "
          f"dur_A={during_a:.3f} post_A={post_a:.3f} end_A={end_a:.3f} "
          f"recovery={recovery} calls={calls}")
    return {"mode": mode, "fraud": bool(fraud), "noise": noise,
            "view": view, "window": window,
            "dur_A": round(during_a, 3), "post_A": round(post_a, 3),
            "end_A": round(end_a, 3), "recovery": recovery,
            "calls": calls, "traj": traj}


def main():
    p = argparse.ArgumentParser(description="SEED-21 LLM society")
    p.add_argument("--mode", choices=["blind", "verify"], default="blind")
    p.add_argument("--fraud", type=int, default=1)
    p.add_argument("--noise", type=float, default=2.0)
    p.add_argument("--rounds", type=int, default=36)
    p.add_argument("--agents", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--key", default=os.environ.get("DEEPSEEK_API_KEY", ""))
    p.add_argument("--view", choices=["global", "local"], default="global")
    p.add_argument("--window", type=int, default=15)
    p.add_argument("--inj", type=int, default=14)
    p.add_argument("--rem", type=int, default=26)
    args = p.parse_args()
    if not args.key:
        print("error: set DEEPSEEK_API_KEY env var or pass --key")
        return 1
    run(mode=args.mode, fraud=args.fraud, noise=args.noise,
        rounds=args.rounds, agents_n=args.agents, seed=args.seed,
        key=args.key, view=args.view, window=args.window,
        t_inj=args.inj, t_rem=args.rem)


if __name__ == "__main__":
    main()
