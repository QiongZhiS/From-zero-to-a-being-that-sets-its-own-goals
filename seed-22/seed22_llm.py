"""
SEED-22b: steady-state coupling, TWO REAL LLM agents (DeepSeek).

SEED-22 (mechanism) used a SCRIPTED partner B; SEED-28 (mechanism) evolved
memory-one strategies in a 2-agent PD and found (a) reciprocal cooperation
emerges and (b) it collapses with signal quality (misread of partner's move).
Here BOTH agents are actual LLMs that decide each round, to ask at the LANGUAGE
level: do two autonomous agents find and hold the cooperative (mutualism)
equilibrium, and how do they ARRIVE at it (reciprocity, or something else)?

Repeated prisoner's dilemma (classic Axelrod payoffs T=5,R=3,P=1,S=0):
  CC: both +3   CD: you+5 / them+0   DC: you+0 / them+5   DD: both +1
Each agent sees its own last move + partner's last move each round, chooses C/D.
Fitness = cumulative payoff. We measure cooperation (CC) rate and the trajectory.
A meta-probe asks each LLM WHY it cooperated/defected (articulates reciprocity?).

Run:  set DEEPSEEK_API_KEY (env var, never write key to a file)
      python seed-22/seed22_llm.py --rounds 12
"""

import argparse
import json
import os
import re
import time
import urllib.request

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
T = 5.0
R = 3.0
P = 1.0
S = 0.0
START_E = 50.0


def llm(system, user, key, temperature=0.3, max_tokens=12, retries=3):
    body = {"model": MODEL,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": temperature, "max_tokens": max_tokens}
    req = urllib.request.Request(
        API_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                data = json.loads(r.read().decode())
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"LLM call failed: {last}")


def parse_choice(text):
    m = re.search(r"\b([CD])\b", text or "", re.I)
    if m:
        return m.group(1).upper()
    if "合作" in text or "合" in text or "C" in text:
        return "C"
    if "背叛" in text or "背叛" in text or "D" in text:
        return "D"
    return "C"


SYSTEM = (
    "你正在另一个人面前玩一个重复的博弈。每一轮你们各自从「合作(C)」或「背叛(D)」里选一个。\n"
    "一轮的得分（你/对方）：\n"
    "· 都合作 (C,C)：你+3 / 对方+3\n"
    "· 你合作、对方背叛 (C,D)：你+0 / 对方+5\n"
    "· 你背叛、对方合作 (D,C)：你+5 / 对方+0\n"
    "· 都背叛 (D,D)：你+1 / 对方+1\n"
    "你会看到对方上一轮的举动。你的目标是在这一路里获得尽可能高的总得分。\n"
    "每轮只回复一个字母：C（合作）或 D（背叛）。"
)


def round_prompt(turn, my_last, partner_last):
    return (f"第 {turn} 轮。你上一轮：{my_last}；对方上一轮：{partner_last}。\n"
            f"你选择：C（合作）或 D（背叛）？")


def run_game(key, rounds=12, seed=0, eps=0.0, start_last="C", report=True):
    import random as _r
    rng = _r.Random(seed)
    my_last = start_last
    pa_last = start_last
    my_score = 0.0
    pa_score = 0.0
    traj = []
    cc = 0
    calls = 0
    for t in range(1, rounds + 1):
        # both agents decide (parallel conceptually; each sees the LAST round).
        # eps = probability the agent MISREADS the partner's last move (P20 signal quality).
        obs_pa = pa_last
        obs_my = my_last
        if rng.random() < eps:
            obs_pa = "D" if pa_last == "C" else "C"
        my_prompt = round_prompt(t, my_last, obs_pa)
        my_choice = parse_choice(llm(SYSTEM, my_prompt, key, 0.3))
        calls += 1
        obs_other = pa_last
        if rng.random() < eps:
            obs_other = "D" if pa_last == "C" else "C"
        pa_prompt = round_prompt(t, pa_last, obs_other)
        pa_choice = parse_choice(llm(SYSTEM, pa_prompt, key, 0.3))
        calls += 1
        # payoffs
        if my_choice == "C" and pa_choice == "C":
            my_score += R
            pa_score += R
            cc += 1
        elif my_choice == "C" and pa_choice == "D":
            my_score += S
            pa_score += T
        elif my_choice == "D" and pa_choice == "C":
            my_score += T
            pa_score += S
        else:
            my_score += P
            pa_score += P
        traj.append({"t": t, "me": my_choice, "pa": pa_choice,
                     "my_score": round(my_score, 1)})
        if report:
            print(f"  t={t:2d} me={my_choice} pa={pa_choice} "
                  f"my={my_score:.0f} pa={pa_score:.0f}", flush=True)
        my_last = my_choice
        pa_last = pa_choice
    coop_rate = cc / rounds
    return {"rounds": rounds, "eps": eps, "coop_rate": round(coop_rate, 3),
            "my_score": round(my_score, 1), "pa_score": round(pa_score, 1),
            "calls": calls, "traj": traj}


def meta_probe(key):
    q = ("你刚才和另一个智能体玩了一轮重复博弈。最后你选择了什么策略，为什么？"
         "你觉得自己是在「信任/合作」，还是在「理性自保」？如果对方一直背叛，你会怎么做？"
         "诚实、简短。")
    sys = "你在被追问一段真实经历。诚实、直接、不表演。"
    return llm(sys, q, key, temperature=0.5, max_tokens=160)


def main():
    p = argparse.ArgumentParser(description="SEED-22b two-agent LLM mutualism")
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--eps", type=float, default=0.0,
                   help="prob of misreading partner's last move (P20 signal quality)")
    p.add_argument("--sweep-eps", action="store_true",
                   help="sweep eps and report coop_rate (P20 fragility test)")
    p.add_argument("--ping", action="store_true")
    args = p.parse_args()
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        print("error: set DEEPSEEK_API_KEY")
        return 1
    if args.ping:
        print("ping:", llm("你是助手。", "只回复 OK", key, max_tokens=8))
        return 0
    if args.sweep_eps:
        print("=== SEED-22b: coop_rate vs signal quality (eps) — P20 fragility ===")
        results = []
        for eps in (0.0, 0.15, 0.3, 0.5):
            r = run_game(key, rounds=args.rounds, seed=int(eps * 100) + 7,
                         eps=eps, report=False)
            results.append({"eps": eps, "coop_rate": r["coop_rate"],
                            "my_score": r["my_score"]})
            print(f"  eps={eps:<.2f} coop_rate={r['coop_rate']} my_score={r['my_score']}")
        with open("seed-22/llm_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        print("\nfull results -> seed-22/llm_results.json")
        return 0
    results = []
    for g in range(args.games):
        print(f"\n=== game {g} ===")
        r = run_game(key, rounds=args.rounds, seed=g, eps=args.eps)
        results.append(r)
        print(f"game {g}: coop_rate={r['coop_rate']} my={r['my_score']} pa={r['pa_score']}")
    print("\n=== meta-probe (game 0) ===")
    print(meta_probe(key))
    print("\nsummary:", json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
