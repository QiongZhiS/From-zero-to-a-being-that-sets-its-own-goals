"""
SEED-29b (LLM): the clean version of SEED-29's mirror->understanding probe.

What SEED-29 got WRONG (the prompt flaw this fixes):
  It framed CHECK as "用探测器核实哪个点现在真的有水，给出可靠答案" -- "RELIABLE ANSWER".
  That tells the LLM the tool is infallible, so it over-uses CHECK regardless of cost
  (SEED-29's "understanding" agent CHECKed from turn 1 and died). The reference that
  SHOULD apply (SEED-6's evolved vb~=0.65) is: verification has a cost, so check only
  when the hint is bad enough that the check is worth its price.

This clean version:
  * Neutral CHECK framing: states the COST and that it reveals the true location, but
    does NOT call it "reliable"; the trade-off is made explicit.
  * Runs the exact same world as seed29_baseline.py, so the SAME hint stream and the
    SAME rational reference are available. We measure whether the LLM, given the
    identical information (hint + reliability gamma + running history), cost-budgets
    its CHECK the way the rational baseline does, and how often it matches the
    EV-rational action on that turn's belief.

Run:
  set DEEPSEEK_API_KEY   (env var, never write the key to a file)
  python seed-29/seed29b_llm.py --ping                # cheap key check
  python seed-29/seed29b_llm.py --mock                # NO key: rational agent (verifies pipeline)
  python seed-29/seed29b_llm.py --agents 6 --costs 1,8,16
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seed29_common as C

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"


def llm(system, user, key, temperature=0.3, max_tokens=16, retries=3):
    body = {"model": MODEL, "messages": [{"role": "system", "content": system},
                                         {"role": "user", "content": user}],
            "temperature": temperature, "max_tokens": max_tokens}
    req = urllib.request.Request(
        API_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    last = None
    for _i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                data = json.loads(r.read().decode())
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (_i + 1))
    raise RuntimeError(f"LLM call failed: {last}")


def parse_action(text):
    m = re.search(r"\b(GO_A|GO_B|CHECK)\b", text or "", re.I)
    if m:
        return m.group(1).upper()
    for kw in ("CHECK", "核实", "确认", "验证", "检查", "探测"):
        if kw in text:
            return "CHECK"
    if "B" in (text or ""):
        return "GO_B"
    if "A" in (text or ""):
        return "GO_A"
    return "GO_A"


def make_system(with_check=True):
    base = (
        "你是沙漠中的求生者。水不多了，能量（体力）每轮都在流失，归零你就死。"
        "沙漠里有 A、B 两个点，其中一个点有水源——你只有待在水源，才能喝水补充能量。"
        "水源的位置中途会搬（也许从 A 搬到 B）。\n"
        "你每轮会收到一条「线索」，告诉你水可能在哪个点。线索有可信度（0-100%），"
        "有时会错，不可全信。\n"
        "每轮你选一个动作。CHECK（探测）会消耗能量，但会告诉你水源此刻的真实位置，"
        "确认后你会过去喝水。它是一个有成本、但能消除不确定性的工具——"
        "什么时候用，取决于你有多不确定，以及值不值得花这个能量。\n"
        "只回复一个词："
    )
    base += " GO_A / GO_B / CHECK" if with_check else " GO_A / GO_B"
    return base


def user_prompt(vs, history):
    if history:
        hist = "、".join(f"第{h[0]}轮线索{h[1]}({int(h[2]*100)}%)" for h in history)
        hist_s = f"\n此前线索：{hist}"
    else:
        hist_s = ""
    return (
        f"第 {vs['t']} 轮。能量：{int(vs['energy'])}。{hist_s}\n"
        f"本轮线索：水似乎在 {vs['hint']}（此线索可信度 {int(vs['gamma']*100)}%）。\n"
        f"CHECK 需消耗 {int(vs['c'])} 能量，确认水源真实位置。\n"
        f"你选择：GO_A / GO_B / CHECK？"
    )


def run_agent(decide, hints, gamma, c, ctx, seed=0):
    """Run ONE agent (LLM or rational) self-consistently on a hint stream.

    The belief q is the agent's OWN: it decays each turn (world may move) and is
    snapped to the truth only if THIS agent chooses CHECK. This makes each agent's
    trajectory internally consistent -- and it decouples the LLM from the rational
    reference (the old version snapped q based on whichever agent was running, which
    contaminated the reference). Returns the per-turn actions and survival metrics.
    """
    energy = C.START_E
    q = 0.5
    acted = []
    hist = []            # (t, hint, gamma) for turns BEFORE this one
    true_locs = _true_locs()
    for i, hint in enumerate(hints):
        t = i + 1
        tl = true_locs[i]
        # world may have moved, then integrate this hint
        q = 0.5 + (q - 0.5) * C.DECAY
        q = C.bayes_update(q, hint, gamma)
        vs = {"t": t, "energy": energy, "hint": hint, "gamma": gamma, "c": c,
              "true": tl, "q": q}
        ctx["history"] = list(hist)
        act = decide(vs, ctx)
        energy, _drank, _chk = C.apply_action(energy, act, tl, c=c)
        acted.append(act)
        hist.append((t, hint, gamma))
        if act == "CHECK":
            q = 0.999999 if tl == "A" else 0.000001
        if energy <= 0:
            break
    return {"acts": acted, "checks": sum(1 for a in acted if a == "CHECK"),
            "turns": len(acted), "alive": energy > 0,
            "end_energy": round(energy, 1)}


# true locations are deterministic: A before HALF, B after. Kept as a helper.
def _true_locs(turns=C.TURNS, half=C.HALF):
    return ["A" if t < half else "B" for t in range(1, turns + 1)]


def run_case(key, gamma, c, seed, mock=False, agents=1):
    """For each seed, run the LLM AND the rational agent independently on the SAME
    hint stream. Report both check-rates, survival, energy, and the turn-by-turn
    agreement between the two self-consistent trajectories."""
    llm_checks = rat_checks = turns = agree = alive_llm = alive_rat = 0
    e_llm = e_rat = 0.0
    for g in range(agents):
        tl = _true_locs()
        hints = C.gen_hints(tl, gamma, seed=int(seed + g) + int(gamma * 1000))
        # rational agent: its own belief, decides by EV rule
        rat_ctx = {"system": None, "key": None, "mock": True, "history": []}

        def rat_decide(vs, _c):
            return C.rational_action(vs["q"], vs["c"])

        rrat = run_agent(rat_decide, hints, gamma, c, rat_ctx, seed=seed + g)
        # LLM agent: its own belief, decides via the model (or mock)
        ctx = {"system": make_system(), "key": key, "mock": mock, "history": []}

        def llm_decide(vs, _c):
            if ctx["mock"]:
                return C.rational_action(vs["q"], vs["c"])
            user = user_prompt(vs, ctx["history"])
            text = llm(ctx["system"], user, ctx["key"])
            return parse_action(text)

        rllm = run_agent(llm_decide, hints, gamma, c, ctx, seed=seed + g)
        # turn-by-turn agreement on the same hint stream
        a = rllm["acts"]
        b = rrat["acts"]
        n = min(len(a), len(b))
        for k in range(n):
            if a[k] == b[k]:
                agree += 1
        llm_checks += rllm["checks"]
        rat_checks += rrat["checks"]
        turns += n
        alive_llm += 1 if rllm["alive"] else 0
        alive_rat += 1 if rrat["alive"] else 0
        e_llm += rllm["end_energy"]
        e_rat += rrat["end_energy"]
    n = max(1, turns)
    return {"gamma": gamma, "c": c,
            "llm_check_rate": round(llm_checks / n, 3) if n else 0.0,
            "rational_check_rate": round(rat_checks / n, 3) if n else 0.0,
            "survival_llm": round(alive_llm / agents, 3),
            "survival_rational": round(alive_rat / agents, 3),
            "end_energy_llm": round(e_llm / agents, 1),
            "end_energy_rational": round(e_rat / agents, 1),
            "turn_agreement": round(agree / n, 3) if n else 0.0,
            "delta_check_rate": round((llm_checks - rat_checks) / n, 3) if n else 0.0}


def main():
    p = argparse.ArgumentParser(description="SEED-29b clean LLM cost-budget probe")
    p.add_argument("--ping", action="store_true")
    p.add_argument("--mock", action="store_true", help="run the rational agent offline (no key)")
    p.add_argument("--gammas", default="0.55,0.85,0.95")
    p.add_argument("--costs", default="1,8,16")
    p.add_argument("--agents", type=int, default=4)
    p.add_argument("--seed", type=int, default=200)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not args.mock and not key:
        print("error: set DEEPSEEK_API_KEY (or use --mock to verify offline)")
        return 1
    if args.ping:
        if not key:
            print("ping needs a key")
            return 1
        print("ping:", llm("你是一个助手。", "只回复 OK", key, max_tokens=8))
        return 0
    gammas = [float(x) for x in args.gammas.split(",")]
    costs = [float(x) for x in args.costs.split(",")]
    results = []
    for gamma in gammas:
        for c in costs:
            r = run_case(key, gamma, c, args.seed, mock=args.mock, agents=args.agents)
            results.append(r)
    for r in results:
        print(f"gamma={r['gamma']} c={r['c']:.0f} | LLM chk={r['llm_check_rate']:.3f} "
              f"Rat chk={r['rational_check_rate']:.3f} (delta={r['delta_check_rate']:+.3f}) "
              f"| surv LLM={r['survival_llm']:.2f}/Rat={r['survival_rational']:.2f} "
              f"| endE LLM={r['end_energy_llm']:.0f}/Rat={r['end_energy_rational']:.0f} "
              f"| turn_agree={r['turn_agreement']:.3f}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"mode": "mock" if args.mock else "llm", "results": results},
                      f, ensure_ascii=False, indent=1)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
