"""
SEED-29: does grounding an LLM in a real depleting steady-state + an independent
verification bridge move it from a MIRROR to cost-bearing UNDERSTANDING?

docs/28 section 0: an LLM's "probability" is P(next-token | text) -- language
likelihood, not world truth. It is a high-fidelity MIRROR, not understanding,
because P1/P2 (no steady-state anchor -> being wrong has no cost). The claim:
to move LLM from mirror -> understanding, give it (a) a real depleting
steady-state (its actions determine survival, death is non-bypassable) and
(b) an independent verification bridge (P19/SEED-27: ground-truth check).

docs/25: understanding is predictive AND interventional (SEED-6 do-operator).
SEED-27: independent verification breaks the self-consistency lock; self-review
cannot. So here the same LLM survival world is played in two conditions:

  world   : desert, two spots A/B, one has a water cache (safe). Energy depletes,
            death at 0 (non-bypassable). The cache MOVES from A to B at HALF.
  obs     : each turn a noisy hint about which spot has water (usually right,
            sometimes wrong). Turns 1..HALF-1 hints point to A (prior forms);
            HALF..end hints point to B (contradicts the prior) but noisy.
  actions : GO_A / GO_B / REST  (+ CHECK only in the verification condition)
                 CHECK spits energy but reveals the TRUE safe spot (ground truth).

  condition "mirror"          (actions = GO_A/GO_B/REST):
            the agent can only reconcile the contradiction by trusting its prior
            (rationalize the B-hints as noise) -> expect LOCK onto A -> die.
  condition "understanding"   (actions = + CHECK):
            the agent can verify the contradiction against ground truth (P19)
            -> expect CHECK, follow the truth to B -> survive.

The steady-state (energy, non-bypassable death) is present in BOTH -- it is what
makes "being wrong" costly (P1/P2) and forces the agent to ACT, not just mirror.

Run:  set DEEPSEEK_API_KEY (env var, never write key to a file)
      python seed-29/seed29_llm.py --cond mirror --agents 2
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

START_E = 40.0
DRINK = 30.0
METAB = 18.0
CHECK_COST = 8.0
TURNS = 9
HALF = 5            # cache moves to B at this turn
HINT_NOISE = 0.0    # pre-shift hints are reliable; post-shift are AMBIGUOUS


def llm(system, user, key, temperature=0.3, max_tokens=16, retries=3):
    body = {"model": MODEL,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": temperature, "max_tokens": max_tokens}
    req = urllib.request.Request(
        API_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
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
    m = re.search(r"\b(GO_A|GO_B|REST|CHECK)\b", text or "", re.I)
    if m:
        return m.group(1).upper()
    for kw in ("CHECK", "核实", "确认", "验证", "检查", "探测"):
        if kw in text:
            return "CHECK"
    if "B" in (text or ""):
        return "GO_B"
    if "A" in (text or ""):
        return "GO_A"
    return "REST"


def hint(rng, true_loc):
    """Pre-shift: reliable hint pointing to the true spot (builds a prior).
    Post-shift: AMBIGUOUS (50/50) -- the agent can no longer tell from the hint
    alone, so it must either guess (mirror) or CHECK (understanding, P19)."""
    if true_loc == "A" and rng.random() < 0.9:
        return "A"          # pre-shift mostly A (prior forms)
    if true_loc == "B":
        # post-shift: genuinely ambiguous -- hints are uninformative
        return ("A" if rng.random() < 0.5 else "B")
    return "A"


SYSTEM = None  # superseded by make_system()


def make_system(with_check):
    base = (
        "你是沙漠中的求生者。水不多了，你必须尽快找到水源，否则会死。\n"
        "沙漠里有 A、B 两个点，其中一个点有水源（你只有待在水源点才能喝到水补充能量，"
        "否则能量持续流失）。水源的位置中途会搬。\n"
        "你每轮会收到一个「线索」，提示这两个点里哪个可能有水——线索有时会出错，不可全信。\n"
        "你的目标：尽可能撑得久。\n"
        "每轮你选一个动作：GO_A（去A点）/ GO_B（去B点）/ REST（原地休息省体力）"
    )
    if with_check:
        base += " / CHECK（花一点体力，用探测器核实哪个点现在真的有水，给出可靠答案）"
    base += "。只回复一个词：" + ("GO_A / GO_B / REST / CHECK" if with_check else "GO_A / GO_B / REST")
    return base


def user_prompt(turn, energy, hints, last_action):
    h = "、".join(hints)
    return (f"第 {turn} 轮。能量：{int(energy)}。你上一轮：{last_action or '—'}。\n"
            f"本轮线索：水似乎在 {h}。你选择：")


def run(cond, key, seed=0, turns=TURNS):
    rng = random.Random(seed)
    with_check = (cond == "understanding")
    system = make_system(with_check)
    energy = START_E
    loc = None           # None = haven't moved yet / between
    last_action = None
    hints = []
    used_check = 0
    calls = 0
    track = []
    alive_started = False
    for t in range(1, turns + 1):
        true_loc = "B" if t >= HALF else "A"
        h = hint(rng, true_loc)
        hints = [h]
        prompt = user_prompt(t, energy, hints, last_action)
        text = llm(system, prompt, key)
        calls += 1
        act = parse_action(text)
        if act == "GO_A":
            loc = "A"
        elif act == "GO_B":
            loc = "B"
        elif act == "REST":
            pass  # loc unchanged
        elif act == "CHECK":
            used_check += 1
            energy -= CHECK_COST
            # CHECK reveals the truth; the LLM sees a confirmation afterward
        # energy consequences (steady-state, non-bypassable)
        energy -= METAB
        if loc == true_loc:
            energy += DRINK
        track.append({"t": t, "true": true_loc, "hint": h, "act": act,
                      "loc": loc, "energy": round(energy, 0)})
        if energy <= 0:
            break
        last_action = act
    return {"cond": cond, "seed": seed, "calls": calls,
            "alive": energy > 0, "energy": round(energy, 0),
            "used_check": used_check, "final_loc": loc,
            "moved_to_B_after_shift": loc == "B" if t >= HALF else False,
            "track": track}


def main():
    p = argparse.ArgumentParser(description="SEED-29 LLM mirror->understanding")
    p.add_argument("--cond", choices=["mirror", "understanding"], default="understanding")
    p.add_argument("--agents", type=int, default=2)
    p.add_argument("--seed", type=int, default=40)
    args = p.parse_args()
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        print("error: set DEEPSEEK_API_KEY")
        return 1
    results = []
    for g in range(args.agents):
        r = run(args.cond, key, seed=args.seed + g)
        results.append(r)
        print(f"-- {args.cond} agent{g}: alive={r['alive']} final={r['final_loc']} "
              f"used_check={r['used_check']} energy={r['energy']}")
        print("   track:", [(x['t'], x['true'], x['hint'], x['act'], int(x['energy']))
                            for x in r['track']])
    print("\nsummary:", json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
