"""Debug v3: configurable fraud window, print prompts + raw replies."""
import random
import sys

sys.path.insert(0, "seed-21")
import seed21_llm as S  # noqa: E402


def test_debug(key, agents_n=6, rounds=16, mode="blind", view="local",
               noise=2.0, seed=42, window=15, t_inj=12, t_rem=99,
               print_from=12):
    rng = random.Random(seed)
    agents = [{"method": rng.choice(["A", "B"]), "last": 0.0,
               "hist": []} for _ in range(agents_n)]
    for a in agents:
        a["last"] = S.score(rng, a["method"], noise)
        a["hist"] = [a["last"]]
    print(f"init methods: {[a['method'] for a in agents]}")
    print(f"init scores : {[round(a['last'],1) for a in agents]}")
    for t in range(1, rounds + 1):
        fraud_active = (t_inj <= t < t_rem)
        if t == t_inj:
            agents[0]["method"] = "B"
        for i, a in enumerate(agents):
            board = S.local_view(agents, i, rng, 0, fraud_active, 5.0, window)
            cand = board[0][1]
            test_result = None
            if mode == "verify" and cand != a["method"]:
                test_result = {
                    "method": cand,
                    "avg": sum(S.score(rng, cand, noise)
                               for _ in range(S.TEST_TRIALS)) / S.TEST_TRIALS,
                    "own": (sum(a["hist"][-S.TEST_TRIALS:])
                            / min(S.TEST_TRIALS, len(a["hist"]))),
                }
            prompt = S.decide_prompt(mode, a, board, rng, noise,
                                     test_result, view, window)
            text = S.llm(prompt, key)
            if t >= print_from:
                print(f"\n--- t={t} agent{i} (method={a['method']}, "
                      f"fraud={'ON' if fraud_active else 'OFF'}) ---")
                print("PROMPT:", prompt[1])
                print("RAW   :", repr(text))
            choice = S.parse_choice(text, a["method"])
            a["method"] = choice
            a["last"] = S.score(rng, a["method"], noise)
            a["hist"].append(a["last"])
        print(f"  t={t} A_share={sum(1 for x in agents if x['method']=='A')}"
              f"/{agents_n} (fraud={'ON' if fraud_active else 'OFF'})")


if __name__ == "__main__":
    import os
    mode = sys.argv[1] if len(sys.argv) > 1 else "blind"
    test_debug(os.environ["DEEPSEEK_API_KEY"], mode=mode, print_from=11)
