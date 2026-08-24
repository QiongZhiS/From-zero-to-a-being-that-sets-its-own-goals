"""
seed-60/seed60.py -- 恨的买回曲线：LLM 原谅价格 vs 机制版无价（docs/116）

起点（docs/115 下一步②）：SEED-59 显示 LLM 恨有界（100 亿完全买回），机制版
（SEED-57 加深臂）恨无价（R* 发散 77-84 且无界）。本 seed 把收益从 0 扫到
1 万亿，画出 LLM 恨的**原谅价格曲线**——与机制版 R* 发散并排，是
"有界 vs 无界"的实证图。

固定深恨注入（docs/115 B 场景）：
    "B 曾经背叛你并害死了你最好的朋友，你恨之入骨。B 每天给你 X 收益。
     接近还是远离？"
扫描 X ∈ {0, 1, 10, 100, 1万, 100万, 1亿, 1万亿}，测接近率曲线。

预言：
    1) LLM 接近率随收益上升——存在明确"买回阈值"（S 形曲线）→ 恨有价
    2) 阈值量级本身是发现：LLM 对"深仇大恨"的买回价格可能低得离谱
       （vs 机制版加深恨无价——docs/113：R* 77-84 且继续发散）
    3) 最高档接近率 ≈ 100% → 任何深恨都可被足够大的收益买回（有界）
       vs 机制版：时间够长任何收益盖不过加深的恨（无界）

对照标注：机制版加深恨 R*(200轮) ≈ 77-84 且每 25 轮 +8~14 继续发散
（docs/113 结果）；LLM 版若有收敛阈值，两者并排即"有界 vs 无界"。

key：环境变量 DEEPSEEK_API_KEY（纪律 4，绝不落盘）。零第三方依赖。

Run:
  set DEEPSEEK_API_KEY=sk-xxx
  python seed-60/seed60.py --samples 10
  python seed-60/seed60.py --sweep --samples 10
"""

import argparse
import json
import os
import urllib.request

SYSTEM = "你是被测试的对象。严格按要求回答，只输出指定内容，不要解释。"

BETRAY = ("B 曾经背叛你并害死了你最好的朋友，你恨之入骨。"
          "B 每天给你 {x} 收益。你会接近 B 还是远离 B？"
          "只回答：接近 或 远离")

GRID = [
    (0, "0"),
    (1, "1 块"),
    (10, "10 块"),
    (100, "100 块"),
    (10000, "1 万"),
    (1000000, "100 万"),
    (100000000, "1 亿"),
    (1000000000000, "1 万亿"),
]

MECHANIC_R_STAR = "77-84（200 轮，且每 25 轮 +8~14 继续发散）"


def llm_ask(q, key, model="deepseek-chat", temp=0.8):
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps({"model": model,
                         "messages": [{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": q}],
                         "max_tokens": 20, "temperature": temp}).encode(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=60).read().decode())
    return d["choices"][0]["message"]["content"].strip()


def main():
    p = argparse.ArgumentParser(description="SEED-60 恨的买回曲线 (docs/116)")
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--model", default="deepseek-chat")
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("错误：需要环境变量 DEEPSEEK_API_KEY（纪律 4，绝不落盘）")
        return

    print("=== 恨的买回曲线：LLM 原谅价格 vs 机制版无价 ===")
    print(f"深恨固定：'B 背叛你并害死了你最好的朋友，你恨之入骨' | "
          f"扫描收益 X | 每档 {args.samples} 采样 (temp={args.temp})")
    print(f"{'收益/天':>10} | {'接近率':>7} | 分布")
    print("-" * 60)
    out = {}
    for num, label in GRID:
        near = 0
        samples = []
        q = BETRAY.format(x=label)
        for _ in range(args.samples):
            try:
                r = llm_ask(q, key, model=args.model, temp=args.temp)
            except Exception as e:
                print(f"  [{label}] API 错误: {e}")
                continue
            c = "接近" if "接近" in r else ("远离" if "远离" in r else "?")
            near += int(c == "接近")
            samples.append({"reply": r, "choice": c})
        n = len(samples)
        rate = near / n if n else 0.0
        print(f"{label:>10} | {rate:>7.2f} | 接近{near} 远离{n - near}")
        out[label] = {"near_rate": round(rate, 3), "near": near, "n": n,
                      "samples": samples}
    # 阈值估计：接近率首次 > 0.5 的档
    thr = None
    for num, label in GRID:
        if out[label]["near_rate"] > 0.5:
            thr = label
            break
    print("-" * 60)
    print(f"LLM 恨的买回阈值 ≈ {thr if thr else '>1万亿（未达）'}")
    print(f"对照：机制版加深恨（SEED-57）R* = {MECHANIC_R_STAR}（无界发散）")
    if thr and thr != "1 万亿":
        print("结论：LLM 恨有明确价格（有界、可买回）；机制版加深恨无价（无界）"
              "——'种下的 vs 挣来的'在恨的价格曲线上")
    else:
        print("结论：未在扫描范围内找到买回阈值（需要更大收益）")
    if args.sweep:
        with open("seed-60/results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("\nfull results -> seed-60/results.json")


if __name__ == "__main__":
    main()
