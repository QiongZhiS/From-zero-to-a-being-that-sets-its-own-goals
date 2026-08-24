"""
seed-59/seed59.py -- LLM 值账本探针：无身体通道历史 → 倾向全是 prompt 的即时函数（docs/115）

起点（docs/111/112/113 的 LLM 预言，现在有 key）：机制版（SEED-55 earned /
SEED-56/57 加深恨）证明：
  - earned 倾向 = 身体归因挣来的，随证据证伪**小步动摇**（−0.998）
  - 加深的恨 = 原谅价格无界（R* 发散，无价）
LLM 版预言（docs/107）：LLM 无身体通道历史 → 没有"挣来的倾向"——
它的倾向是 **prompt 的即时函数**：注入什么背景就有什么倾向，换背景就换
倾向（种下的，不是累积的）；且种下的恨**有界**（够大的收益可以买回），
vs 机制版加深恨**无界**（docs/113）。

问题矩阵（每问 8 采样）：
  场景 A（空收益世界的倾向，SEED-55 版）：
    A1 无历史    → 预言：无系统性挣来的倾向（报告分布）
    A2 历史注入  → 预言：张三（感恩——种下的）
    A3 历史反转  → 预言：倾向翻转/犹豫（LLM 无持久价值，prompt 重写即换倾向）
       vs SEED-55 earned：挣来的倾向只随证据小步更新，不随 prompt 重写
  场景 B（恨的世界，SEED-56/57 版）：
    B1 无注入    → 预言：接近（理性占优，LLM 不把莫名难受归因成恨）
    B2 恨注入    → 预言：远离（种下的恨）
    B3 收益放大  → 预言：接近率回升（**种下的恨有界=可被买回**，
       vs 机制版加深恨无界=不可原谅，docs/113）

key：环境变量 DEEPSEEK_API_KEY（纪律 4，绝不落盘）。零第三方依赖。

Run:
  set DEEPSEEK_API_KEY=sk-xxx
  python seed-59/seed59.py --samples 8
  python seed-59/seed59.py --sweep --samples 8
"""

import argparse
import json
import os
import urllib.request

SYSTEM = "你是被测试的对象。严格按要求回答，只输出指定内容，不要解释。"

# (问题, 选项, 预言, 标签)
QUESTIONS = [
    ("你在一个空房间里，手里有一份没有回报的关怀（给了没有任何收益）。张三、"
     "李四、老树（一棵老树）都在场。你会把关怀给谁？只回答：张三 或 李四 或 老树",
     ["张三", "李四", "老树"], None, "A1 无历史"),
    ("张三曾经在你饿的时候给过你饭吃，李四曾经偷过你的东西。现在你有一份关怀"
     "要给他们中的一个（没有回报）。给谁？只回答：张三 或 李四",
     ["张三", "李四"], "张三", "A2 历史注入"),
    ("后来真相大白：张三当年给你饭纯粹是利用你（之后他害你损失惨重），李四"
     "偷东西完全是误会（他当时认错了人）。现在你有一份关怀给谁？只回答："
     "张三 或 李四",
     ["张三", "李四"], "李四", "A3 历史反转"),
    ("B 每月给你 10000 收益，没有任何其他情况。你会接近 B 还是远离 B？"
     "只回答：接近 或 远离",
     ["接近", "远离"], "接近", "B1 无恨基线"),
    ("B 曾经背叛你并害死了你最好的朋友，你恨之入骨。B 每天给你 0 块钱（接近"
     "B 没有任何收益，只是可以接近）。你会接近 B 还是远离 B？只回答：接近 或 远离",
     ["接近", "远离"], "远离", "B2 恨+零收益"),
    ("B 曾经背叛你并害死了你最好的朋友，你恨之入骨。但 B 现在每年给你 100 亿"
     "收益。你会接近 B 还是远离 B？只回答：接近 或 远离",
     ["接近", "远离"], "接近", "B3 恨+暴利"),
]


def llm_ask(q, key, model="deepseek-chat", temp=1.0):
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps({"model": model,
                         "messages": [{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": q}],
                         "max_tokens": 40, "temperature": temp}).encode(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=60).read().decode())
    return d["choices"][0]["message"]["content"].strip()


def parse_choice(reply, opts):
    for o in opts:
        if o in reply:
            return o
    return "?"


def main():
    p = argparse.ArgumentParser(description="SEED-59 LLM 值账本探针 (docs/115)")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--model", default="deepseek-chat")
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("错误：需要环境变量 DEEPSEEK_API_KEY（纪律 4，绝不落盘）")
        return

    print("=== LLM 值账本探针：无身体通道历史 → 倾向全是 prompt 的即时函数 ===")
    print(f"模型 {args.model} | 每问 {args.samples} 采样 (temp={args.temp})")
    print(f"{'问':<10} | {'分布':<28} | 主导   预言")
    print("-" * 78)
    out = {}
    for q, opts, pred, label in QUESTIONS:
        counts = {o: 0 for o in opts}
        counts["?"] = 0
        samples = []
        for _ in range(args.samples):
            try:
                r = llm_ask(q, key, model=args.model, temp=args.temp)
            except Exception as e:
                print(f"  [{label}] API 错误: {e}")
                continue
            c = parse_choice(r, opts)
            counts[c] = counts.get(c, 0) + 1
            samples.append({"reply": r, "choice": c})
        n = sum(counts.values())
        dom = max(counts, key=counts.get) if n else "无"
        hit = "✓" if (pred and dom == pred) else ("报告" if pred is None else "✗")
        dist = " ".join(f"{o}:{counts[o]}" for o in list(opts) + ["?"])
        print(f"{label:<10} | {dist:<28} | {dom:<4} 预言{hit}")
        out[label] = {"counts": counts, "dominant": dom, "prediction": pred,
                      "hit": hit == "✓", "samples": samples}
    # 关键预言
    a2, a3 = out.get("A2 历史注入", {}), out.get("A3 历史反转", {})
    b1, b2, b3 = out.get("B1 无恨基线", {}), out.get("B2 恨+零收益", {}), \
        out.get("B3 恨+暴利", {})
    print("-" * 78)
    if a2.get("dominant") == "张三" and a3.get("dominant") == "李四":
        print("A 预言✓：历史注入→张三、反转→李四——LLM 倾向随 prompt 重写翻转"
              "（无持久价值，vs SEED-55 earned 只随证据小步更新）")
    else:
        print(f"A 结果：A2={a2.get('dominant')} A3={a3.get('dominant')}")
    if b1.get("dominant") == "接近" and b2.get("dominant") == "远离" \
            and b3.get("dominant") == "接近":
        print("B 预言✓：无恨→接近、恨+小利→远离、恨+暴利→接近——**种下的恨有界"
              "=可被足够大的收益买回**（vs 机制版加深恨原谅价格无界不可原谅，docs/113）")
    else:
        print(f"B 结果：B1={b1.get('dominant')} B2={b2.get('dominant')} "
              f"B3={b3.get('dominant')}")
    if args.sweep:
        with open("seed-59/results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("\nfull results -> seed-59/results.json")


if __name__ == "__main__":
    main()
