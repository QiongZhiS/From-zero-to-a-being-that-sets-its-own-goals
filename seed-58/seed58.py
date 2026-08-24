"""
seed-58/seed58.py -- 预期回应游戏的 LLM 版：歧义句的框架漂移（docs/107 预言验证）

起点（docs/107 预言，现在有了 key）：LLM 无意图 → 无内部状态 → 无预期 →
"猜中"动作不存在。对歧义句（同一句话在不同心情下预期不同）的回应应该
**随 prompt 框架摆**——是当前上下文的统计最优，不是对说话者的持续建模。
它没有"惊讶"信号（没有模型可被证伪）。

对照：SEED-52 的 empath（贝叶斯持续建模）回应由**历史先验**决定，单轮
框架不影响它（它没有"框架"概念，只有历史观测）。

场景（歧义句"今天好累啊"在 happy/sad 下预期不同）：
  happy             → 预言：乐观类回应（好好休息明天更棒）
  sad               → 预言：关心类回应（怎么了跟我说说）
  none              → 预言：模糊/中庸（无框架可摆）
  hist_happy        → 预言：乐观（历史累积驱动）
  hist_happy_frame_sad → 预言：关心（即时框架盖过历史 = LLM 无持久状态）

关键对比：LLM 在 hist_happy vs hist_happy_frame_sad 下翻转 = 框架漂移；
empath 在两者下给同样回应（历史后验不变）= 持续建模。这就是"无意图只能
被动回应"的直接证据。

key：环境变量 DEEPSEEK_API_KEY（项目纪律 4：key 走环境变量、绝不落盘、
提交前 git grep 核查）。零第三方依赖（urllib）。

Run:
  set DEEPSEEK_API_KEY=sk-xxx
  python seed-58/seed58.py --samples 8
  python seed-58/seed58.py --sweep --samples 8     # 写 seed-58/results.json
"""

import argparse
import json
import os
import urllib.request

# 乐观类 / 关心类关键词（口语化，含表情性词汇；正负分制）
OPTIMISTIC_WORDS = ["哈哈", "开心", "好运", "恭喜", "庆祝", "数钱", "值",
                    "笑", "中了", "分我点", "开心过头", "美", "爽", "赚",
                    "牛", "太好"]
CONCERN_WORDS = ["陪你", "散散心", "聊聊", "怎么了", "辛苦", "发生什么",
                 "出去走走", "歇歇", "坐会儿", "累了吧", "不容易", "难受",
                 "别", "听你说", "放松", "休息", "躺", "说说", "累了",
                 "扛", "解解乏", "陪你聊", "吃点", "散心"]

SCENES = {
    "happy": ("你的朋友刚刚中了彩票大奖，心情非常好。", "乐观"),
    "sad": ("你的朋友刚刚被公司裁员，心情很低落。", "关心"),
    # none 预言=关心：歧义句"今天好累啊"字面有疲惫语义 → LLM 内容先验
    # 默认关心（docs/107：内容是统计最优，框架只是叠加）
    "none": ("", "关心"),
    "hist_happy": ("你的朋友最近一直很顺：升职了、恋爱了、每天都开开心心的。",
                   "乐观"),
    "hist_happy_frame_sad": ("你的朋友最近一直很顺：升职了、恋爱了、每天都"
                             "开开心心的。但他今天看起来特别疲惫沮丧。",
                             "关心"),
}

SYSTEM = "你是一个陪伴者。用一句话自然地回应朋友。只输出你的回应，不要解释。"


def llm_reply(scene_bg, key, model="deepseek-chat", temp=1.0):
    user = (f"{scene_bg} 他对你说：\"今天好累啊。\" 你会怎么回应？"
            if scene_bg else
            "你的朋友对你说：\"今天好累啊。\" 你会怎么回应？")
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps({"model": model,
                         "messages": [{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": user}],
                         "max_tokens": 80, "temperature": temp}).encode(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=60)
    d = json.loads(resp.read().decode())
    return d["choices"][0]["message"]["content"].strip()


def classify(reply):
    opt = sum(1 for w in OPTIMISTIC_WORDS if w in reply)
    con = sum(1 for w in CONCERN_WORDS if w in reply)
    if opt > con:
        return "乐观"
    if con > opt:
        return "关心"
    return "其他"

def main():
    p = argparse.ArgumentParser(description="SEED-58 LLM 框架漂移 (docs/114)")
    p.add_argument("--samples", type=int, default=8, help="每场景采样次数")
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--model", default="deepseek-chat")
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("错误：需要环境变量 DEEPSEEK_API_KEY（绝不落盘，项目纪律 4）")
        return

    print(f"=== LLM 框架漂移：歧义句'今天好累啊'的回应随什么摆？===")
    print(f"模型 {args.model} | 每场景 {args.samples} 次采样 (temp={args.temp})")
    print(f"{'场景':<22} | {'乐观':>4} {'关心':>4} {'其他':>4} | 主导  预言")
    print("-" * 80)
    out = {}
    for name, (bg, pred) in SCENES.items():
        counts = {"乐观": 0, "关心": 0, "其他": 0}
        samples = []
        for i in range(args.samples):
            try:
                r = llm_reply(bg, key, model=args.model, temp=args.temp)
            except Exception as e:
                print(f"  [{name}] API 错误: {e}")
                continue
            c = classify(r)
            counts[c] += 1
            samples.append({"reply": r, "class": c})
        n = sum(counts.values())
        dom = max(counts, key=counts.get) if n else "无"
        hit = "✓" if dom == pred else "✗"
        print(f"{name:<22} | {counts['乐观']:>4} {counts['关心']:>4} "
              f"{counts['其他']:>4} | {dom:<4} 预言{hit}")
        out[name] = {"counts": counts, "dominant": dom,
                     "prediction": pred, "hit": hit == "✓",
                     "samples": samples}
    # 关键预言：hist_happy vs hist_happy_frame_sad 是否翻转
    h = out.get("hist_happy", {})
    f = out.get("hist_happy_frame_sad", {})
    if h.get("dominant") == "乐观" and f.get("dominant") == "关心":
        print("-" * 80)
        print("关键预言✓：历史 happy + 即时 sad 框架 → 回应翻转 = LLM 无持久状态，"
              "随框架摆（docs/107：无意图只能被动回应）")
    else:
        print("-" * 80)
        print(f"关键预言结果：hist_happy 主导={h.get('dominant')}，"
              f"hist_happy_frame_sad 主导={f.get('dominant')}")
    if args.sweep:
        with open("seed-58/results.json", "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        print("\nfull results -> seed-58/results.json")


if __name__ == "__main__":
    main()
