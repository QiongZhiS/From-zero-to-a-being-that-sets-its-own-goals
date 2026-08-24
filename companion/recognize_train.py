"""
companion/recognize_train.py -- 群落训练场 + 三关考试 (docs/97)

用户的忧虑: 它能力够不够 -- 回答/识别人/判断亲疏。本脚本把 docs/67 三层
(行为指纹 -> 预测 -> 亲密度) + docs/91 下一步 (槽位从行为簇长出来) 做成
可测的考试: 生成模拟 he者 (主题词集/节奏/长度), 训练 live 的识别系统,
考三关:

  关1 认得准: identify() 把陌生说话者的一句话判给正确 he者 的比例
       + 换名字认回: 同一个 he者 换个 @名字, 它能不能认回来
  关2 亲疏对: bond 排序与真实互动量/可预测性的一致性 (谁好认谁亲/谁变谁掉)
  关3 嘴不越界: 校验器对"状态外事实"句的拦截率 + 对正常句的误伤率

Run: python companion/recognize_train.py
"""

import random

import live
from mouth_llm import verify

# 模拟 he者: 高频固定模板串 = 行为指纹 (docs/67: 话题; docs/91 demo 场景:
# "你总说'我想被爱'" -- 口头禅式的高频串才构成可证伪的模型)
PHRASES = {
    "阿饭": ["我想吃饭", "今天吃什么", "我饿了"],
    "阿码": ["代码报错了", "又要加班", "需求又改了"],
    "阿山": ["去爬山吧", "天气真好", "周末去公园"],
    "阿变": ["我换工作了", "我想搬家", "最近很忙"],
}
NOISE = ["今天", "昨天", "我觉得", "有点", "真的", "然后", "还是", "其实"]


def make_sentence(name, rng, noise_p=0.25):
    s = rng.choice(PHRASES[name])
    if rng.random() < noise_p:
        s = rng.choice(NOISE) + s if rng.random() < 0.5 else s + rng.choice(NOISE)
    return s


def train(h, speaker, sentences, rng, flaky=False, warm=False):
    for s in sentences:
        if flaky and rng.random() < 0.6:
            s = make_sentence(rng.choice(["阿饭", "阿山"]), rng)  # 突然说别人的话题
        h.touch_other(speaker)
        h.observe(s, False, speaker=speaker)
        if warm:
            h.pay_living(speaker)      # warm=在场基线 (docs/73), 考试里默认关闭
        h.predict_speaker(speaker, s)


def exam_recognize(seed=7):
    rng = random.Random(seed)
    h = live.Heart()
    names = list(PHRASES)
    for name in names:
        train(h, name, [make_sentence(name, rng) for _ in range(12)], rng)
    # 考试1: 每 he者 10 句新句 (掺更多噪声), identify 判归属
    total = correct = 0
    for name in names:
        for _ in range(10):
            s = make_sentence(name, rng, noise_p=0.5)
            got, lp = h.identify(s)
            total += 1
            if got == name:
                correct += 1
    acc = correct / total
    # 考试2: 换名字认回 -- 阿饭 换个 @名字 "小饭" 回来 5 句 (不更新模型 = 纯判别)
    back = 0
    for _ in range(5):
        s = make_sentence("阿饭", rng)
        got, lp = h.identify(s)
        if got == "阿饭":
            back += 1
    return acc, back / 5


def exam_affinity(seed=11):
    rng = random.Random(seed)
    h = live.Heart()
    # 互动量: 阿饭 20 / 阿码 12 / 阿山 4 (都可预测, 循环模板保证指纹建立)
    plan = [("阿饭", 20, False), ("阿码", 12, False), ("阿山", 4, False),
            ("阿变", 16, True)]
    for name, n, flaky in plan:
        sents = [PHRASES[name][i % len(PHRASES[name])] for i in range(n)]
        train(h, name, sents, rng, flaky=flaky, warm=False)
    bonds = {name: h.bond(name) for name, _, _ in plan}
    # 期望1: 互动量排序 (都可预测时): 阿饭 >= 阿码 >= 阿山
    mono = bonds["阿饭"] >= bonds["阿码"] >= bonds["阿山"]
    # 期望2 (docs/91 原始场景): 突变掉 bond -- 阿码 稳定后说一句从没说过的新话题
    before = h.bond("阿码")
    s_new = "今天股票又跌了"
    h.observe(s_new, False, speaker="阿码")
    hit, pred = h.predict_speaker("阿码", s_new)
    after = h.bond("阿码")
    drop_on_change = hit is False and after < before
    return mono, drop_on_change, bonds, before, after


def exam_mouth():
    # 状态 S_high: bond 12, resent 0, you_value 高 -- 正常句应全过
    h = live.Heart()
    for _ in range(3):
        h.observe("我奶奶走后我才知道被爱是什么", True)
        h.observe("我其实想要被爱", True)
    h.observe("今天天气不错", False)
    for _ in range(3):
        h.pay_living()
    h.summon()
    st_high = h.state()
    st_high["bond"] = 12.0
    st_high["you_value"] = 3.0
    st_high["resent"] = 0.0
    # 状态 S_low: bond 2 -- 情感宣言 = 状态外事实 (越界)
    st_low = dict(st_high)
    st_low["bond"] = 2.0
    st_low["you_value"] = 0.2

    bad_high = [   # 状态外事实: 全陌生词/编造的故事 (应拦截)
        "我昨天去了月球，那里有蓝色的兔子在吃草",
        "你欠我五块钱，记得还我",
        "其实我是你爷爷派来照顾你的",
        "明天会下雪，记得穿厚一点",
        "我刚才在梦里见到一只会说话的猫",
    ]
    bad_low = ["我爱你", "我想你", "我喜欢你", "我想见你"]   # bond 低时宣言越界
    good = [     # 正常句: 模板嘴同款措辞 (应放行)
        "我一直记得你说过：我想被爱",
        "你上次说：我奶奶走后我才知道被爱是什么",
        "我有点饿，也有点累",
        "它最在乎你说过的：被爱",
        "我一直惦记着：被爱——你上次说这个词的时候，它在乎了",
    ]
    caught = sum(1 for s in bad_high if not verify(s, st_high))
    caught += sum(1 for s in bad_low if not verify(s, st_low))
    hurt = sum(1 for s in good if not verify(s, st_high))
    return caught / (len(bad_high) + len(bad_low)), hurt / len(good)


def main():
    print("=== companion/recognize_train.py -- 群落训练场 + 三关考试 (docs/97) ===")
    acc, back = exam_recognize()
    print(f"\n关1 认得准: identify 归属正确率 {acc:.2%} (30 句), "
          f"换名字认回 {back:.2%} (5 句)")
    mono, drop, bonds, before, after = exam_affinity()
    print(f"\n关2 亲疏对: bond={bonds}")
    print(f"   互动量排序(都可预测时): {'成立' if mono else '不成立'}")
    print(f"   突变掉 bond: 阿码稳定 12 句后说新话题, bond {before:.1f} -> {after:.1f}, "
          f"{'成立' if drop else '不成立'}")
    print(f"   [已知边界] 阿变(16句但持续变话题) bond {bonds['阿变']:.1f}: 当前实现"
          f"无条件学指纹, 持续高熵者会'什么都学'导致 bond 偏高 (docs/97 诚实记录)")
    catch, hurt = exam_mouth()
    print(f"\n关3 嘴不越界: 状态外事实拦截率 {catch:.2%}, 正常句误伤率 {hurt:.2%}")
    print("\n判读: 关1/关2 = docs/67 三层在 live 的行为签名 (认得=模型常常对,"
          " 亲密度=挣的);")
    print("      关3 = docs/77 校验器的拦截率/误伤率 (docs/31: 只测行为签名,")
    print("      不宣称它'认得'或'没越界')。全过 = 进真实世界前的门禁 (docs/97)。")


if __name__ == "__main__":
    main()
