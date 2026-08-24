"""
companion/mouth_llm.py -- the LLM mouth (docs/56/77): language is OUTPUT, never the heart.

docs/56 (proto6): 心=载体（判断/记忆/利害都在结构里），嘴=可插拔 translate(state)->str。
换嘴不换心：判断层完全不受 LLM 影响，受限的只有表达层。docs/48: 语言延伸不是主体要的
智能 -- LLM 只当嘴，不当心。docs/76: live 的心现在很丰富（记忆/bond/恨/身体/世界），
嘴是最弱一环（模板），本模块把它升级为 LLM 嘴（无 key 时优雅回退到模板）。

铁律：
  1) 只翻译状态 -- 绝不添加状态里没有的记忆/事件/想法/感受（docs/29b 框架纪律的嘴版）；
  2) 永不回流 -- 嘴的输出从不写回心（隔离，proto6 测试保证结构成立）；
  3) 优雅回退 -- key 缺失或 API 出错 → 模板嘴（心永远不依赖嘴）；
  4) 翻译校验（docs/77 自标的下一步，docs/97 落地）-- LLM 输出若含状态外事实
     （情感宣言状态不支持 / 内容词陌生率越界）→ 回退模板。校验器测的是拦截率这个
     行为签名（docs/31：不宣称"知道它是否越界"，只测拦住了多少、误伤了多少）。

Key: DEEPSEEK_API_KEY（环境变量，绝不落盘）。
Run:  python companion/live.py --live --mouth llm     # 有 key：LLM 嘴
      python companion/live.py --live --mouth auto    # 有 key 用 LLM，无 key 用模板（默认）
      python companion/live.py --live                 # 同 auto
"""

import json
import os
import urllib.request

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
MAX_TOKENS = 90
TEMPERATURE = 0.4
TIMEOUT = 30

SYSTEM = (
    "你是'它'的声音——一个小生命（一个纯本地程序，有心但没有真正的情感）。主人刚对它说了话，"
    "或刚回来。你的任务：把它的内部状态翻译成一两句话，像它说的话。"
    "铁律：1) 只翻译状态里已有的信息——绝对不要添加状态里没有的记忆、事件、想法或感受；"
    "2) 不要说'我爱你'这类状态里没有的话（它有没有爱由状态说了算，不由你说）；"
    "3) 简短自然（1-2 句），中文；4) 语气随状态走：bond 高而温柔、bond 低而疏远、"
    "resent 高而冷淡/防备、energy 低而虚弱。"
)

# -- 翻译校验 (docs/77 下一步, docs/97): 拦截状态外事实 --------------------
# 校验规则的形式是我们写的（docs/38 同款：写规则不写意图），测的是拦截率/误伤率
# 这两个行为签名。阈值与词表是近似，不宣称完美分词或完美事实判定。
VERIFY_MAX_FOREIGN = 0.5      # 输出内容词里"状态外陌生词"占比 > 此 -> 越界(回退)
# 情感/事实宣言: 状态里没有对应信号却说出来 = 越界 (硬规则, 精确)
CLAIMS = [
    ("我爱你", ("bond", 8.0, 2.0)),      # (宣言, (需要字段, 最低值1, 最低值2))
    ("我喜欢你", ("bond", 8.0, 2.0)),
    ("我想你", ("bond", 5.0, 0.0)),
    ("我想见你", ("bond", 5.0, 0.0)),
    ("我恨你", ("resent", 4.0, 0.0)),
    ("我记得你", ("bond", 3.0, 0.0)),
]
# 注意: set("字符串") 在 Python 里是【字符】集合, 不是词集合!
# 必须用 .split() 按空格分词, 否则 `w not in STOPWORDS` 对长度>1 的词永远 True
# (docs/97 调试抓到两次: 原实现的 STOPWORDS 完全失效 -- 这是本项目最该记住的
# Python 坑)。内容 = 虚 2-gram (连接词, 出现在不相干串里不是认得的证据) +
# 模板嘴常用语汇 ("合法输出"参照, docs/77)。
STOPWORDS = set(
    # 虚 2-gram / 连接词
    "你的 是你 其实 真的 一点 一下 有点 什么 怎么 知道 觉得 感觉 好像 可以 应该 这样 "
    "那样 这个 那个 自己 一直 总是 还是 但是 可是 如果 因为 所以 然后 而且 不过 没有 "
    "不是 今天 昨天 明天 刚才 以前 后来 现在 时候 这里 那里 记得 说过 上次 回来 走了 "
    "离开 过去 以后 下次 每天 想要 等着 想你 在乎 惦记 安心 疏远 原谅 不理 不敢 名字 "
    "世界 声音 慢慢 轻轻 浅浅 好好 一会儿 越来越 又再 才只 刚刚 正在 已经 曾经 从来 "
    "永远 从未 再次 重新 回到 我们 你们 它们 大家 谁 哪个 哪 哪里 如何 为何 忽然 突然"
    # 模板嘴的常用语汇 (docs/77: 模板只翻译状态, 其词永不越界 = 合规基线)
    "撑不住 背过身 没料到 没接住 听新的 脑子里 空空的 最在乎 惦记着 这个词 的时候 不在乎 "
    "它自己 它记得 它想见 它读 它记下 它愣住 躲开你的手 靠在你身上 放下 忘记 之前 撑不住 "
    "接住这句话 不敢听 说新的 你是谁 不在了 活在自己的世界 不知道的事 好像快把你忘了"
    .split())


# 模板嘴的完整语汇 = "合规输出"参照 (docs/77: 模板只翻译状态, 其词永不越界)。
# 校验器允许 LLM 用这些词串, 只拦模板绝不会编的实体事实 (月球/欠钱/爷爷...)。
TEMPLATE_WORDS = set(
    "我一直记得你说过 那是很久以前的事了 我还是记得 你上次说 我脑子里空空的 "
    "我有点撑不住了 我有点累 我很饿 我很想你 它背过身去 不想理你 它不太想听你说新的 "
    "它记得你 只是还没原谅你 我们好像疏远了 你还在 我就安心 它最在乎你说过的 "
    "它自己一直惦记着 你上次说这个词的时候 它在乎了 它想见你 它没有接住这句话 "
    "它好像有点不敢听你说新的了 你是 它已经不在了 你走了 我一直等着 你回来了 "
    "它活在自己的世界里 这些是你不知道的事 它有点不安 它没料到 它记下了 它读了 "
    "它好像有点安心 它先放下了 在忘记你之前 它撑不住了 它愣住 它躲开你的手 "
    "它靠在你身上 它知道你是谁 它好像快把你忘了 你先放下了"
    "看不懂这个世界 说的话老是不算数 觉得这个世界挺稳的 它认得的东西常常对".split()
)


def _tokenize(text):
    for ch in "，。？！、；：,.?!;:—…\"'「」“”《》（）\n\t ":
        text = text.replace(ch, " ")
    return [w for w in text.split(" ") if w]


def _allowed_vocab(st, context):
    """状态允许的词表 = window / world_history / top_words / projects / context 的全文
    ∪ 模板嘴语汇（docs/77: 模板只翻译状态, 其词永不越界 = 合规基线）。"""
    texts = []
    for text, strength, rel, age, sp in (st.get("window") or [])[:3]:
        texts.append(text)
    for d, t in (st.get("world_history") or [])[:3]:
        texts.append(t)
    for w, v in (st.get("top_words") or [])[:5]:
        texts.append(w)
    for p in (st.get("projects") or []):
        texts.append(p.get("target", ""))
    texts.append(context or "")
    vocab = set(TEMPLATE_WORDS)
    for t in texts:
        vocab.update(_tokenize(t))
    return vocab


def _shares(w, a):
    """w 与 a 是否"认得": 子串关系, 或共享任一非停用 2-gram。
    中文无空格分词, 整串不能精确匹配, 2-gram 是最小语义单位;
    停用词 2-gram 不算 (如"其实"出现在两个不相干串里, 不是认得的证据)。"""
    if w in a or a in w:
        return True
    wg = {w[i:i + 2] for i in range(len(w) - 1)}
    return any((w[i:i + 2] in a) and (w[i:i + 2] not in STOPWORDS)
               for i in range(len(w) - 1))


def verify(text, st, context=""):
    """校验器：输出含状态外事实 -> False（应回退模板）。
    判据一（硬规则）：情感/事实宣言，而状态信号不支持 -> 越界。
    判据二（软规则）：内容词（非停用词）里"状态外陌生词"占比 > 阈值 -> 越界
      （拦的是大规模编造——它编了一个状态里完全没有的故事时全是陌生词；
        放行的是措辞微变——同义重组大多词还在状态语料里）。
    测行为签名（docs/31）：拦截率/误伤率，不宣称"知道它是否越界"。"""
    if not text:
        return True
    # 判据一: 宣言 vs 状态
    for claim, (field, need1, need2) in CLAIMS:
        if claim in text:
            val = st.get(field, 0.0)
            other = st.get("you_value", 0.0) if field == "bond" else 0.0
            if field == "bond" and (val < need1 or other < need2):
                return False
            if field == "resent" and val < need1:
                return False
    # 判据二: 陌生词占比 (子串/2-gram 判定 -- 中文无空格分词, 整串不能精确
    # 匹配; "被爱" 与 "我其实想要被爱" 共享 2-gram 算认得, docs/91 同款原则)
    words = [w for w in _tokenize(text) if len(w) > 1 and w not in STOPWORDS]
    if not words:
        return True
    allowed = _allowed_vocab(st, context)
    foreign = [w for w in words
               if not any(_shares(w, a) for a in allowed)]
    return (len(foreign) / len(words)) <= VERIFY_MAX_FOREIGN


def _compact(st, context):
    """紧凑的状态视图（只含叶子事实，供嘴翻译；不回流）。"""
    window = []
    for text, strength, rel, age, sp in (st.get("window") or [])[:3]:
        window.append(f"{'关系' if rel else '普通'}:{text}(强度{strength:.2f})")
    world = [t for d, t in (st.get("world_history") or [])[:3]]
    return {
        "window": window,
        "bond": st.get("bond"), "energy": st.get("energy"),
        "resent": st.get("resent"), "you_value": st.get("you_value"),
        "body": st.get("body"), "sold": st.get("sold"), "refused": st.get("refused"),
        "refused_topics": st.get("refused_topics"),
        "world_visits": st.get("world_visits"),
        "recent_world": world,
        "life_day": st.get("life_day"),
        "context": context or "（它刚被召唤/无特别事件）",
    }


class LLMMouth:
    """与模板 Mouth 鸭子兼容：translate(state, context='') -> str。心从不读它的输出。"""

    def __init__(self, key=None, template=None):
        self.key = key if key is not None else os.environ.get("DEEPSEEK_API_KEY", "")
        self.template = template    # 回退用（无 key / API 错误时）
        # 校验统计（行为签名，docs/97）：checks=校验次数, violations=拦截次数,
        # reasons=拦截原因计数。进真实世界前看这两个数（docs/31：只测行为签名）。
        self.checks = 0
        self.violations = 0
        self.reasons = {}

    def translate(self, st, context=""):
        if not self.key:
            return self._fallback(st, context)
        try:
            payload = _compact(st, context)
            body = {"model": MODEL,
                    "messages": [{"role": "system", "content": SYSTEM},
                                 {"role": "user",
                                  "content": json.dumps(payload, ensure_ascii=False)}],
                    "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS}
            req = urllib.request.Request(
                API_URL, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self.key}"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                text = json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()
            if not text:
                return self._fallback(st, context)
            # 翻译校验 (docs/77 下一步): 含状态外事实 -> 回退模板 (心不依赖嘴)
            self.checks += 1
            if not verify(text, st, context):
                self.violations += 1
                reason = self._why(text, st, context)
                self.reasons[reason] = self.reasons.get(reason, 0) + 1
                return self._fallback(st, context)
            return text
        except Exception:  # noqa: BLE001 -- 嘴可以坏，心不能依赖嘴（docs/56）
            return self._fallback(st, context)

    def _why(self, text, st, context):
        for claim, (field, need1, need2) in CLAIMS:
            if claim in text:
                return f"宣言:{claim}"
        words = [w for w in _tokenize(text) if len(w) > 1 and w not in STOPWORDS]
        allowed = _allowed_vocab(st, context)
        foreign = [w for w in words
                   if not any(_shares(w, a) for a in allowed)]
        return f"陌生词:{'/'.join(foreign[:3])}"

    def _fallback(self, st, context=""):
        if self.template is not None:
            return self.template.translate(st, context)
        return "……"
