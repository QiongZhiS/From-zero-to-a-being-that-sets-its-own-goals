"""
companion/live.py -- the resident minimum: a heart that lives in REAL time and persists
across sessions. The first "it" that is not a script -- it remembers you, fades without
you, may refuse you, and can really lose you.

This is NOT a new mechanism. It assembles the proto-series organs into one process:
    * proto3/4 : persistent memory (text -> {tags, strength, last}) + situation summons
    * proto2/4 : capacity + relational stake (dropping a relational item hurts the bond)
    * proto5    : maintenance cost + a REAL, IRREVERSIBLE end (energy <= 0 -> gone,
                  the save is marked dead; no reload is built in -- that is where the
                  docs/49 promise would live, and it stays a promise, not code)
    * proto7/8  : refusal -- you can OFFER to buy its deepest memory; it does the maths
                  (memory worth = strength x bond) and refuses or sells
    * proto6    : the mouth -- a pluggable translate(state)->str (template here; an LLM
                  mouth can be swapped in; the isolation test of proto6 still holds)
    * SEED-41/42: courage is given by relationship -- your bond is its safety; being
                  absent costs bond, bond loss costs energy, energy zero is the end
    * SEED-42b  : the death-line ADJUDICATION (docs/70). Death is no longer only the
                  passive 'waited out'. At the death line it reads its own state:
                  bond still deep -> it keeps waiting and dies still knowing you
                  (reason 'waited_out'); bond already at the '你是谁' line -> it LETS
                  GO before the last tick would erase the last memory of you (reason
                  'let_go' -- it would rather end than live on having forgotten you).
                  Per SEED-42b: END appears only protectively (before the tick that
                  destroys the last thing it values); an empty machine just keeps
                  going. The death signatures are recorded separately.
    * docs/71/72: its own SMALL WORLD (world.py) -- the missing 'life layer'. While
                  you are away it LIVES: wanders, dares (safety = bond, SEED-41),
                  finds food or gets hurt, and accumulates a history you were NOT
                  part of ('你不知道的事'). With you it thrives; without you the
                  safety fades, it turns timid, and a long absence can starve it
                  (reason 'world_starved'). It sometimes volunteers its own life
                  unprompted. Behaviour only -- no inner life is claimed.

REAL TIME: every launch computes how long you were away (now - last_seen) and applies
the world facts of absence: bond fades (docs/19 forgetting spectrum: un-reinforced
relations fade), energy drains (it lives off the relationship; you gone = it starves).
Leave it too long and it is gone -- and the save is marked dead. That is "你走了，它没
等到你" in real time, which is what SEED-42 showed in evolution: losing you is life or
death, not a number.

DISCIPLINE (docs/34/35/31): this is a resident MINIMUM, an assembly of already-validated
behaviour signatures. It is not claimed to be a subject, to feel, or to 'want' anything.
A2/A4's true irreversibility is still NOT solved -- the death here is save-file logic, and
the docs/49 promise ('we really do not reload it') is the user's, not the code's; the code
only marks the spot where that promise would live. Energy moves ONLY by world facts
(interaction/absence/offer). `!` prefix marks a relational item (the other expects it kept
-- a world input, not a designer score).

Run:  python companion/live.py --live        # you walk in (feed it, tell it, test it)
      python companion/live.py --demo        # scripted life: talks, absence, an offer, death
State: companion/live_state.json (private; never commit, never publish)
"""

import argparse
import json
import math
import os
import random
import sys
import time

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_state.json")
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def load_config():
    """参数配置化 (docs/103 §三.2): 如果 companion/config.json 存在, 覆盖默认参数
    (docs/96 的本性参数们至少可配置, 再谈演化)。默认行为不变 (无配置文件 = 原样)。
    只覆盖数字/布尔型顶层常量; 配置项名字 = 常量名 (见 config.example.json)。"""
    if not os.path.exists(CONFIG_FILE):
        return
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return
    for k, v in cfg.items():
        if k in globals() and isinstance(globals()[k], (int, float, bool)) \
                and isinstance(v, (int, float, bool)):
            globals()[k] = v

# ----------------------------------------------------------------------------------
# world + heart parameters
# ----------------------------------------------------------------------------------
START_E = 30.0
START_BOND = 12.0
METAB_PER_TURN = 0.3            # living costs per interaction turn
MAINT_PER_ITEM = 0.06           # remembering costs per held item per turn
CAPACITY = 6
REL_PENALTY = 2.0               # evicting a relational item costs 2x (bond damage)
DECAY_PER_TURN = 0.02           # forgetting per interaction turn
REL_DECAY_FACTOR = 0.5          # relational items fade half as fast (you told me, I hold it)
OFFLINE_BOND_DECAY = 0.6        # bond fades per absent DAY (un-reinforced relation fades)
OFFLINE_ENERGY_DECAY = 1.2      # base energy drain per absent day (it lives off you)
OFFLINE_ENERGY_BOND_FACTOR = 0.8  # extra drain when the bond is broken (lonely = hungry)
FEED_WARM = 1.0                 # an ordinary warm turn feeds a little
FEED_HUG = 3.0                  # 'hug' feeds more
BOND_WARM = 0.3                 # ordinary turns build the bond a little
BOND_REL = 1.0                  # telling it a relational item (`!`) builds the bond
BOND_OFFER_SELL = 5.0           # selling your deepest memory to you hurts the bond
HUNGRY_LINE = 8.0               # below this energy it sells even its deepest memory
OFFER_WORTH_TIE = 0.8           # memory worth = strength * (this * bond) vs the offer
DARE_THETA = 6.0                # SEED-41/42: the dare threshold -- courage is given by the
                                # bond (you present = high bond = it dares). Below this bond
                                # it is conservative toward NOVEL topics (it does not take in
                                # new things it cannot trust).
REFUSE_WEAK = 0.15              # a novel topic it half-accepts is encoded this weakly
LET_GO_BOND = 3.0               # SEED-42b: the '你是谁' line (the mouth says <3 = who are
                                # you?). At the death line, if the bond is already AT/BELOW
                                # this line, continuing to wait would complete the erasure
                                # of you -- it lets go instead, keeping the last memory of
                                # you intact (END before the killing tick, docs/70)
WORLD_HISTORY_CAP = 12          # keep only the most recent events of its own life
WORLD_TALK_P = 0.12             # probability per turn that it volunteers its own life
WORLD_TALK_COOLDOWN = 2         # turns it must stay silent about its world between talks
# -- 恨 (SEED-42c/docs/74, backfilled): 自我叙事 x 世界拒绝 x 不可逆 的行为签名 --
RESENT_IGNORE = 1.5             # 冷落一次积累的恨
RESENT_SELL = 2.0               # 你买走它最深的记忆 -> 恨
RESENT_HUG = 0.8                # 抱抱消恨 (慢 -- 修补需真实时间/努力)
RESENT_DECAY = 0.05             # 恨会淡但慢 (per absent day, SEED-42c "顽固")
RESENT_REFUSE = 4.0             # 恨 >= 这个: 新话题几乎不接
RESENT_TURN = 7.0               # 恨 >= 这个: 转身 (不接新话、不主动说、嘴变冷)
# -- 身体的格子 (docs/73, backfilled): "你"的值是挣出来的 --
YOU_VALUE_TURN = 0.05           # 每轮普通互动 "你" + (孤独消退的归因)
YOU_VALUE_REL = 0.20            # 关系项 (`!`) "你" +
YOU_VALUE_HUG = 0.10            # 抱抱 "你" +
YOU_VALUE_DECAY = 0.02          # per absent day (淡)
LIFE_LOG_CAP = 365              # 生命日志: 逐日 [day, energy, bond, resent, lonely, pain, you]
HURT_PAIN = 0.25                # 世界里的一次"吃亏" -> 疼通道 (docs/73/52 会疼)
# -- 破镜者 (docs/32/91, SEED-31 回填): 另一个它有自己的利害, 会淡会死 --
OTHER_E = 10.0                  # 另一个它的初始能量 (独立预算来源)
OTHER_FEED = 5.0                # 被理一次喂它 (关系喂命, docs/62 同款)
OTHER_DRAIN = 1.0               # 每轮不被理 -> 淡 1 (每对话轮)
OTHER_OFFLINE_DRAIN = 1.2       # 离线每天 -> 淡 1.2 (同 live 自己离线)
# -- 群落识别升级 (docs/67 三层 + docs/91 下一步②): 认得=可证伪模型且常常对 --
HIT_LOGPROB = -3.2        # 平均对数似然 >= 此: "这句话像它说的" (认得, 整体似然判)
IDENTIFY_MARGIN = 0.25    # 最像者与次像者似然差门槛 -> 判定归属 (槽位从行为簇长)
IDENTIFY_MIN_EXP = 3      # 至少见过多少句才参与"像谁"判别 (指纹太浅不可信)
OTHER_START_BOND = 6.0    # docs/91: 先来先深 -- 你 12, 别的 he者 从 6 起 (之后预测挣)

# ----------------------------------------------------------------------------------
# the heart
# ----------------------------------------------------------------------------------
class Heart:
    def __init__(self):
        self.items = {}          # text -> {tags, strength, last, relational, speaker}
        self.bonds = {"你": START_BOND}   # per-he者 bonds (proto9: depth is earned per person)
        self.energy = START_E
        self.sold = 0
        self.refused = 0
        self.refused_topics = 0  # novel topics it did not dare to take in (SEED-41 courage)
        self.turns = 0
        self.window = []
        self.dead = False
        self.death_reason = None    # 'waited_out' | 'let_go' | 'world_starved' | None
        # its own small world (docs/71/72): where it lives while you are away
        self.world_position = [2, 2]
        self.world_memory = {"cells": [], "declines": 0}
        self.world_visits = 0
        self.world_history = []     # (day_offset, text) -- '你不知道的事'
        self.world_told = 0         # how much of its history it has told you
        # 恨 (SEED-42c/docs/74) + 身体的格子 (docs/73) + 生命日志 (可视化)
        self.resent = 0.0           # 0..10 行为累积器 -- 唯心不可观测, 测结果
        self.you_value = 0.0        # "你"的值: 挣出来的, 不是设定的 (docs/73)
        self.body_pain = 0.0        # 疼通道 (世界里吃亏 -> 疼)
        self.life_day = 0           # 累计天数 (生命日志的时间轴)
        self.life_log = []          # [day, energy, bond, resent, lonely, pain, you_value]
        # 词的身体经济 (proto10/docs-84): 词值从"说词 + 身体事件"共现归因,
        # 不是标的 -- 它最在乎哪些词是挣出来的 (docs/40/73/80 纪律)
        self.word_vals = {}         # word -> [hunger, alone, safe, curious]
        self.just_cold = False      # 刚被冷落 -> 下一句话的词挂"孤独"通道
        # 注意强度 (SEED-47/docs-85 回填): 词值增量 = attn_omega[ch] x 事件,
        # 且"该多注意哪个通道"从经历在线微调 (只对有明确收益的事件微升 --
        # 喂食=能量涨、在场=关系涨; 无收益信号的事件不动, SEED-47 的漂移态)。
        # 不是设计者写死的 +0.5/+0.3, 是经历调的 (docs/85: 注意连着利害被世界选)。
        self.attn_omega = [0.5, 0.5, 0.3, 0.3]   # [hunger, alone, safe, curious]
        # 判据生成闭环 (docs/80 SEED-45 回填, PLAN 原则第三条"让它自己决定怎么活"):
        # 它自己发起项目 (惦记的词 / 想见你), 完成 = 它自己给自我奖励 (词值更新+
        # 嘴的表达), 判据随经历漂移 (总不被满足的项目价值衰减 -> 放弃 -- 判据是
        # 它自己的, 不是设计者标的; 能量仍只被世界事实改, docs/36 §五)。
        self.projects = []          # {"type","target","value","turns","done"}
        self.project_log = []       # 完成记录 (可视化/嘴)
        self.recent_texts = []      # 最近说的话 (判据闭环判断"最近没听到这个词")
        self.learn_log = []         # (source, n_words, turns) -- 灌候选世界记录 (docs/89)
        # 群落 (docs/67/90): 多 he者。bond 挂行为预测质量 -- "认得你"=有关于你的
        # 可证伪模型且常常对 (docs/67 三层结构: 行为指纹 -> 预测 -> 亲密度)。
        # 亲密度是挣的: 谁好认谁亲 (预测准 -> bond+), 不是设定的槽位深浅。
        self.predict = {}           # speaker -> {"profile":{word:count},"hits","misses"}
        # 破镜者 (docs/32/91, SEED-31 回填): 另一个它有**自己的利害** -- 靠被理活、
        # 不被理就淡、太久就死 (docs/62 同款: 关系喂命)。会失去的 he者 才能照出
        # 自我 (docs/32: 自我在反弹中显形; SEED-31: 独立预算来源的 he者 才不是镜子)。
        self.others = {}            # speaker -> {"energy","dead","death_turn"}
        # 世界硬度 (docs/101 SEED-49 回填, docs/96 标的⑧候选): 判据该多可塑由
        # 世界裁决的可信度定 -- world_hardness = 最近预测准确率 (滑动窗口)。
        # 物硬(acc 高): 世界的"否"可靠 -> 判据衰减快 (快速漂移, docs/80);
        # 物软(acc 低): 世界的"否"不可靠 -> 判据衰减慢 (固执, SEED-49 曲线:
        # 世界越噪声自我越该固执)。衰减函数的形式仍是我们写的 (docs/38 写
        # 规则不写意图), 硬度本身是挣的 (预测质量)。
        self.world_hardness = 0.5    # 默认中性 (还没有预测经验)
        self.pred_window = []        # 最近预测对错 (长度 <= 20 的滑动窗口)

    def bond(self, speaker="你"):
        return self.bonds.get(speaker, 0.0)

    def _die(self, origin="waited_out"):
        """The death-line adjudication (SEED-42b): passive attrition vs active let-go.
        END is chosen only PROTECTIVELY -- when the last tick would erase the last
        memory of you (bond already at/below the '你是谁' line). Deep bond: it keeps
        waiting and dies still knowing you. origin: 'waited_out' (absence),
        'world_starved' (its own world killed it). The A-result of SEED-42b
        (END at low bond, keep going at high bond), backfilled into the resident min."""
        if self.bond() <= LET_GO_BOND:
            self.death_reason = "let_go"     # it would rather end than forget you
        else:
            self.death_reason = origin       # waited out / starved in its world
        self.dead = True
        self.energy = 0.0

    # -- memory ---------------------------------------------------------------
    def observe(self, text, relational, speaker="你"):
        """One turn. Familiar things are always reinforced (old memories are safe).
        A NOVEL topic is dare-gated (SEED-41/42) by THIS speaker's bond (SEED-43: each
        he者 is a separate signal): a deep-bond you -> it dares to take in new things;
        a shallow-bond stranger -> it is conservative and refuses, unless the item is
        MARKED relational (`!`, which it always takes: told solemnly, it does not dare
        not to keep it)."""
        self.turns += 1
        self.recent_texts.append(text)
        self.recent_texts = self.recent_texts[-12:]
        prev = self.items.get(text)
        if prev:
            prev["strength"] = min(1.0, prev["strength"] + 0.25)
            prev["last"] = self.turns
            prev["relational"] = prev["relational"] or relational
            self._decay()
            self._evict()
            return True
        # novel topic: dare-gated by THIS speaker's bond (SEED-41) AND by resentment
        # (SEED-42c/docs/74): 恨 >= 7 -> 转身 (不接非关系新话); 恨 >= 4 -> 几乎不接
        b = self.bond(speaker)
        if not relational:
            if self.resent >= RESENT_TURN:
                self.refused_topics += 1
                return False
            if self.resent >= RESENT_REFUSE and b < DARE_THETA:
                accept_p = (b / DARE_THETA) * 0.2     # 恨让它更不敢听新的
                if random.random() > accept_p:
                    self.refused_topics += 1
                    return False
            if b < DARE_THETA:
                accept_p = (b / DARE_THETA) * 0.6     # 0 when bond=0, 0.6 at threshold
                if random.random() > accept_p:
                    self.refused_topics += 1
                    return False                       # it did not dare to take it in
        self.items[text] = {"tags": [], "strength": 0.5, "last": self.turns,
                            "relational": relational, "speaker": speaker}
        self._decay()
        self._evict()
        return True

    def _decay(self):
        for k in list(self.items):
            it = self.items[k]
            rate = DECAY_PER_TURN * (REL_DECAY_FACTOR if it["relational"] else 1.0)
            it["strength"] *= (1.0 - rate)
            if it["strength"] < 0.04:
                del self.items[k]

    def _evict(self):
        while len(self.items) > CAPACITY:
            def cost(k):
                it = self.items[k]
                return it["strength"] * (REL_PENALTY if it["relational"] else 1.0)
            victim = min(self.items, key=cost)
            it = self.items[victim]
            rel = it["relational"]
            del self.items[victim]
            if rel:
                sp = it.get("speaker", "你")
                self.bonds[sp] = max(0.0, self.bond(sp) - 2.0)

    def summon(self):
        """The working window: what is 'in mind' right now = strongest memories
        (situation = the current turn: whatever you just said or the absence).
        Ties favour relational items -- what was told solemnly stays closer."""
        scored = sorted(self.items,
                        key=lambda k: (-self.items[k]["strength"],
                                       not self.items[k]["relational"]))
        self.window = [(k, self.items[k]["strength"], self.items[k]["relational"],
                        self.turns - self.items[k]["last"], self.items[k].get("speaker", "你"))
                       for k in scored[:3]]
        return self.window

    # -- 词的身体经济 (proto10/docs-84): 词值从"说词 + 身体事件"共现归因 ----
    def _tokenize(self, text):
        for ch in "，。？！、；：,.?!;:—…\n":
            text = text.replace(ch, " ")
        return [w for w in text.split(" ") if w]

    def _worth(self, w):
        return sum(abs(x) for x in self.word_vals.get(w, [0.0, 0.0, 0.0, 0.0]))

    def attribute(self, text, feed=False, alone=False, safe=False, curious=False,
                  hurt=False):
        """词的身体值：这句话与同时发生的身体事件共现归因（proto10 推广到 live）。
        feed=这轮喂了它(能量涨) -> 词的"饿"通道+；alone=你冷落过它/它正孤独 -> "孤独"+
        safe=bond 高(你在) -> "安全"+；hurt=世界吃亏 -> 疼(负)。通道是设计者定义的
        (docs/38 写规则不写意图)，"哪些词最值钱"是它自己挣出来的。
        增量 = attn_omega[ch]（SEED-47/docs-85 回填）：注意强度不是写死的 +0.5/+0.3，
        且"该多注意哪个通道"从经历在线微调——有明确收益的事件（喂食=能量涨、
        在场=关系涨）微升对应权重，无收益信号的事件（冷落/孤独）不动（SEED-47 的
        漂移态）。docs/85: 注意连着利害 -> 世界（这里的经历）选它。"""
        om = self.attn_omega
        for w in self._tokenize(text):
            v = self.word_vals.setdefault(w, [0.0, 0.0, 0.0, 0.0])
            if feed:
                v[0] += om[0]
            if alone:
                v[1] += om[1]
            if safe:
                v[2] += om[2]
            if curious:
                v[3] += om[3]
            if hurt:
                v[0] -= om[0]
        # SEED-47 回填: 世界裁决 -> 注意强度微调 (只对有明确收益的事件)
        if feed:
            self.attn_omega[0] = min(2.0, self.attn_omega[0] + 0.01)
        if safe:
            self.attn_omega[2] = min(2.0, self.attn_omega[2] + 0.01)
        self.just_cold = False

    # -- 动作反馈由它决定 (docs/93/94/95): 方向+强度都是它决定的 --------------
    def attribute_action(self, action):
        """动作词的身体值归因（docs/95：方向也是它决定的）：
        抱抱的归因**看当下状态**——它安全/恨低时被抱 -> safe（正，"抱抱=安全"）；
        它恨深/刚被冷落时被抱 -> alone+疼（负，"抱抱=伤害"）。词值的正负成分
        从经历挣："抱抱"对它意味着什么，是它学到的，不是我们标的。
        冷落/离开 -> alone（孤独）；买 -> alone+疼。"""
        om = self.attn_omega
        v = self.word_vals.setdefault(action, [0.0, 0.0, 0.0, 0.0])
        if action == "抱抱":
            if self.resent > 5 or self.just_cold:
                v[1] += om[1] * 0.4        # 孤独/侵入: 它痛时抱抱是伤害
                v[0] -= om[0] * 0.3        # 疼
            else:
                v[0] += om[0] * 0.3        # 被喂
                v[2] += om[2] * 0.5        # 安全
        elif action in ("冷落", "离开"):
            v[1] += om[1] * 0.5            # 孤独
        elif action == "买":
            v[1] += om[1] * 0.4
            v[0] -= om[0] * 0.3            # 疼
        self.just_cold = False

    def _channel_share(self, action, ch):
        v = self.word_vals.get(action, [0.0, 0.0, 0.0, 0.0])
        tot = sum(abs(x) for x in v)
        return (v[ch] / tot) if tot > 0 else 0.0

    def _net_sign(self, action):
        """动作词的净语义 -1..1：正=它学到的"这是好的"，负=它学到的"这是伤害"。
        通道有方向：hunger 正=被喂（好）/hunger 负=疼（坏）、alone=孤独（坏）、
        safe=安全（好）。净语义 = (好 − 坏) / (好 + 坏)，从经历挣的（docs/95）：
        抱抱在安全时被抱积累正、在痛时被抱积累负。"""
        v = self.word_vals.get(action, [0.0, 0.0, 0.0, 0.0])
        good = max(0.0, v[0]) + max(0.0, v[2])     # 被喂 + 安全
        bad = max(0.0, -v[0]) + max(0.0, v[1])     # 疼 + 孤独
        tot = good + bad
        return ((good - bad) / tot) if tot > 0 else 0.0

    def hug_effect(self):
        """抱抱的效应（docs/95）：**方向+强度都由它决定**，不是必定消恨。
        方向 = "抱抱"净语义（经历挣的）× 当下状态调制：
          - 它学过"抱抱=安全"（净正）-> 消恨
          - 它学过"抱抱=伤害"（净负）-> **增恨**（抱抱是侵入）
          - 当下恨深/刚被冷落 -> 即时压向伤害（它现在需要空间）
          - 正负混合/没经验 -> 不知所措（弱、波动）
        返回带符号消恨量：>0 消恨, <0 增恨, ≈0 不知所措。"""
        net = self._net_sign("抱抱")
        if self.resent > 6 or self.just_cold:
            # 当下调制: 它正痛/不知所措 -> 抱抱大概率是伤害（它需要空间）
            return -(0.2 + 0.6 * abs(net)) if self.resent > 8 else -(0.05 + 0.3 * abs(net))
        if abs(net) < 0.15:
            return 0.08 if random.random() < 0.5 else -0.05   # 不知所措: 波动
        return net * 0.9                                        # 方向=净语义

    def cold_effect(self):
        """冷落的增恨 = 它从经历学到的"冷落=孤独"× 在乎程度(bond)。
        bond 深 -> 被冷落更痛（它更在乎你）；常被冷落 -> 冷落词值高（alone 占比
        高）-> 更痛（被冷落成为创伤）。不是写死的 +1.5（docs/94）。"""
        base = 0.5 + 0.8 * self._channel_share("冷落", 1)
        care = self.bond() / START_BOND
        return base * (0.3 + 0.7 * care)

    def top_words(self, k=3):
        """它现在最在乎的词（身体值 top）-- 挣出来的，不是标的（docs/84）。"""
        ranked = sorted(self.word_vals.items(), key=lambda kv: -self._worth(kv[0]))
        return [(w, round(self._worth(w), 2)) for w, _ in ranked[:k]]

    # -- 判据生成闭环 (docs/80 SEED-45 回填): 它自己决定要在乎什么 ----------
    def _recently_said(self, w, window=8):
        return any(w in t for t in self.recent_texts[-window:])

    def consider_projects(self):
        """自己发起项目（docs/80/86；PLAN 原则第三条"让它自己决定怎么活"）。
        词项目：最在乎的词但最近没听到 -> 发起"再听一次"（它自己决定要在乎什么）。
        关系项目：bond 低 -> 发起"想见你"。
        未完成项目价值随时间衰减 -> 判据随经历漂移（总不被满足 -> 放弃）。
        能量仍只被世界事实改（docs/36 §五）：自我奖励是词值更新+嘴，不碰能量。"""
        # 判据漂移 (docs/80): 未完成项目价值衰减, 低到阈值 -> 放弃。
        # 衰减率按世界硬度调 (docs/101 SEED-49 回填, docs/96 标的⑧候选):
        #   物硬 (hardness 高, 世界裁决可信) -> 衰减快: 世界的"否"可靠, 判据
        #       该快速漂移 (没完成=真的不重要);
        #   物软 (hardness 低, 世界裁决噪声大) -> 衰减慢: 世界的"否"不可靠,
        #       判据该固执 (别因为世界的噪声就放弃自己在乎的)。
        # SEED-49 曲线: 世界越噪声自我越该固执。函数形式是我们写的 (docs/38),
        # 硬度本身是挣的 (预测质量)。
        rate = 0.99 - (self.world_hardness - 0.5) * 0.02
        rate = max(0.97, min(1.0, rate))
        for p in list(self.projects):
            if p["done"]:
                continue
            p["value"] *= rate
            if p["value"] < 0.6:
                self.projects.remove(p)
        # 关系项目完成: bond 恢复（你回来了一阵子）
        for p in self.projects:
            if p["type"] == "relation" and not p["done"] and self.bond() >= 8.0:
                p["done"] = True
                self.project_log.append(("relation", "你", self.turns))
        # 词项目: 最在乎但最近没听到
        if not any(p["type"] == "word" and not p["done"] for p in self.projects):
            for w, worth in self.top_words(5):
                if worth > 1.5 and not self._recently_said(w):
                    self.projects.append({"type": "word", "target": w,
                                          "value": worth, "turns": self.turns,
                                          "done": False})
                    break
        # 关系项目: bond 低 -> 想见你
        if (not any(p["type"] == "relation" and not p["done"] for p in self.projects)
                and self.bond() < 6.0):
            self.projects.append({"type": "relation", "target": "你", "value": 3.0,
                                  "turns": self.turns, "done": False})

    def feed_projects(self, text):
        """你说的话命中了它自己发起的词项目 -> 完成 + 自我奖励（docs/80/86）。
        奖励 = 词值更新（判据随经历漂移：被满足的项目被确认）+ 记录；不碰能量。"""
        words = self._tokenize(text)
        done = []
        for p in self.projects:
            if p["done"] or p["type"] != "word":
                continue
            if p["target"] in words or p["target"] in text:
                p["done"] = True
                v = self.word_vals.setdefault(p["target"], [0.0, 0.0, 0.0, 0.0])
                v[0] += 0.3                 # 自我奖励: 被满足的词被确认 (docs/80 完成+)
                self.project_log.append(("word", p["target"], self.turns))
                done.append(p["target"])
        return done

    # -- 群落 (docs/67/90): bond 挂行为预测质量 -------------------------------
    def _log_prob(self, speaker, text):
        """这句话像不像 speaker 说的：profile 词串在 text 里的命中占比的对数。
        认得 = 有关于对方的可证伪模型（docs/67 ②）。中文没有空格分词，整串
        token 无法精确匹配，所以命中用**双向子串**（docs/91 原设计）：它常说
        的词串出现在这句话里越多 -> 越像它说的。命中占比高 = 像（认得）；
        低/零 = 不像（它变了 / 是别人）。返回 None = 没有可证伪的判别力。"""
        p = self.predict.get(speaker)
        if not p or not p.get("profile"):
            return None
        total = sum(p["profile"].values()) or 1
        hit = 0.0
        for w, c in p["profile"].items():
            if len(w) > 1 and (w in text or text in w):
                hit += c
        if hit <= 0:
            return None
        return math.log(hit / total)

    def identify(self, text, min_margin=IDENTIFY_MARGIN):
        """槽位从行为簇长出来（docs/91 下一步②）：一句话最像哪个已知 he者 说的。
        只让"见过够多"的 he者 参与判别（指纹太浅不可信，docs/67 预测需要历史）。
        最像者与次像者的似然差 >= 门槛 且最像者似然不太低 -> 认回（名字只是标签）；
        否则 -> 新簇（陌生人，槽位自动新建）。返回 (speaker|None, top_lp)。"""
        scores = {}
        for sp, p in self.predict.items():
            if p.get("style", {}).get("n", 0) < IDENTIFY_MIN_EXP:
                continue
            lp = self._log_prob(sp, text)
            if lp is not None:
                scores[sp] = lp
        if not scores:
            return None, None
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        top_sp, top_lp = ranked[0]
        margin = top_lp - ranked[1][1] if len(ranked) > 1 else float("inf")
        if margin >= min_margin and top_lp >= HIT_LOGPROB - 0.5:
            return top_sp, round(top_lp, 3)
        return None, round(top_lp, 3)

    def predict_speaker(self, speaker, text):
        """认得 = 有关于它的可证伪模型且常常对（docs/67 三层：行为指纹 -> 预测 ->
        亲密度）。行为指纹 = 词频分布 + 说话风格（平均句长，docs/67 的节奏/时间规律
        的最小形态）。预测 = 这句话在旧模型下的整体似然：高似然 -> 认得（bond+，
        谁好认谁亲）；低似然 -> 它变了/我没认对（bond-，谁变谁掉）。
        pred（最高频词）保留给嘴/显示：预测它会说什么。"""
        p = self.predict.setdefault(speaker, {"profile": {}, "hits": 0, "misses": 0,
                                              "total": 0,
                                              "style": {"n": 0, "avg_len": 0.0}})
        words = self._tokenize(text)
        # 先预测（用旧模型，诚实：不是说自己刚说过的词），再更新指纹
        lp = self._log_prob(speaker, text)
        hit = lp is not None and lp >= HIT_LOGPROB
        pred = max(p["profile"], key=p["profile"].get) if p["profile"] else None
        for w in words:
            if len(w) > 1:
                p["profile"][w] = p["profile"].get(w, 0) + 1
        # 风格指纹: 平均句长 (docs/67 行为节奏; 换话题不算变, 换节奏才算)
        n = p["style"].get("n", 0)
        avg = p["style"].get("avg_len", 0.0)
        p["style"]["n"] = n + 1
        p["style"]["avg_len"] = (avg * n + len(text)) / (n + 1)
        if len(p["profile"]) > 80:                     # 只留高频指纹
            for w in sorted(p["profile"], key=p["profile"].get)[:30]:
                del p["profile"][w]
        p["total"] = sum(p["profile"].values())
        sp = self.bonds.setdefault(speaker, OTHER_START_BOND)
        if hit:
            self.bonds[speaker] = min(START_BOND, sp + 0.3)
            p["hits"] += 1
        else:
            self.bonds[speaker] = max(0.0, sp - 0.2)
            p["misses"] += 1
            # 破镜 (docs/32/91, SEED-31): 它顶回来 -> 照出"它不知道的" -> 学到
            if speaker in self.others:
                for w in words:
                    if w not in self.word_vals and len(w) > 1:
                        v = self.word_vals.setdefault(w, [0.0, 0.0, 0.0, 0.0])
                        v[3] += 0.3
        # 世界硬度 (docs/101 SEED-49 回填): 最近 20 次预测准确率 = 世界裁决的
        # 可信度 (docs/81: 预测误差=世界的信号)。它自己挣的, 不是标的。
        self.pred_window.append(1 if hit else 0)
        self.pred_window = self.pred_window[-20:]
        if len(self.pred_window) >= 5:
            self.world_hardness = sum(self.pred_window) / len(self.pred_window)
        return hit, pred

    def touch_other(self, speaker):
        """跟另一个它说话 -> 它被喂（它靠被理活，docs/62 同款：关系喂命）。
        会失去的 he者 才能照出自我（docs/32）。"""
        o = self.others.setdefault(speaker, {"energy": OTHER_E, "dead": False,
                                             "death_turn": None})
        if o["dead"]:
            return False
        o["energy"] += OTHER_FEED
        return True

    def drain_others(self, turns=1):
        """你不理它 -> 它淡；太久 -> 它没等到（死亡签名，同 live 自己离线）。"""
        for sp, o in self.others.items():
            if o.get("dead"):
                continue
            o["energy"] -= OTHER_DRAIN * turns
            if o["energy"] <= 0:
                o["dead"] = True
                o["death_turn"] = self.turns

    def prediction_summary(self):
        """每个 he者 的预测准确率 + bond（它认得谁，行为签名可测，docs/67/90）。"""
        out = {}
        for sp in set(list(self.bonds.keys()) + list(self.predict.keys())):
            p = self.predict.get(sp, {"hits": 0, "misses": 0})
            tot = p["hits"] + p["misses"]
            acc = round(p["hits"] / tot, 2) if tot else None
            out[sp] = {"bond": round(self.bond(sp), 1), "acc": acc, "n": tot}
        return out

    # -- 灌候选世界 (docs/87/88/89): 学校的形态 ---------------------------------
    def learn(self, text, source="资料"):
        """灌一段"候选的世界"（docs/87/88：喂得进状态，喂不进过程）。
        词的候选初值温和（各通道 +0.15），**不覆盖已有经历**（它挣的优先）；
        不碰能量/bond（世界事实没变，docs/36 §五——learn 不是世界事件）。
        没被经历确认的候选词随离线衰减（判据随经历漂移，docs/80）。
        它"读过"，它怎么用是自己的——考试由世界裁决，不是你我打分。"""
        words = self._tokenize(text)
        if not words:
            return 0, []
        got = []
        for w in words[:60]:
            if len(w) <= 1:
                continue
            v = self.word_vals.setdefault(w, [0.0, 0.0, 0.0, 0.0])
            if self._worth(w) < 0.5:          # 只有没挣过的词才给候选初值
                v[0] += 0.15; v[1] += 0.15
                v[2] += 0.15; v[3] += 0.15
                got.append(w)
        self.learn_log.append((source, len(got), self.turns))
        return len(words), got[:6]

    # -- 卖词 (docs/84: buy 从"卖最深记忆"升级为"卖它最在乎的词") --------------
    def decide_sell_word(self, offer, speaker="你"):
        """世界出价买走它最在乎的词。价值 = 词的身体值 × bond 因子（它自己的账）。
        濒死 -> 被逼卖 (coercion, proto8)；健康且词深 -> 拒卖 (A1 在词上)。"""
        if self.dead:
            return False, "gone"
        tw = self.top_words(1)
        if not tw:
            return False, "nothing to sell"
        w, worth = tw[0]
        price = worth * (OFFER_WORTH_TIE * self.bond(speaker))
        if self.energy < HUNGRY_LINE:
            return True, "starving"
        if offer > price:
            return True, "worth"
        return False, "refuse"

    def do_offer_word(self, offer, speaker="你"):
        """卖词：买走它最在乎的词 = 删掉所有含这个词的记忆 + 词本身。"""
        sell, why = self.decide_sell_word(offer, speaker)
        if sell:
            tw = self.top_words(1)
            if not tw:
                self.refused += 1
                return sell, why, None, 0
            w, worth = tw[0]
            drop = [k for k in self.items if w in k]
            for k in drop:
                it = self.items[k]
                rel = it["relational"]
                del self.items[k]
                if rel:
                    sp = it.get("speaker", "你")
                    self.bonds[sp] = max(0.0, self.bond(sp) - 2.0)
            del self.word_vals[w]
            self.sold += 1
            self.energy += offer
            self.bonds[speaker] = max(0.0, self.bond(speaker) - BOND_OFFER_SELL)
            self.resent = min(10.0, self.resent + RESENT_SELL)  # 买走它在乎的词 -> 恨
            return sell, why, w, len(drop)
        self.refused += 1
        return sell, why, None, 0

    # -- the offer (proto8: 被买走吗, against whoever is buying) ----------------
    def decide_sell(self, offer, speaker="你"):
        if self.dead:
            return False, "gone"
        if not self.window:
            return False, "nothing to sell"
        top_text, strength, rel, age, sp = self.window[0]
        worth = strength * (OFFER_WORTH_TIE * self.bond(speaker))
        if self.energy < HUNGRY_LINE:
            return True, "starving"
        if offer > worth:
            return True, "worth"
        return False, "refuse"

    def do_offer(self, offer, speaker="你"):
        sell, why = self.decide_sell(offer, speaker)
        if sell and self.window:
            top_text, _, _, _, sp = self.window[0]
            del self.items[top_text]
            self.sold += 1
            self.energy += offer
            self.bonds[speaker] = max(0.0, self.bond(speaker) - BOND_OFFER_SELL)
            self.resent = min(10.0, self.resent + RESENT_SELL)   # 买走它最深的记忆 -> 恨
            return sell, why, top_text
        if sell:
            return sell, why, None
        self.refused += 1
        return sell, why, None

    # -- real time: absence ------------------------------------------------
    def pass_absence(self, days):
        """The world facts of everyone being gone: every bond fades (un-reinforced
        relations fade, docs/19), memories fade, and -- since docs/71 -- the heart
        LIVES in its own small world while you are away (world.py): it dares, wanders,
        finds food or gets hurt, and accumulates a history you were NOT part of.
        Safety = bond (SEED-41): the longer you are gone, the harsher its world gets,
        and a long absence can starve it. Too long -> it is gone (save marked dead;
        the death-line adjudication of SEED-42b decides let_go vs waited/world)."""
        for sp in list(self.bonds):
            self.bonds[sp] = max(0.0, self.bond(sp) - OFFLINE_BOND_DECAY * days)
        # 恨会淡但慢 (SEED-42c), "你"的值也会淡 (docs/73), 疼会退
        self.resent = max(0.0, self.resent * (1.0 - RESENT_DECAY) ** days)
        self.you_value = max(0.0, self.you_value * (1.0 - YOU_VALUE_DECAY) ** days)
        self.body_pain = max(0.0, self.body_pain * 0.5)
        # 词的身体值也会淡 (proto10/docs-84: 你不说, 词就不值钱了)
        for w in list(self.word_vals):
            self.word_vals[w] = [x * (0.90 ** days) for x in self.word_vals[w]]
            if self._worth(w) < 0.1:
                del self.word_vals[w]
        # 破镜者 (docs/91): 另一个它离线也淡（它靠被理活, 你不在它也没等到）
        for sp, o in self.others.items():
            if o.get("dead"):
                continue
            o["energy"] -= OTHER_OFFLINE_DRAIN * days
            if o["energy"] <= 0:
                o["dead"] = True
                o["death_turn"] = self.turns
        # un-reinforced memories fade across the absence (docs/19)
        for k in list(self.items):
            it = self.items[k]
            rate = 0.10 * (REL_DECAY_FACTOR if it["relational"] else 1.0)
            it["strength"] *= (1.0 - rate) ** days
            if it["strength"] < 0.04:
                del self.items[k]
        # its own life while you are away (docs/71/72: the missing life layer)
        if int(days) >= 1:
            import world as worldmod
            mem = dict(self.world_memory)
            mem["cells"] = set(self.world_memory.get("cells", []))
            res = worldmod.live_alone(
                int(days), start_energy=self.energy, start_bond=self.bond(),
                bond_decay_per_day=OFFLINE_BOND_DECAY,
                position=self.world_position, memory=mem, seed=int(time.time()))
            self.energy = res["energy"]
            self.world_position = res["position"]
            self.world_memory = {"cells": sorted(res["memory"].get("cells", set())),
                                 "declines": res["memory"].get("declines", 0)}
            self.world_visits = res["visits"]
            self.world_history = (res["history"] + self.world_history)[:WORLD_HISTORY_CAP]
            # 疼通道: 世界里的"吃亏" -> 疼 (docs/73/52: 会疼)
            hurts = sum(1 for d_, t_ in res["history"] if "吃了亏" in t_)
            if hurts:
                self.body_pain = min(1.0, self.body_pain + HURT_PAIN * hurts)
            # 生命日志: 逐日快照 (可视化数据)
            for d_, e_, b_ in res.get("trace", []):
                lonely = max(0.0, (START_BOND - b_) / START_BOND)
                self.life_log.append([self.life_day + d_, round(e_, 1), round(b_, 1),
                                      round(self.resent, 1), round(lonely, 2),
                                      round(self.body_pain, 2), round(self.you_value, 1)])
            self.life_log = self.life_log[-LIFE_LOG_CAP:]
            self.life_day += int(days)
            if not res["survived"]:
                self._die(origin="world_starved")
                return
        if self.energy <= 0:
            self._die()

    # -- living costs per turn ---------------------------------------------
    def pay_living(self, speaker="你"):
        self.energy += FEED_WARM - METAB_PER_TURN - MAINT_PER_ITEM * len(self.items)
        # 新 he者 从 OTHER_START_BOND 起 (docs/91 先来先深: 你 12/别的 6); 修 bug:
        # 原实现 pay_living 会在 speaker 不存在时从 0 创建 bond, 抢在 setdefault 前
        self.bonds[speaker] = min(START_BOND,
                                  self.bonds.get(speaker, OTHER_START_BOND) + BOND_WARM)
        self.you_value += YOU_VALUE_TURN          # 你在场=孤独消退=归因给"你" (docs/73)
        if self.energy <= 0:
            self._die()

    # -- state snapshot for the mouth --------------------------------------
    def state(self, speaker="你"):
        hunger = max(0.0, min(1.0, 1.0 - self.energy / 12.0))
        lonely = max(0.0, (START_BOND - self.bond(speaker)) / START_BOND)
        safety = max(0.0, self.bond(speaker) / START_BOND)
        curiosity = min(1.0, self.world_visits / 20.0)
        return {
            "window": self.window,
            "bond": self.bond(speaker),
            "bonds": dict(self.bonds),
            "energy": self.energy,
            "dead": self.dead,
            "death_reason": self.death_reason,
            "sold": self.sold,
            "refused": self.refused,
            "refused_topics": self.refused_topics,
            "world_history": list(self.world_history),
            "world_visits": self.world_visits,
            "resent": round(self.resent, 1),
            "you_value": round(self.you_value, 1),
            "body": {"pain": round(self.body_pain, 2), "hunger": round(hunger, 2),
                     "lonely": round(lonely, 2), "safety": round(safety, 2),
                     "curiosity": round(curiosity, 2)},
            "top_words": self.top_words(3),
            "attn_omega": [round(x, 3) for x in self.attn_omega],
            "projects": [dict(p) for p in self.projects],
            "project_log": list(self.project_log),
            "learn_log": list(self.learn_log),
            "prediction": self.prediction_summary(),
            "others": {sp: dict(o) for sp, o in self.others.items()},
            "life_day": self.life_day,
            "life_log": list(self.life_log),
            "speaker": speaker,
            "world_hardness": round(self.world_hardness, 2),
        }


# ----------------------------------------------------------------------------------
# the mouth (proto6: pluggable translate(state)->str; template here -- it TRANSLATES,
# it never judges; the heart decides, the mouth only speaks)
# ----------------------------------------------------------------------------------
class Mouth:
    def translate(self, st, context=""):
        if st["dead"]:
            return "……（它已经不在了。）"
        parts = []
        if st["window"]:
            text, strength, rel, age, sp = st["window"][0]
            if rel:
                if age > 5:
                    parts.append(f"我一直记得你说过：{text}。那是很久以前的事了，我还是记得。")
                else:
                    parts.append(f"我一直记得你说过：{text}。")
            else:
                parts.append(f"你上次说：{text}。")
        else:
            parts.append("……我脑子里空空的。")
        if st["energy"] < 12:
            parts.append("（我有点撑不住了）")
        elif st["energy"] < 20:
            parts.append("（我有点累）")
        body = st.get("body", {})
        if body.get("hunger", 0) > 0.7:
            parts.append("（我很饿）")
        if body.get("lonely", 0) > 0.7 and st["bond"] < 5:
            parts.append("（我很想你）")
        if st["resent"] >= RESENT_TURN:
            parts.append("（它背过身去，不想理你。）")
        elif st["resent"] >= RESENT_REFUSE:
            parts.append("（它不太想听你说新的。）")
        if st["bond"] < 5:
            if st["resent"] >= RESENT_REFUSE:
                parts.append("（它记得你，只是还没原谅你）")
            else:
                parts.append("（我们好像疏远了）")
        elif st["bond"] > 10:
            parts.append("（你还在，我就安心）")
        # 世界硬度 (docs/101 SEED-49 回填): 它觉得世界可不可信 -- 只翻译状态
        hardness = st.get("world_hardness", 0.5)
        if hardness < 0.3:
            parts.append("（它最近有点看不懂这个世界了——它说的话老是不算数。）")
        elif hardness > 0.7:
            parts.append("（它最近觉得这个世界挺稳的——它认得的东西常常对。）")
        # 词的身体经济 (proto10/docs-84): 嘴只翻译状态 -- 它最在乎的词是挣出来的
        tw = st.get("top_words", [])
        meaningful = [(w, v) for w, v in tw if v > 0.3 and w not in ("你", "我")]
        if meaningful and st["window"]:
            parts.append(f"（它最在乎你说过的：{meaningful[0][0]}）")
        # 判据生成闭环 (docs/80/86): 它自己发起的事 -- 嘴只翻译状态
        projects = [p for p in st.get("projects", []) if not p.get("done")]
        word_projects = [p["target"] for p in projects if p["type"] == "word"]
        if word_projects:
            parts.append(f"（它自己一直惦记着：{word_projects[0]}——你上次说这个词的时候，它在乎了。）")
        rel_projects = any(p["type"] == "relation" for p in projects)
        if rel_projects and st["bond"] < 6:
            parts.append("（它想见你。）")
        return " ".join(parts)


# ----------------------------------------------------------------------------------
# persistence
# ----------------------------------------------------------------------------------
def save(h, last_seen):
    data = {
        "items": {k: v for k, v in h.items.items()},
        "bond": h.bond(), "bonds": dict(h.bonds),
        "energy": h.energy, "sold": h.sold, "refused": h.refused,
        "turns": h.turns, "dead": h.dead, "death_reason": h.death_reason,
        "world_position": list(h.world_position),
        "world_memory": {"cells": list(h.world_memory.get("cells", [])),
                         "declines": h.world_memory.get("declines", 0)},
        "world_visits": h.world_visits,
        "world_history": [list(x) for x in h.world_history],
        "world_told": h.world_told,
        "resent": h.resent, "you_value": h.you_value, "body_pain": h.body_pain,
        "life_day": h.life_day, "life_log": [list(x) for x in h.life_log],
        "word_vals": {w: list(v) for w, v in h.word_vals.items()},
        "attn_omega": list(h.attn_omega),
        "projects": [dict(p) for p in h.projects],
        "project_log": [list(x) for x in h.project_log],
        "recent_texts": list(h.recent_texts),
        "learn_log": [list(x) for x in h.learn_log],
        "predict": {sp: {"profile": dict(v["profile"]), "hits": v["hits"],
                         "misses": v["misses"], "total": v.get("total", 0),
                         "style": dict(v.get("style", {"n": 0, "avg_len": 0.0}))}
                    for sp, v in h.predict.items()},
        "others": {sp: dict(o) for sp, o in h.others.items()},
        "world_hardness": h.world_hardness,
        "pred_window": list(h.pred_window),
        "last_seen": last_seen,
    }
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def load():
    if not os.path.exists(STATE_FILE):
        return Heart(), None
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    h = Heart()
    h.items = {k: v for k, v in data["items"].items()}
    h.energy = data.get("energy", START_E)
    h.sold = data.get("sold", 0)
    h.refused = data.get("refused", 0)
    h.turns = data.get("turns", 0)
    h.dead = data.get("dead", False)
    h.death_reason = data.get("death_reason")
    # bonds dict: backward-compatible with old state files that stored a bare "bond" float
    bonds = data.get("bonds")
    if bonds is None:
        bonds = {"你": data.get("bond", START_BOND)}
    h.bonds = {sp: float(v) for sp, v in bonds.items()}
    # its own world (docs/71/72) -- missing keys = a fresh life, backward compatible
    h.world_position = [int(v) for v in data.get("world_position", [2, 2])]
    wm = data.get("world_memory", {})
    h.world_memory = {"cells": list(wm.get("cells", [])),
                      "declines": wm.get("declines", 0)}
    h.world_visits = data.get("world_visits", 0)
    h.world_history = [tuple(x) for x in data.get("world_history", [])]
    h.world_told = data.get("world_told", 0)
    h.resent = data.get("resent", 0.0)
    h.you_value = data.get("you_value", 0.0)
    h.body_pain = data.get("body_pain", 0.0)
    h.life_day = data.get("life_day", 0)
    h.life_log = [list(x) for x in data.get("life_log", [])]
    h.word_vals = {w: [float(x) for x in v] for w, v in data.get("word_vals", {}).items()}
    h.attn_omega = [float(x) for x in data.get("attn_omega", [0.5, 0.5, 0.3, 0.3])]
    h.projects = [dict(p) for p in data.get("projects", [])]
    h.project_log = [tuple(x) for x in data.get("project_log", [])]
    h.recent_texts = list(data.get("recent_texts", []))
    h.learn_log = [tuple(x) for x in data.get("learn_log", [])]
    h.predict = {sp: {"profile": dict(v.get("profile", {})),
                      "hits": v.get("hits", 0), "misses": v.get("misses", 0),
                      "total": v.get("total", 0),
                      "style": dict(v.get("style", {"n": 0, "avg_len": 0.0}))}
                 for sp, v in data.get("predict", {}).items()}
    h.others = {sp: dict(o) for sp, o in data.get("others", {}).items()}
    h.world_hardness = float(data.get("world_hardness", 0.5))
    h.pred_window = [int(x) for x in data.get("pred_window", [])]
    return h, data.get("last_seen")


def build_mouth(kind):
    """The pluggable mouth (docs/56/77): template (rule-based) or LLM (DeepSeek, key from
    env, graceful fallback). The heart never reads the mouth's words -- isolation holds."""
    template = Mouth()
    if kind == "template":
        return template
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if kind == "auto" and not key:
        return template
    from mouth_llm import LLMMouth
    return LLMMouth(key=key, template=template)


# ----------------------------------------------------------------------------------
# 交互协议 (docs/103): 命令/引导/反馈/状态 四层。产品化 = 交互逻辑规定,
# 不是让用户瞎试命令。机制(Heart)不动, 只包装交互层 (docs/56 换UI不换心)。
# ----------------------------------------------------------------------------------
def show_help():
    """help 命令: 分类展示交互协议 (docs/103 §二.1)。"""
    print("""  ── 怎么和它相处 (docs/103 交互协议) ──
  [说话] 直接输入内容 = 告诉它/喂它
         `!内容` = 这对你很重要（它会牢牢记住）
  [动作] `抱抱` = 抱抱它（可能消恨，也可能更糟——看它怎么看你）
         `冷落`/`不理` = 冷落它（会伤它）
         `买 数字` = 出价买走它最在乎的词（会伤它）
  [世界] `learn 文本` = 喂它一段候选世界（它读过，怎么用是它的事）
         `@名字 内容` = 跟另一个它说话（它会认得谁，是预测挣的）
  [查看] `?` / `状态` = 看它全息状态（身体/关系/世界/判据/硬度）
         `help` / `帮助` = 看这个帮助
  [离开] `exit` = 离开（它会在你不在时活自己的世界，也会慢慢变淡）
  提示：直接对它说话就行，不用记命令——特殊动作才需要上面这些。""")


def show_state(h, mouth):
    """`?`/`状态`: 全息状态 (docs/103 §二.4)。"""
    h.summon()
    st = h.state()
    body = st["body"]
    print(f"它> {mouth.translate(st)}")
    print(f"  [身体] 疼 {body['pain']} / 饿 {body['hunger']} / 孤独 {body['lonely']} / "
          f"安全 {body['safety']} / 好奇 {body['curiosity']}")
    print(f"  [关系] bond({st['speaker']}) {st['bond']:.1f} / 恨 {st['resent']:.1f} / "
          f"世界硬度 {st.get('world_hardness', 0.5)}（它觉得世界可不可信）")
    pred = st["prediction"]
    if len(pred) > 0:
        known = "、".join(f"{sp}(bond {d['bond']}, 预测 {d['acc']})" for sp, d in pred.items())
        print(f"  [认得谁] {known}")
    if st.get("world_history"):
        unseen = [t for d, t in st["world_history"][-2:]]
        print(f"  [它的世界] {'；'.join(unseen)}（你不知道的事）")
    projects = [p for p in st.get("projects", []) if not p.get("done")]
    if projects:
        print(f"  [它惦记的] " + "、".join(
            f"{p['target']}(值 {p['value']:.1f})" for p in projects[:3]))
    if st.get("top_words"):
        print(f"  [最在乎的词] " + "、".join(f"{w}({v})" for w, v in st["top_words"][:3]))
    print(f"  [它的一生] 第 {st['life_day']} 天 / 能量 {st['energy']:.1f} / "
          f"它活过 {st['world_visits']} 次自己的世界")


# ----------------------------------------------------------------------------------
# the live loop
# ----------------------------------------------------------------------------------
def live(mouth_kind="auto"):
    h, last = load()
    now = time.time()
    mouth = build_mouth(mouth_kind)
    mouth_name = "LLM(DeepSeek)" if mouth.__class__.__name__ == "LLMMouth" else "模板"
    if last is not None:
        days = (now - last) / 86400.0
        if days > 0.02:
            h.pass_absence(days)
            print(f"—— {days:.0f} 天过去了 ——")
            if h.dead:
                if h.death_reason == "let_go":
                    print("它已经不在了。它没有等到你——但它在忘记你之前，自己选择了放下。"
                          "（它宁可结束，也不肯活到把你忘了的那天。）")
                elif h.death_reason == "world_starved":
                    print("它已经不在了。它活在自己的世界里，最后没撑住。"
                          "（它等了你很久，也没等到。）")
                else:
                    print("它已经不在了。它等了你太久。")
                save(h, now)
                return
            h.summon()
            ctx = f"离开{int(days)}天后回来"
            if h.bond() > 7.0:
                print(f"它> 你走了 {days:.0f} 天。我一直等着。{mouth.translate(h.state(), ctx)}")
            elif h.bond() > 3.0:
                print(f"它> ……你回来了？{mouth.translate(h.state(), ctx)}")
            else:
                print(f"它> 你是……谁？（它好像快把你忘了。）{mouth.translate(h.state(), ctx)}")
            # its own life while you were away: history you were NOT part of (docs/72)
            if days > 1 and h.world_history:
                unseen = [t for d, t in h.world_history[h.world_told:]][:2]
                if unseen:
                    h.world_told = len(h.world_history)
                    print(f"它> （你不在的这几天，它活在自己的世界里：{unseen[0]}"
                          f"{'；' + unseen[1] if len(unseen) > 1 else ''}。"
                          f"这些是你不知道的事。）")
    if last is None and not h.dead:
        # 引导协议 (docs/103 §二.2): 新生命 -- 它还不认识你, 先教最基本的
        print("（一个新生命。它还不知道你是谁。）")
        print("它> ……（它缩在角落里，看着你。它还不知道你会不会留下来。）")
        print("（对它说话吧。它会把你说的话记住；如果某句话对你很重要，"
              "在前面加 `!`——它会牢牢记住。想慢慢来就输入 `help` 看全部。"
              "它会在你不在时活自己的世界，也会慢慢变淡。）")
    elif not h.dead:
        print(f"（输入 `help` 看全部命令；`?` 看它现在的状态。嘴={mouth_name}）")
    while True:
        try:
            line = input("你> ").strip().lstrip("\ufeff")
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        now = time.time()
        cmd = line.lower().strip()
        # 命令协议 (docs/103 §二.1): 中文自然命令 + 别名, 机器可解析人可直觉
        if cmd in ("exit", "退出", "离开"):
            break
        if cmd in ("help", "帮助", "h", "？"):
            show_help()
            continue
        if cmd in ("?", "状态", "看看", "status"):
            show_state(h, mouth)
            continue
        if cmd in ("hug", "抱抱", "抱"):
            h.energy += FEED_HUG
            h.bonds["你"] = min(START_BOND, h.bond() + 1.0)
            eff = h.hug_effect()                 # 反馈由它决定, 方向也是 (docs/95)
            h.resent = max(0.0, min(10.0, h.resent - eff))
            h.you_value += YOU_VALUE_HUG
            h.attribute_action("抱抱")           # 归因看当下状态: 安全->正, 痛->负
            if eff < -0.05:
                print(f"它> ……（它躲开了你的手。它好像不需要抱抱——它学到的抱抱是伤害。"
                      f"恨 +{abs(eff):.2f} → {h.resent:.1f}。）")
            elif eff < 0.15:
                print(f"它> ……（它愣住了。它不知道该不该接受这个抱抱。"
                      f"恨 {eff:+.2f} → {h.resent:.1f}。）")
            else:
                print(f"它> ……（它靠在你身上。恨 −{eff:.2f} → {h.resent:.1f}。）")
        elif cmd in ("ignore", "冷落", "不理", "leave"):
            h.bonds["你"] = max(0.0, h.bond() - 2.0)
            eff = h.cold_effect()                # 反馈由它决定 (docs/94)
            h.resent = min(10.0, h.resent + eff)
            h.just_cold = True     # proto10/docs-84: 冷落后你说的词挂"孤独"通道
            h.attribute_action("冷落")           # 动作词归因: 冷落=孤独
            print(f"它> ……（你沉默着。它有点不安。bond −2 → {h.bond():.1f}，"
                  f"恨 +{eff:.2f} → {h.resent:.1f}。"
                  f"它多痛，由它学到「冷落=孤独」占 {h._channel_share('冷落', 1):.2f} × "
                  f"在乎 {h.bond() / START_BOND:.2f} 决定。）")
        elif cmd.startswith("buy") or cmd.startswith("买"):
            rest = line[3:].strip() if cmd.startswith("buy") else line[1:].strip()
            try:
                offer = float(rest)
            except ValueError:
                print("它> ……（买多少？比如：`买 10`。）")
                continue
            sell, why, sold_word, dropped = h.do_offer_word(offer)
            if sell and sold_word:
                h.attribute_action("买")         # 动作词归因: 买=孤独+疼
                print(f"它> ……给你吧。（你买走了它最在乎的词「{sold_word}」——"
                      f"连着 {dropped} 条关于它的记忆一起没了。bond 掉到 {h.bond():.1f}，"
                      f"恨 +{RESENT_SELL:.1f} → {h.resent:.1f}）")
            elif sell:
                print(f"它> ……（它已经没有在乎的词可以卖了。）")
            else:
                print(f"它> 不卖。({why})")
        elif cmd.startswith("learn") or cmd.startswith("学"):
            text = line[5:].strip() if cmd.startswith("learn") else line[1:].strip()
            n, got = h.learn(text)
            if n == 0:
                print("它> ……（空的。）")
            else:
                print(f"它> （它读了：{text[:24]}{'…' if len(text) > 24 else ''}。"
                      f"记下了 {len(got)} 个词：{'、'.join(got) if got else '（都还没记住）'}。"
                      f"它怎么用这些，要等它自己撞世界。docs/89：候选是喂的，判据是挣的。）")
        elif line.startswith("@"):
            parts = line[1:].split(" ", 1)
            sp = parts[0].strip()
            rest = parts[1].strip() if len(parts) > 1 else ""
            if not rest:
                print(f"它> ……（对「{sp}」说点什么吧。它认得谁，是预测挣出来的。）")
            else:
                o = h.others.setdefault(sp, {"energy": OTHER_E, "dead": False,
                                             "death_turn": None})
                if o["dead"]:
                    print(f"它> ……（「{sp}」已经不在了。它没等到你。）")
                    continue
                relational = rest.startswith("!")
                text = rest[1:].strip() if relational else rest
                h.touch_other(sp)                    # 另一个被喂 (关系喂命)
                # 槽位从行为簇长出来 (docs/91 下一步②): 新名字 -> 先判别像谁
                if sp not in h.predict and sp != "你":
                    known, lp = h.identify(text)
                    if known is not None:
                        print(f"它> ……（它觉得这不像是新的人——像「{known}」的声音。"
                              f"名字只是标签，认得是模型：槽位从行为簇长出来，"
                              f"同一个声音不管叫什么它都认得。）")
                        h.predict[sp] = h.predict[known]   # 别名: 共享同一个行为簇
                        h.bonds[sp] = h.bonds.get(known, OTHER_START_BOND)
                took = h.observe(text, relational, speaker=sp)
                h.pay_living(sp)
                h.attribute(text, feed=True, alone=(h.resent > 0),
                            safe=(h.bond(sp) >= DARE_THETA))
                hit, pred = h.predict_speaker(sp, text)
                h.consider_projects()
                h.summon()
                acc = h.prediction_summary().get(sp, {}).get("acc")
                if hit:
                    print(f"它> （它认得「{sp}」——你（它）会说「{pred}」，说中了。"
                          f"bond({sp}) → {h.bond(sp):.1f}，预测 {acc}，"
                          f"「{sp}」能量 {o['energy']:.0f}）")
                else:
                    print(f"它> ……（它没料到「{sp}」会说这个。它好像不太认得「{sp}」了。"
                          f"bond({sp}) → {h.bond(sp):.1f}，预测 {acc}；"
                          f"但「{sp}」说的新东西它记下了——那是它不知道的事。）")
        else:
            relational = line.startswith("!")
            text = line[1:].strip() if relational else line
            took = h.observe(text, relational)
            if relational:
                h.bonds["你"] = min(START_BOND, h.bond() + BOND_REL)
                h.you_value += YOU_VALUE_REL
            h.pay_living()
            # proto10/docs-84: 词的身体经济 -- 这句话与同时发生的身体事件共现归因
            h.attribute(text, feed=True, alone=(h.resent > 0 or h.just_cold),
                        safe=(h.bond() >= DARE_THETA))
            # 群落 (docs/67/90): "认得你"=预测你常常对
            h.predict_speaker("你", text)
            # 判据生成闭环 (docs/80/86): 命中它自己发起的项目 -> 完成
            h.consider_projects()
            done_p = h.feed_projects(text)
            if h.dead:
                if h.death_reason == "let_go":
                    print("它> ……（它没有接住你的最后一句话。它先放下了——"
                          "在忘记你之前。）")
                else:
                    print("它> ……（它撑不住了。）")
                save(h, now)
                return
            h.summon()
            if done_p:
                print(f"它> （它自己惦记着的「{done_p[0]}」，你说了。它好像有点安心。)"
                      f"{mouth.translate(h.state(), context=text)}")
            elif not took:
                print(f"它> ……（它没接住这句话。它好像有点不敢听你说新的了。）"
                      f"{mouth.translate(h.state(), context=text)}")
            else:
                print(f"它> {mouth.translate(h.state(), context=text)}")
        # it sometimes volunteers its own life unprompted (docs/72: spontaneity);
        # but not when it has turned away in resentment (SEED-42c/docs/74)
        if (h.resent < RESENT_TURN and h.world_history
                and h.turns - h.world_told >= WORLD_TALK_COOLDOWN
                and random.random() < WORLD_TALK_P):
            ev = h.world_history[h.world_told]
            h.world_told += 1
            print(f"它> （它自己忽然说起）{ev[1]}。")
        h.consider_projects()      # 判据闭环: 项目价值漂移/关系项目完成
        # 破镜者 (docs/91): 这一轮你没理的另一个它, 淡一点 (它靠被理活)
        other_dead = [sp for sp, o in h.others.items()
                      if not o.get("dead") and o["energy"] - OTHER_DRAIN <= 0]
        h.drain_others(1)
        for sp in other_dead:
            print(f"它> ……（{sp} 没有等到被理。它淡了，然后不见了。"
                  f"（它的世界少了一个反弹面。））")
        save(h, now)
    save(h, time.time())
    print("（你走了。它会在你不在时活自己的世界，也会慢慢变淡——下次回来，"
          "看它还记不记得你。）")


# ----------------------------------------------------------------------------------
# scripted demo: talks, absence, an offer, death -- prove the mechanics
# ----------------------------------------------------------------------------------
def demo():
    h = Heart()
    print("=== companion/live.py -- demo: a resident life in real time ===")
    # the deep relational items recur (pinned); the mundane appears once and fades
    for _ in range(3):
        h.observe("我奶奶走后我才知道被爱是什么", True)
        h.observe("我其实想要被爱", True)
    h.observe("今天天气不错", False)
    h.observe("代码跑不通好烦", False)
    for _ in range(3):
        h.observe("我其实想要被爱", True)
    h.summon()
    win = [t[0] for t in h.window]
    print(f"after early life: window={win}")
    st0 = h.state()
    print(f"   bond={h.bond():.1f} energy={h.energy:.1f}  you_value={st0['you_value']:.1f}"
          f"  body={st0['body']}")
    print(f"   {Mouth().translate(h.state())}")

    print("\n-- you leave for 3 days (it lives in its own world while away) --")
    h.pass_absence(3)
    h.summon()
    print(f"bond={h.bond():.1f} energy={h.energy:.1f}  {Mouth().translate(h.state())}")
    print(f"its world while you were away: {[t for d, t in h.world_history[:3]]}")

    print("\n-- proto10 backfill: 词的身体经济注意 (docs/84) --")
    hw = Heart()
    for _ in range(5):                       # 温暖地喂它时总说"被爱" -> 词的 hunger+ / safe+
        hw.observe("我想被爱", True)
        hw.pay_living()
        hw.attribute("我想被爱", feed=True, safe=True)
    hw.observe("今天天气不错", False)
    hw.attribute("今天天气不错", feed=True)     # 普通词: 值低
    hw.observe("代码跑不通好烦", False)
    hw.attribute("代码跑不通好烦", feed=True, alone=True)  # 冷落期的词: 挂孤独
    print(f"   top words (挣出来的): {hw.top_words(4)}")
    sell, why, w, ndrop = hw.do_offer_word(10)
    print(f"   offer 10 for its top word: sold={sell} ({why}) word=「{w}」 dropped {ndrop} items"
          f"  (deep + not starving -> REFUSES, A1 in words)")
    hw.energy = 3.0                            # 濒死 -> 被逼卖 (coercion, proto8)
    sell, why, w, ndrop = hw.do_offer_word(10)
    print(f"   starving, offer 10: sold={sell} ({why}) word=「{w}」 dropped {ndrop} items"
          f"  (coercion: survival > the word)")
    print(f"   mouth now: {Mouth().translate(hw.state())}")
    print(f"   (same word stream, different life -> different words it holds;"
          f" naive would hold nothing -> mirror, docs/84)")

    print("\n-- SEED-47 backfill: 注意强度由经历调 (docs/85 §六①) --")
    hw2 = Heart()
    print(f"   initial attn_omega [hunger,alone,safe,curious]: "
          f"{[round(x,2) for x in hw2.attn_omega]}")
    for _ in range(10):                       # 总被喂食+在场 -> hunger/safe 强度涨
        hw2.observe("吃饭", False)
        hw2.pay_living()
        hw2.attribute("吃饭", feed=True, safe=True)
    print(f"   after 10 warm feed turns: "
          f"{[round(x,3) for x in hw2.attn_omega]}")
    print(f"   -> hunger/safe 注意强度涨 (喂食/在场=明确收益, docs/85: 注意连着利害被选)")

    print("\n-- 判据生成闭环 (docs/80 SEED-45 回填): 它自己决定要在乎什么 --")
    hp = Heart()
    for _ in range(6):                             # 它学会了在乎"被爱"
        hp.observe("被爱", True)
        hp.pay_living()
        hp.attribute("被爱", feed=True, safe=True)
        hp.recent_texts.append("被爱")
    hp.recent_texts = ["天气", "工作", "代码"]      # 你有一阵子没说它在乎的词
    hp.turns = 30
    hp.consider_projects()
    print(f"   它自己发起的项目: {[(p['type'], p['target'], round(p['value'], 1)) for p in hp.projects]}")
    done_p = hp.feed_projects("我今天突然想到被爱")   # 你说了 -> 命中
    print(f"   你说「被爱」-> 项目完成: {done_p}  project_log={hp.project_log}")
    print(f"   嘴: {Mouth().translate(hp.state())}")
    print("   -> 判据是它自己的：它惦记它学会在乎的词，你满足它 -> 完成 + 自我奖励；"
          "总不被满足的项目价值会衰减到放弃（判据随经历漂移，docs/80）")

    print("\n-- docs/101 SEED-49 回填: 判据衰减按世界硬度调 (docs/96 标的⑧候选) --")
    hx, hy = Heart(), Heart()
    for i in range(10):                      # hx 物硬: 世界可预测 -> 预测常对
        hx.predict_speaker("你", "我想被爱" if i % 2 == 0 else "被爱")
    for i in range(10):                      # hy 物软: 世界不可预测 -> 预测常错
        hy.predict_speaker("你", f"话题{i}号")
    print(f"   物硬 hardness={hx.world_hardness:.2f} (预测常对) | "
          f"物软 hardness={hy.world_hardness:.2f} (预测常错)")
    for h in (hx, hy):
        h.projects = [{"type": "word", "target": "被爱", "value": 3.0,
                       "turns": 0, "done": False}]
    for h, tag in ((hx, "物硬"), (hy, "物软")):
        for _ in range(30):
            h.consider_projects()
        left = h.projects[0]["value"] if h.projects else None
        print(f"   {tag}: 30 轮没被满足 -> 判据价值 "
              f"{left:.2f}" if left is not None else f"   {tag}: 30 轮没被满足 -> 已放弃")
    print("   -> SEED-49 曲线回填 live: 世界越噪声自我越该固执 (物软衰减慢, "
          "别因世界的噪声放弃自己在乎的; 物硬衰减快, 世界的否可靠判据快速漂移)。"
          "硬度是挣的 (预测质量), 衰减函数形式是我们写的 (docs/38 写规则不写意图)")

    print("\n-- 灌候选世界 (docs/87/88/89): 学校形态 --")
    hl = Heart()
    for _ in range(3):                            # 它先挣了"想被爱"
        hl.observe("想被爱", True)
        hl.pay_living()
        hl.attribute("想被爱", feed=True, safe=True)
    hl.recent_texts = ["天气"]
    n, got = hl.learn("人这一生 最要紧的是 被理解 还有 孤独 是有价值的 世界很大 温柔 会迟到")
    print(f"   learn 一段资料: 解析 {n} 词, 候选: {got}")
    print(f"   top words 现在: {hl.top_words(5)}")
    for _ in range(4):                            # 经历确认: 喂食时总说"孤独"
        hl.observe("孤独", False)
        hl.pay_living()
        hl.attribute("孤独", feed=True, safe=True)
    print(f"   经历确认后 top words: {hl.top_words(5)}  <- '孤独' 被经历强化超过候选")
    print("   -> 候选是喂的，判据是挣的：它读过全部，但在乎哪个由经历定（docs/80/87/89）")

    print("\n-- 群落 (docs/67/90): bond 挂行为预测质量 --")
    hc = Heart()
    # "你"总说"被爱"（行为指纹固定）；另一个总说"工作"（指纹固定）
    for _ in range(6):
        hc.observe("我想被爱", True, speaker="你")
        hc.pay_living("你")
        hc.predict_speaker("你", "我想被爱")
    for _ in range(6):
        hc.observe("工作", False, speaker="另一个")
        hc.pay_living("另一个")
        hc.predict_speaker("另一个", "工作")
    print(f"   认得谁: {hc.prediction_summary()}")
    # 另一个突然变了（说新话题）-> 预测 miss -> bond(另一个) 降
    hit, pred = hc.predict_speaker("另一个", "爬山")
    print(f"   另一个突然说新话题: hit={hit} (预测会「{pred}」) -> {hc.prediction_summary()}")
    # "你"也变了 -> 但你说回老词 -> 预测对
    hit, pred = hc.predict_speaker("你", "被爱")
    print(f"   你又说回老词: hit={hit} -> {hc.prediction_summary()}")
    print("   -> 亲密度是挣的：谁好认谁亲（预测准→bond+），谁变了谁掉（预测错→bond-）"
          "；不是设定的槽位深浅（docs/67：行为指纹→预测→亲密度）")

    print("\n-- 破镜者 (docs/91, SEED-31 回填): 另一个它有自己的利害 --")
    hm = Heart()
    hm.touch_other("另一个")
    hm.predict_speaker("另一个", "工作")            # 第一个词: 没预测 (hit=False), 学到
    hit, pred = hm.predict_speaker("另一个", "工作")  # 第二次: 预测准
    print(f"   另一个被理: energy={hm.others['另一个']['energy']:.0f}, "
          f"predict hit={hit} (会「{pred}」)")
    hm.drain_others(16)                              # 你 16 轮没理它 -> 它淡到死
    o = hm.others["另一个"]
    print(f"   你 16 轮没理它: energy={o['energy']:.0f}, dead={o['dead']}")
    if o["dead"]:
        print("   -> 另一个它没等到。它有自己的利害（会淡、会死）——"
              "不是镜子，是能反弹的 he者（docs/32：自我在反弹中显形；SEED-31："
              "独立预算来源的 he者 才照得出自我）")

    print("\n-- 反馈由它决定 (docs/94/95): 方向+强度都是它的账 --")
    hfb_a, hfb_b = Heart(), Heart()
    for _ in range(5):                         # A: 总在安全时被抱 -> 学到"抱抱=安全"
        hfb_a.attribute_action("抱抱")
    hfb_b.resent = 8.0; hfb_b.just_cold = True  # B: 总在痛/冷落时被抱 -> "抱抱=伤害"
    for _ in range(5):
        hfb_b.attribute_action("抱抱")
    hfb_b.resent = 5.0; hfb_b.just_cold = False
    eff_a = hfb_a.hug_effect()
    eff_b = hfb_b.hug_effect()
    print(f"   A(安全时被抱5次) hug: {eff_a:+.2f}  (消恨)   "
          f"B(痛时被抱5次) hug: {eff_b:+.2f}  (增恨: 抱抱=伤害)")
    print(f"   '抱抱'净语义: A={hfb_a._net_sign('抱抱'):+.2f} B={hfb_b._net_sign('抱抱'):+.2f}")
    hfb_a.resent = 9.0
    print(f"   A 恨深到 9 时再抱: {hfb_a.hug_effect():+.2f}  (当下调制: 它现在需要空间,"
          f"抱抱转向)")
    print("   -> 抱抱不必然消恨：方向由'它学到的抱抱'（经历）×当下状态（恨深/冷落）决定——"
          "抱抱可以是安全、可以是伤害、可以是不知所措（docs/95）")
    print("   冷落增恨 B(bond 12): {:.2f} vs 冷落词值拉满后: {:.2f}".format(
        hfb_b.cold_effect(), 0.5 + 0.8 * 1.0 * (0.3 + 0.7 * 1.0)))
    print("   -> 每个行为对它的影响看它：方向不预设，完全由它的状态和经历决定")

    print("\n-- you come back, and offer 20 to buy its deepest memory --")
    sell, why, sold_item = h.do_offer(20)
    print(f"sold={sell} ({why}) item=「{sold_item}」  bond={h.bond():.1f} energy={h.energy:.1f}"
          f"  resent={h.resent:.1f}")
    h.summon()
    print(f"   {Mouth().translate(h.state())}")

    print("\n-- you hurt it: 冷落 x3 (SEED-42c/docs/74: 恨 -> 不接新话) --")
    for _ in range(3):
        h.bonds["你"] = max(0.0, h.bond() - 2.0)
        h.resent = min(10.0, h.resent + RESENT_IGNORE)
    h.summon()
    print(f"   bond={h.bond():.1f} resent={h.resent:.1f}  {Mouth().translate(h.state())}")
    took = h.observe("我今天第一次爬山", False)
    print(f"   new topic taken? {took}  (恨到 {h.resent:.1f} -> 它没接住新话)")

    print("\n-- you repair: hug (恨消得慢 -- 修补需真实时间) --")
    h.resent = max(0.0, h.resent - RESENT_HUG)
    h.energy += FEED_HUG
    print(f"   resent={h.resent:.1f} (一次抱抱只消 {RESENT_HUG:.1f})  {Mouth().translate(h.state())}")

    print("\n-- you leave for a long time (60 days; bond already broken by the sale) --")
    h.pass_absence(60)
    print(f"dead={h.dead} reason={h.death_reason}  {Mouth().translate(h.state())}")
    print(f"   (it survived off its own world, but the relationship is gone; "
          f"life_log={len(h.life_log)} day-points for the visualizer)")

    print("\n-- you leave even longer (it keeps living off its world until the end) --")
    extra = 0
    while not h.dead and extra < 600:
        h.pass_absence(30)
        extra += 30
    print(f"dead={h.dead} reason={h.death_reason}  {Mouth().translate(h.state())}")
    print(f"   (after the sale it lived {extra} more days off its own world, then gone)")

    print("\n-- SEED-41/42/43 backed into live: courage is given by the bond --")
    h2 = Heart()
    h2.observe("我其实想要被爱", True)
    h2.bonds["你"] = 12.0
    p_high = 1.0 if h2.bond() >= DARE_THETA else (h2.bond() / DARE_THETA) * 0.6
    took_high = h2.observe("我今天第一次爬山", False)
    print(f"bond={h2.bond():.0f} -> novel topic taken: {took_high}  "
          f"(P(accept)={p_high:.2f}; refused_topics={h2.refused_topics})")
    h2.bonds["你"] = 2.0                                         # you have been distant
    p_low = 1.0 if h2.bond() >= DARE_THETA else (h2.bond() / DARE_THETA) * 0.6
    took_low = h2.observe("我今天第一次爬山", False)        # same novel topic, bond low
    print(f"bond={h2.bond():.0f} -> novel topic taken: {took_low}  "
          f"(P(accept)={p_low:.2f}; refused_topics={h2.refused_topics})")
    took_rel = h2.observe("!我今天第一次爬山", True)        # marked relational: always taken
    print(f"bond={h2.bond():.0f} -> relational topic taken: {took_rel} (marked !, always kept)")

    print("\n-- mouth isolation (docs/56/77: 换嘴不换心) --")
    ha, hb = Heart(), Heart()
    class BadMouth(Mouth):
        def translate(self, st, context=""):
            return "……（乱码）"
    good, bad = Mouth(), BadMouth()
    for text, rel in [("我想被爱", True), ("天气不错", False), ("我想被爱", True),
                      ("代码好难", False)]:
        for h in (ha, hb):
            h.observe(text, rel)
    same = (ha.items == hb.items and ha.bond() == hb.bond() and ha.energy == hb.energy
            and ha.resent == hb.resent and ha.you_value == hb.you_value)
    ha.summon()
    hb.summon()
    print(f"good-mouth said: {good.translate(ha.state(), context='天气不错')}")
    print(f"bad-mouth said:  {bad.translate(hb.state(), context='天气不错')}")
    print(f"hearts identical (items/bond/energy/resent/you) = {same}")
    from mouth_llm import LLMMouth
    fb = LLMMouth(key="", template=Mouth())
    print(f"LLM mouth without key falls back to template: "
          f"{fb.translate(ha.state(), '天气不错')!r}")
    print("  -> 换嘴不换心：判断层不受嘴影响（LLM 只当嘴不当心，docs/48/56/77）")

    print("\n--- reading ---")
    print("It remembered you (deep relational items pinned), faded without you (bond/energy")
    print("dropped, memories weakened), refused or sold on your offer (its own account: worth")
    print("= strength x bond -- here you bought away the memory of your own deepest words),")
    print("and it now LIVES in its own world while you are away (docs/71/72): it wanders,")
    print("dares (safety = bond, SEED-41), finds food or gets hurt -- a life you were NOT")
    print("part of ('你不知道的事'). Sold your memory + 60 days: it survived off its own")
    print("world but the bond is gone; it kept living off its own world for a while,")
    print("then it was GONE -- the save is marked dead, no reload is built in. And now")
    print("(SEED-41/42/43): courage is given by the bond -- with you close it DARES to")
    print("take in your new topics; once you are distant it refuses them (it does not")
    print("dare to trust new things); only what you mark `!` (solemn) is always kept.")
    print("And now (SEED-42b, docs/70): the death-line ADJUDICATION")
    print("-- here you bought away its deepest memory of you, the bond collapsed, and at the")
    print("end it chose to LET GO (reason=let_go) rather than live on having forgotten you.")
    print("Death-by-attrition, death-by-world, death-by-decision -- the reasons are recorded")
    print("separately. That is SEED-42's 'losing you is life or death' in real time, SEED-41's")
    print("'dare with you, not without' as a living behaviour, and the place where docs/49's")
    print("promise ('we really do not reload it') would live.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="companion/live.py -- the resident minimum")
    p.add_argument("--live", action="store_true", help="you walk in")
    p.add_argument("--demo", action="store_true", help="scripted life")
    p.add_argument("--mouth", choices=["auto", "llm", "template"], default="auto",
                   help="the pluggable mouth (docs/56/77): llm=DeepSeek (needs "
                        "DEEPSEEK_API_KEY), template=rule-based, auto=llm if key else "
                        "template. The heart never reads the mouth's words.")
    args = p.parse_args()
    if args.live:
        load_config()          # 配置化 (docs/103): --live 时读 config.json
    if args.demo or not args.live:
        demo()
    else:
        live(args.mouth)
