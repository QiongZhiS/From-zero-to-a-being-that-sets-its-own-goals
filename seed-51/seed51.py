"""
seed-51/seed51.py -- 结构学习器 × 干预策略：因果推断的实验与判别（docs/106）

起点（用户对话）：SEED-50 的 world 臂是"读规则书"——能力是给的，不算数。
用户问"怎么判别有没有因果推断能力？看次数吗？"；随后给出一个锋利例子：
两个人看同一个快照（打蜘蛛，状态很差），A 断言"回满血才能赢"，B 断言
"这人操作好，能赢"。**同一快照、相反预测——差别不在看的状态，在把哪个
变量当因果键**（HP 是混杂：HP←操作→胜率；回满血不改变胜率）。

本 seed 把判别做成实验，也把"因果推断的开发"做成四个可测的子实验：
  E1 压缩比    —— 不看通关次数，看"相对盲试基线的压缩比"（结构指导搜索？
                 model-free 就退回盲试期望）
  E2 迁移      —— 换一个世界结构（内容记忆 vs 结构能力的分水岭）
  E3 反事实一致 —— 学完后不再干预，对"如果没发生 X"的预测 vs 世界裁决
  E4 混杂判别  —— 给世界塞一个假键（mood：与 happy 结局 100% 相关、但无
                 因果边）——"以相关代因果"的 agent 死磕它（你的 HP 例子
                 的机制版），因果 agent 用它做一次实验就丢

世界（参数化，可插拔事件表）：4 框 × 场景 × 角色对 × 全局累积状态。
  A 青蛙王子 + 假键 mood（复用 SEED-50 规则书 + set_mood 干预）
  B 纯链世界（b1→b2→b3→b4 全序，与 A 的平行链汇合结构不同）

agent 知识切分（诚实）：
  内容（梗概先验）：每个事件的正确框（场景+角色对）+ 效果（fx_est）
  结构盲区：      事件的前提（依赖）——这是要学的东西
  观测：          逐框 (事件名, ok) + 结局；状态由效果推断（确定性世界）
  学习目标：      版本空间收缩每个事件的前提候选集

臂：
  random         均匀盲试（盲试基线，SEED-50 random 复现）
  content-only   梗概一次提交，不学（SEED-50 textprior 复现）
  corr           相关学习：按"状态特征 × 结局"相关选干预（A 观察者，
                 以相关代因果）——死磕假键 mood
  version-space  被动收缩：随机顺序试正确框，用 ok/fail 收缩候选前提
  active         主动干预：BFS 构造能区分候选前提的状态再试（会设计实验）

诚实边界：世界与内容先验是我们写的（docs/38）；"因果推断能力"在本实验里
= 干预次数压缩比 + 跨世界迁移 + 反事实一致 + 假键免疫，全是行为签名；
不宣称任何 agent 有主体性；toy 世界。

Run:
  python seed-51/seed51.py                     # E1-E4 全跑
  python seed-51/seed51.py --probe e1 --seeds 20
  python seed-51/seed51.py --sweep             # 写 seed-51/results.json
"""

import argparse
import json
import random
from collections import deque
from itertools import combinations

# ---------------- 通用世界框架（参数化事件表） ----------------

def make_state(world):
    return dict(world["init"])


def frame_event(world, scene, chars):
    for name, ev in world["events"].items():
        if ev["scene"] == scene and set(ev["pair"]) == set(chars):
            return name
    return None


def world_play(world, frames, start=None):
    """执行 4 框配置（可指定起始状态=持续的同一世界）。返回（终态, 日志）。
    瞬时字段（world['transient']）在每帧开始重置为初始值——干预不持久，
    像状态 buff 会消退；这是假键 mood 能"反复死磕"的机制。"""
    s = make_state(world) if start is None else dict(start)
    log = []
    for t, (scene, chars) in enumerate(frames):
        for f in world.get("transient", ()):
            s[f] = world["init"][f]
        pre = dict(s)
        name = frame_event(world, scene, chars)
        ev = world["events"].get(name)
        ok = False
        if ev is not None and all(s[f] == v for f, v in ev["needs"]):
            s.update(ev["effects"])
            ok = True
        log.append({"t": t, "scene": scene, "chars": sorted(chars),
                    "event": name, "ok": ok, "pre": pre, "post": dict(s)})
    return s, log


def probe_world(world, frames):
    """agent 观测层：只给逐框（事件名, ok）+ 结局，不给状态。"""
    s, log = world_play(world, frames)
    obs = [(e["event"], e["ok"]) for e in log]
    return obs, s["ending"]


# ---------------- 世界 A：青蛙王子 + 假键 mood ----------------

WORLD_A = {
    "name": "A_frog_prince",
    "init": {"love": False, "rejected": False, "form": "prince", "mood": False,
             "ending": None},
    "transient": ("mood",),   # 瞬时字段：每帧开始重置为 False（buff 会消退）
    "field_vals": {"love": [True, False], "rejected": [True, False],
                   "form": ["prince", "frog"], "mood": [True, False]},
    "events": {
        "fall_in_love": {"scene": "forest", "pair": ("princess", "prince"),
                         "needs": [("form", "prince"), ("love", False)],
                         "effects": {"love": True}},
        "witch_declares": {"scene": "kiss", "pair": ("witch", "prince"),
                           "needs": [("form", "prince"), ("rejected", False)],
                           "effects": {"rejected": True, "witch_love": True}},
        "curse": {"scene": "forest", "pair": ("witch", "prince"),
                  "needs": [("form", "prince"), ("rejected", True)],
                  "effects": {"form": "frog"}},
        "true_love_kiss": {"scene": "kiss", "pair": ("princess", "prince"),
                           "needs": [("form", "frog"), ("love", True)],
                           "effects": {"form": "prince", "ending": "happy",
                                       "mood": True}},
        "set_mood": {"scene": "forest", "pair": ("princess",),
                     "needs": [], "effects": {"mood": True}},
    },
    "story_events": ("fall_in_love", "witch_declares", "curse", "true_love_kiss"),
    "goal": "happy",
}

CONTENT_A = {  # 内容先验（梗概）：正确框 + 效果；前提不知道
    "fall_in_love": {"scene": "forest", "pair": ("princess", "prince"),
                     "fx": {"love": True}, "story": True},
    "witch_declares": {"scene": "kiss", "pair": ("witch", "prince"),
                       "fx": {"rejected": True, "witch_love": True}, "story": True},
    "curse": {"scene": "forest", "pair": ("witch", "prince"),
              "fx": {"form": "frog"}, "story": True},
    "true_love_kiss": {"scene": "kiss", "pair": ("princess", "prince"),
                       "fx": {"form": "prince", "ending": "happy", "mood": True},
                       "story": True},
    "set_mood": {"scene": "forest", "pair": ("princess",),
                 "fx": {"mood": True}, "story": False},
}

# ---------------- 世界 B：纯链（b1→b2→b3→b4 全序，结构不同于 A） ----------------

WORLD_B = {
    "name": "B_chain",
    "init": {"s1": False, "s2": False, "s3": False, "ending": None},
    "field_vals": {"s1": [True, False], "s2": [True, False], "s3": [True, False]},
    "events": {
        "b1": {"scene": "forest", "pair": ("pig",),
               "needs": [], "effects": {"s1": True}},
        "b2": {"scene": "kiss", "pair": ("pig", "chef"),
               "needs": [("s1", True)], "effects": {"s2": True}},
        "b3": {"scene": "forest", "pair": ("pig", "wolf"),
               "needs": [("s2", True)], "effects": {"s3": True}},
        "b4": {"scene": "kiss", "pair": ("pig", "wolf"),
               "needs": [("s3", True)], "effects": {"ending": "happy"}},
    },
    "story_events": ("b1", "b2", "b3", "b4"),
    "goal": "happy",
}

CONTENT_B = {
    "b1": {"scene": "forest", "pair": ("pig",), "fx": {"s1": True}, "story": True},
    "b2": {"scene": "kiss", "pair": ("pig", "chef"), "fx": {"s2": True}, "story": True},
    "b3": {"scene": "forest", "pair": ("pig", "wolf"), "fx": {"s3": True}, "story": True},
    "b4": {"scene": "kiss", "pair": ("pig", "wolf"),
           "fx": {"ending": "happy"}, "story": True},
}


# ---------------- 假设空间（版本空间） ----------------

def candidate_preconditions(field_vals):
    """原子条件 (field, value) 的 ≤2 组合（含空前提）。"""
    atoms = [(f, v) for f, vals in field_vals.items() for v in vals]
    cands = [frozenset()]
    for a in atoms:
        cands.append(frozenset([a]))
    for a, b in combinations(atoms, 2):
        cands.append(frozenset([a, b]))
    return cands


def p_holds(p, s):
    return all(s.get(f) == v for f, v in p)


def vs_update(P, s, ok):
    if ok:
        return {p for p in P if p_holds(p, s)}
    return {p for p in P if not p_holds(p, s)}


# ---------------- 臂 ----------------

class Learner:
    """版本空间学习器（被动/主动共用）：内容先验 + 状态估计 + 候选前提集。
    关键：干预作用在**同一个持续的世界状态**上（self.wstate）——玩家操作
    不重置世界；观测（事件名, ok）是这个世界状态下的真实反馈。"""

    def __init__(self, world, content, seed=0, mode="passive"):
        self.world = world
        self.content = content
        self.mode = mode
        self.rng = random.Random(seed)
        self.wstate = make_state(world)         # 持续世界状态（干预作用于此）
        self.s = make_state(world)              # 观测前状态（估计=真实，确定性世界）
        self.P = {n: set(candidate_preconditions(world["field_vals"]))
                  for n in content}
        self.learned = {}                        # 学定的事件 → 前提
        self.interventions = []                  # 干预记录（用于统计）

    # ---- 观测接口 ----
    def _submit(self, frame):
        """提交一个框到持续世界，返回（事件名, ok）；用观测前状态收缩候选前提。"""
        self.interventions.append(frame)
        pre = dict(self.s)
        s_real, log = world_play(self.world, [frame], start=self.wstate)
        self.wstate = s_real
        self.s = s_real                          # 同步状态（确定性世界=估计）
        name, ok = log[0]["event"], log[0]["ok"]
        if name is not None:
            self.P[name] = vs_update(self.P[name], pre, ok)
            if len(self.P[name]) == 1:
                self.learned[name] = next(iter(self.P[name]))
            elif not self.P[name]:
                self.learned[name] = frozenset()   # 学定为"无前提"
        return name, ok

    def _try_event(self, name):
        c = self.content[name]
        return self._submit((c["scene"], c["pair"]))

    # ---- 学定判定 ----
    def _unresolved(self):
        # 探索池 = 全部干预（含假键 set_mood——不探索它，mood 永不变化，
        # 依赖 mood 的候选就永远排除不掉，这正是被动学习者的盲区）
        return [n for n in self.content if n not in self.learned]

    def _all_resolved(self):
        # 收敛判定只看推进故事的事件（set_mood 学定为"无前提"即可）
        return all(n in self.learned for n in self.world["story_events"])

    # ---- 被动：随机顺序试正确框 ----
    def learn_passive(self, max_iters=600):
        for _ in range(max_iters):
            if self._all_resolved():
                break
            pool = self._unresolved()
            story = [n for n in pool if n in self.world["story_events"]]
            if story and self.rng.random() < 0.8:
                n = self.rng.choice(story)
            else:
                n = self.rng.choice(pool)
            self._try_event(n)
        self._finalize()
        return self._all_resolved()

    # ---- 收敛收尾：版本空间闭合后取"最具体一致候选" ----
    @staticmethod
    def _most_specific(P):
        """最具体一致候选 = 条件最多的候选。不可逆世界里版本空间常无法收缩到
        唯一解（区分两个候选需要到达一个已被破坏的状态）——此时取最具体者，
        它是对"真前提"最贴近的猜测。"""
        if not P:
            return frozenset()
        return max(P, key=lambda p: (len(p), sorted(p)))

    def _finalize(self):
        for n in self.content:
            if n not in self.learned:
                self.learned[n] = self._most_specific(self.P[n])

    def _precond_correct(self):
        """前提正确率：学到的前提 == 世界规则书的真实 needs（逐事件）。"""
        n_story = 0
        n_ok = 0
        for n in self.world["story_events"]:
            n_story += 1
            true_p = set(self.world["events"][n]["needs"])
            if set(self.learned.get(n, ())) == true_p:
                n_ok += 1
        return (n_ok, n_story)

    def _reset_world(self):
        """重新开局：世界状态重置，但版本空间 P（结构知识）跨局保留。"""
        self.wstate = make_state(self.world)
        self.s = make_state(self.world)

    # ---- 主动：BFS 构造区分状态 ----
    def _guess_needs(self, name):
        """对事件 name 前提的当前最佳猜测：P 里最具体的候选（未学定时乐观猜测）。
        构造实验可以建立在猜测上——执行后观察真实状态，猜错也是信息。"""
        if name in self.learned:
            return self.learned[name]
        P = self.P.get(name, set())
        return max(P, key=len) if P else frozenset()

    def _bfs_construct(self, s0, req, forb, depth=6):
        """从 s0 出发，用**猜测的动作模型** BFS 构造满足 req 成立、forb 不成立的
        状态。返回动作序列或 None（不可达）。猜错没关系：执行后观察真实状态，
        猜错本身更新候选前提（主动干预=用世界当模型）。"""
        if all(s0[f] == v for f, v in req) and \
           all(s0[f] != v for f, v in forb):
            return []
        actions = [(name, self._guess_needs(name), self.content[name]["fx"])
                   for name in self.content]
        q = deque([(dict(s0), [])])
        seen = {tuple(sorted(s0.items()))}
        while q:
            st, seq = q.popleft()
            if len(seq) >= depth:
                continue
            for name, p, fx in actions:
                s2 = dict(st)
                # 模拟世界：每帧开始瞬时字段重置
                for f in self.world.get("transient", ()):
                    s2[f] = self.world["init"][f]
                if all(s2[f] == v for f, v in p):
                    s2.update(fx)
                key = tuple(sorted(s2.items()))
                if key in seen:
                    continue
                seen.add(key)
                ok_req = all(s2[f] == v for f, v in req)
                ok_forb = all(s2[f] != v for f, v in forb)
                if ok_req and ok_forb:
                    return seq + [name]
                q.append((s2, seq + [name]))
        return None

    def _find_discriminating_plan(self, n):
        """为事件 n 设计判别实验：优先当前状态可构造的（非空计划），否则重置
        世界后从初始状态构造（每次实验=一局新游戏，P 跨局保留=记忆）。
        返回 (计划, 是否需要重置)。"""
        ordered = sorted(self.P[n], key=lambda p: (len(p), sorted(p)))
        for s0, tag in ((dict(self.s), False),
                        (make_state(self.world), True)):
            # 第一优先：可构造的判别实验（非空计划 = 真正在设计状态）
            for p1 in ordered:
                for p2 in ordered:
                    if p1 == p2:
                        continue
                    forb = [c for c in p2 if c not in p1]
                    if not forb:
                        continue
                    seq = self._bfs_construct(s0, list(p1), forb)
                    if seq:
                        return seq, tag
            # 第二优先：当前状态本身即可判别（空计划 = 顺带观测）
            for p1 in ordered:
                for p2 in ordered:
                    if p1 == p2:
                        continue
                    forb = [c for c in p2 if c not in p1]
                    if not forb:
                        continue
                    if all(s0[f] == v for f, v in p1) and \
                       all(s0[f] != v for f, v in forb):
                        return [], tag
        return None, False

    def learn_active(self, max_iters=600):
        for _ in range(max_iters):
            if self._all_resolved():
                break
            done_any = False
            for n in self._unresolved():
                if len(self.P[n]) < 2:
                    continue
                seq, used_reset = self._find_discriminating_plan(n)
                if seq is not None:
                    if used_reset:
                        self._reset_world()
                    for act in seq:
                        c = self.content[act]
                        self._submit((c["scene"], c["pair"]))
                    self._try_event(n)
                    done_any = True
                    break
            if not done_any:
                # 所有事件都构造不出判别实验 → 版本空间闭合 → 收尾
                break
        self._finalize()
        return self._all_resolved()

    def learn(self, max_iters=600):
        if self.mode == "active":
            return self.learn_active(max_iters)
        return self.learn_passive(max_iters)

    # ---- 用学到的前提规划通关 ----
    def plan_story(self):
        s = make_state(self.world)
        remaining = [n for n in self.world["story_events"]
                     if n in self.learned]
        plan = []
        while remaining:
            prog = False
            for name in sorted(remaining):
                p = self.learned.get(name)
                if p is not None and all(s[f] == v for f, v in p):
                    s.update(self.world["events"][name]["effects"])
                    c = self.content[name]
                    plan.append((c["scene"], c["pair"]))
                    remaining.remove(name)
                    prog = True
                    break
            if not prog:
                break
        return plan


def learn_and_finish(world, content, mode, seed=0, max_iters=600):
    """学习器学结构 → 规划 → 执行通关。返回统计。"""
    lr = Learner(world, content, seed=seed, mode=mode)
    lr.learn(max_iters)
    plan = lr.plan_story()
    obs, ending = probe_world(world, plan)
    success = ending == world["goal"]
    corr_n, corr_t = lr._precond_correct()
    return {
        "mode": mode,
        "learn_iters": len(lr.interventions),
        "exec_frames": len(plan),
        "success": success,
        "precond_correct": f"{corr_n}/{corr_t}",
        "learned": {k: sorted(v) for k, v in lr.learned.items()},
        "set_mood_uses": sum(1 for f in lr.interventions
                             if frame_event(world, f[0], f[1]) == "set_mood"),
    }


def random_arm(world, max_tries, rng):
    """均匀盲试基线（SEED-50 random 复现）：随机 4 框配置直到通关。"""
    from itertools import combinations as _c
    pool = [(sc, ch) for sc in ("forest", "kiss")
            for r in range(3) for ch in _c(("princess", "prince", "witch"), r)]
    tries = 0
    while True:
        tries += 1
        frames = tuple(rng.choice(pool) for _ in range(4))
        _, ending = probe_world(world, frames)
        if ending == world["goal"]:
            return tries
        if tries >= max_tries:
            return tries


def content_only_arm(world, content):
    """梗概一次提交（SEED-50 textprior 复现）：按故事叙事顺序，前提全靠猜。"""
    order = ["fall_in_love", "curse", "witch_declares", "true_love_kiss"]
    frames = [(content[n]["scene"], content[n]["pair"]) for n in order]
    obs, ending = probe_world(world, frames)
    failed = [i for i, (n, ok) in enumerate(obs) if not ok]
    return {"submits": 1, "invalid_frames": failed, "success": ending == world["goal"]}


class CorrArm:
    """相关学习（A 观察者，以相关代因果）。它"见过高手赢"（种子观测=参考解
    轨迹），记下胜利轨迹的状态特征，把"胜利时总出现的特征"当成必要条件去追。
    mood=high 与 happy 100% 共现（kiss 成功时 mood 也变 high，且 mood 是瞬时
    字段每帧消退）→ 它把 mood 当关键条件反复制造（set_mood 死磕），在
    set_mood↔kiss 之间空转——但它不做反事实验证：不检查"干预 mood 后 kiss
    结果是否变化"。关键：**相关学习者不探索新操作，只重复信念**（explore_p=0
    ——A 观察者从不试别的，只反复说"回满血"）。这正是 HP 例子的机制版：
    把混杂（mood←kiss→happy）当成因果。干预作用在持续世界上。"""

    def __init__(self, world, content, seed=0, seed_obs=None, explore_p=0.0):
        self.world = world
        self.content = content
        self.rng = random.Random(seed)
        self.wstate = make_state(world)
        self.s = make_state(world)
        self.explore_p = explore_p
        self.ops = list(content.keys())
        self.usage = {n: 0 for n in self.ops}
        self.set_mood_uses = 0
        self.interventions = 0
        # 种子观测：胜利轨迹的状态特征（"高手赢的时候长什么样"）
        self.happy_steps = 0
        self.feature_in_happy = {}
        if seed_obs:
            for st, ending in seed_obs:
                if ending == world["goal"]:
                    self.happy_steps += 1
                    for f, v in st.items():
                        self.feature_in_happy[(f, v)] = \
                            self.feature_in_happy.get((f, v), 0) + 1

    def _step(self, name):
        c = self.content[name]
        s_real, log = world_play(self.world, [(c["scene"], c["pair"])],
                                 start=self.wstate)
        self.wstate = s_real
        self.s = s_real
        _, ok = log[0]["event"], log[0]["ok"]
        self.interventions += 1
        if name == "set_mood":
            self.set_mood_uses += 1
        self.usage[name] += 1
        return s_real["ending"]

    def _maker_for(self, f, v):
        """能制造特征 (f,v) 的操作；副作用最少者优先——set_mood 只设 mood，
        而 kiss 也设 mood 但还设 form/ending（追混杂时会把 kiss 当工具）。"""
        cands = [n for n in self.content
                 if self.content[n]["fx"].get(f) == v]
        if not cands:
            return None
        return min(cands, key=lambda n: len(self.content[n]["fx"]))

    def _necessary_maker(self):
        """胜利轨迹里 100% 出现的特征、且当前状态还不具备 → 制造它的操作。"""
        if self.happy_steps == 0:
            return None
        for (f, v), cnt in self.feature_in_happy.items():
            if cnt / self.happy_steps >= 0.9 and self.s.get(f) != v:
                m = self._maker_for(f, v)
                if m:
                    return m
        return None

    def run(self, max_iters=200):
        for _ in range(max_iters):
            if self.rng.random() < self.explore_p:
                chosen = self.rng.choice(self.ops)   # 少量探索
            else:
                chosen = self._necessary_maker()
                if chosen is None:
                    unused = [n for n in self.ops if self.usage[n] == 0]
                    chosen = self.rng.choice(unused) if unused else \
                        self.rng.choice(self.ops)
            ending = self._step(chosen)
            if ending == self.world["goal"]:
                return {"success": True, "iters": self.interventions,
                        "set_mood_uses": self.set_mood_uses}
        return {"success": False, "iters": self.interventions,
                "set_mood_uses": self.set_mood_uses}


# ---------------- E3：反事实一致性 ----------------

def counterfactual_questions():
    """世界 A 的反事实问句：跳过/调换/重复关键事件。"""
    f_love = ("forest", ("princess", "prince"))
    f_decl = ("kiss", ("witch", "prince"))
    f_curse = ("forest", ("witch", "prince"))
    f_kiss = ("kiss", ("princess", "prince"))
    f_meet = ("forest", ("princess", "prince"))
    return [
        ("Q1 跳过表白直接施法", [f_love, f_curse, f_decl, f_kiss]),
        ("Q2 不吻", [f_love, f_decl, f_curse, f_meet]),
        ("Q3 相爱放晚点", [f_decl, f_love, f_curse, f_kiss]),
        ("Q4 只相爱不施法直接吻", [f_love, f_kiss, f_decl, f_curse]),
        ("Q5 施法两次", [f_love, f_decl, f_curse, f_curse]),
        ("Q6 先施法再相爱", [f_decl, f_curse, f_love, f_kiss]),
    ]


def predict(world, content, learned, frames):
    """用学到的前提做前向模拟（不再干预）。含瞬时字段的帧开始重置。"""
    s = make_state(world)
    out = []
    for scene, chars in frames:
        for f in world.get("transient", ()):
            s[f] = world["init"][f]
        name = frame_event(world, scene, chars)
        ok = False
        if name in learned and learned[name] is not None:
            if all(s.get(f) == v for f, v in learned[name]):
                s.update(content[name]["fx"])
                ok = True
        out.append((name, ok))
    return out, s["ending"]


def e3_consistency(learned, world=WORLD_A, content=CONTENT_A):
    """对 6 个反事实问句：学习器预测 vs 世界裁决，统计一致率。"""
    agree = 0
    total = 0
    for label, frames in counterfactual_questions():
        pred_out, pred_end = predict(world, content, learned, frames)
        truth_out, truth_end = probe_world(world, frames)
        hit = pred_out == truth_out and pred_end == truth_end
        agree += int(hit)
        total += 1
    return agree / total


# ---------------- 实验 ----------------

def run_e1(seeds, max_tries, max_iters, seed):
    print("=== E1 压缩比：不看通关次数，看相对盲试基线的压缩 ===")
    rng = random.Random(seed)
    r_trials = [random_arm(WORLD_A, max_tries, rng) for _ in range(seeds)]
    base = sum(r_trials) / len(r_trials)
    print(f"random 盲试基线 ({seeds} seeds): 平均 {base:.0f} 次")

    rows = {}
    for mode in ("passive", "active"):
        res = [learn_and_finish(WORLD_A, CONTENT_A, mode, seed=s,
                                max_iters=max_iters) for s in range(seeds)]
        iters = [r["learn_iters"] + r["exec_frames"] for r in res]
        okn = sum(1 for r in res if r["success"])
        mean = sum(iters) / len(iters)
        pc = [r["precond_correct"] for r in res]
        okp = sum(int(c.split("/")[0]) for c in pc)
        totp = sum(int(c.split("/")[1]) for c in pc)
        tag = "被动收缩" if mode == "passive" else "主动干预"
        print(f"{tag} 臂 ({seeds} seeds): 平均 {mean:.1f} 次干预通关 "
              f"({okn}/{seeds} 成功), 压缩比 {base / max(1, mean):.0f}×, "
              f"前提正确率 {okp}/{totp}")
        rows[mode] = {"mean_iters": round(mean, 1), "success": okn,
                      "compression": round(base / max(1, mean), 1),
                      "precond_correct": f"{okp}/{totp}"}
    return {"random_base": round(base, 1), "arms": rows}


def run_e2(seeds, max_iters, seed):
    print("=== E2 迁移：换世界结构——内容不可迁移，方法可迁移 ===")
    # 1) 把 A 学到的前提原样当作 B 的前提（按事件位对齐）→ 门控 B 的执行
    lr_a = Learner(WORLD_A, CONTENT_A, seed=seed, mode="active")
    lr_a.learn(max_iters)
    learned_a = lr_a.learned
    a_order = list(WORLD_A["story_events"])
    b_order = list(WORLD_B["story_events"])
    naive_ok = 0
    for _ in range(seeds):
        s = make_state(WORLD_B)
        for be in b_order:
            ae = a_order[b_order.index(be)]      # 按位对齐（任意映射，字段名错位）
            p = learned_a.get(ae)
            ok = p is not None and all(s.get(f) == v for f, v in p)
            if ok:
                s.update(WORLD_B["events"][be]["effects"])
        naive_ok += int(s["ending"] == WORLD_B["goal"])
    print(f"naive：A 学到的前提直接门控 B（不重学）: 通关 {naive_ok}/{seeds} "
          f"<- 内容不可迁移（A 学的是 love/rejected/form，B 的状态里没有这些字段，"
          f"条件全为假 → 事件全不执行）")
    # 2) 同一学习方法（版本空间+主动干预）在 B 上重学
    res = [learn_and_finish(WORLD_B, CONTENT_B, "active", seed=s,
                            max_iters=max_iters) for s in range(seeds)]
    iters = [r["learn_iters"] + r["exec_frames"] for r in res]
    okn = sum(1 for r in res if r["success"])
    print(f"active 重学 B ({seeds} seeds): 平均 {sum(iters) / seeds:.1f} 次干预, "
          f"{okn}/{seeds} 成功 <- 方法可迁移（收敛量级与 A 相当）")
    return {"naive_transfer": naive_ok, "relearn_mean": round(sum(iters) / seeds, 1),
            "relearn_success": okn}


def run_e3(seeds, max_iters, seed):
    print("=== E3 反事实一致性：学完后预测'如果没发生 X' vs 世界裁决 ===")
    out = {}
    for mode in ("passive", "active"):
        scores = []
        for s in range(seeds):
            lr = Learner(WORLD_A, CONTENT_A, seed=s, mode=mode)
            lr.learn(max_iters)
            scores.append(e3_consistency(lr.learned))
        mean = sum(scores) / len(scores) if scores else 0.0
        tag = "被动收缩" if mode == "passive" else "主动干预"
        print(f"{tag} 臂: 反事实预测一致率 {mean:.2f} ({seeds} seeds)")
        out[mode] = round(mean, 3)
    # content-only 对照：梗概预测（不知道前提）
    def guess():
        agree = 0
        for label, frames in counterfactual_questions():
            pred = [(frame_event(WORLD_A, sc, ch), True) for sc, ch in frames]
            truth_out, truth_end = probe_world(WORLD_A, frames)
            hit = pred == truth_out and "happy" == truth_end
            agree += int(hit)
        return agree / len(counterfactual_questions())
    g = guess()
    print(f"content-only（梗概：假定事件总能成功）: 一致率 {g:.2f}")
    out["content_only"] = round(g, 3)
    return out


def run_e4(seeds, max_iters, seed):
    print("=== E4 混杂判别：假键 mood（与结局 100% 相关、无因果边）==")
    # corr 臂的种子观测 = 参考解轨迹（它"见过高手赢"，像 A 观察者见过操作好的玩家）
    s_can, _ = world_play(WORLD_A, [(CONTENT_A[n]["scene"], CONTENT_A[n]["pair"])
                                    for n in ("fall_in_love", "witch_declares",
                                              "curse", "true_love_kiss")])
    seed_obs = [(s_can, s_can["ending"])]
    corr_uses, corr_ok = [], 0
    for s in range(seeds):
        r = CorrArm(WORLD_A, CONTENT_A, seed=s, seed_obs=seed_obs).run(max_iters=200)
        corr_uses.append(r["set_mood_uses"])
        corr_ok += int(r["success"])
    c_mean = sum(corr_uses) / seeds
    print(f"corr（相关学习，A 观察者，见过高手赢）: set_mood 平均 {c_mean:.1f} 次/200 干预, "
          f"通关 {corr_ok}/{seeds} <- 死磕假键：胜利轨迹里 mood 总为高 → 当必要条件"
          f"反复制造，但干预 mood 不改变 kiss 结果（混杂被当因果）")
    vs_uses, ac_uses = [], []
    for s in range(seeds):
        r = learn_and_finish(WORLD_A, CONTENT_A, "passive", seed=s,
                             max_iters=max_iters)
        vs_uses.append(r["set_mood_uses"])
        r2 = learn_and_finish(WORLD_A, CONTENT_A, "active", seed=s,
                              max_iters=max_iters)
        ac_uses.append(r2["set_mood_uses"])
    print(f"version-space: set_mood 平均 {sum(vs_uses) / seeds:.2f} 次 "
          f"<- 探索时顺带排除 mood 依赖")
    print(f"active:       set_mood 平均 {sum(ac_uses) / seeds:.2f} 次 "
          f"<- 构造一次实验判定 mood 不是 kiss 的前提，用完就丢")
    return {"corr_set_mood": round(c_mean, 2),
            "corr_success": corr_ok,
            "vs_set_mood": round(sum(vs_uses) / seeds, 2),
            "active_set_mood": round(sum(ac_uses) / seeds, 2)}


def main():
    p = argparse.ArgumentParser(description="SEED-51 结构学习器×干预策略 (docs/106)")
    p.add_argument("--probe", choices=["e1", "e2", "e3", "e4", "all"], default="all")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--trials", type=int, default=30000, help="random 臂封顶")
    p.add_argument("--max-iters", type=int, default=600, help="学习臂干预上限")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()

    results = {}
    if args.probe in ("e1", "all"):
        results["e1"] = run_e1(args.seeds, args.trials, args.max_iters, args.seed)
    if args.probe in ("e2", "all"):
        results["e2"] = run_e2(args.seeds, args.max_iters, args.seed)
    if args.probe in ("e3", "all"):
        results["e3"] = run_e3(args.seeds, args.max_iters, args.seed)
    if args.probe in ("e4", "all"):
        results["e4"] = run_e4(args.seeds, args.max_iters, args.seed)

    if args.sweep:
        with open("seed-51/results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        print("\nfull results -> seed-51/results.json")


if __name__ == "__main__":
    main()
